import os
import json
import tqdm
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, Adagrad
from torch.optim.lr_scheduler import LambdaLR

from data_utils import load_word2vec, load_tacred_dataset, save_json
from model import CNNForRE, to_parallel, to_fp16, save_model

try:
    import apex
    from apex import amp
except ModuleNotFoundError:
    apex = None

def load_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word2vec", type=str, default=None)
    parser.add_argument("--glove", type=str, default=None)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--pad_token", type=str, default="[PAD]")
    parser.add_argument("--unk_token", type=str, default="[UNK]")
    parser.add_argument("--vocab", type=int, default=64000)

    parser.add_argument("--seq_len", type=int, default=100)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument('--fp16_opt_level', type=str, default="O1")

    parser.add_argument("--filter_num", type=int, default=150)
    parser.add_argument("--pos_emb_dim", type=int, default=50)
    parser.add_argument("--pos_dis_limit", type=int, default=50)
    parser.add_argument('--filters', nargs='*', default=[2,3,4,5])

    parser.add_argument("--num_workers", type=int, default=5)

    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--label_weights", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_batch_size",type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-3)

    parser.add_argument("--optim", choices=["adam", "adagrad"], default="adam")
    parser.add_argument("--lr_dec_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--adam_B1", type=float, default=0.9)
    parser.add_argument("--adam_B2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-6)

    parser.add_argument("--entity_mask", action="store_true")
    parser.add_argument("--emb_freeze", action="store_true")
    parser.add_argument("--emb_unfreeze", type=int, default=None)

    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    return parser.parse_args()

def args_to_dict(args):
    return {k:str(v) for k, v in args.__dict__.items()}

def print_args(args):
    state = args_to_dict(args)
    state = json.dumps(state, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    print(state)

def save_args(output_dir, args):
    state = args_to_dict(args)
    save_json(f"{output_dir}/args.json", state)

def set_seed(args):
    #random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, dataset, dev_dataset, model):
    model.train()
    dataloader = DataLoader(dataset, shuffle=args.shuffle, batch_size=args.batch_size, num_workers = args.num_workers)
    args.total_steps = len(dataloader) * args.epoch // args.gradient_accumulation_steps

    lr_lambda = lambda epoch: 0.9 ** (epoch)
    if args.optim == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr,
                          betas=(args.adam_B1, args.adam_B2), weight_decay=args.weight_decay, eps=args.adam_eps)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        optimizer = Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    model, optimizer = to_fp16(args, model, optimizer)

    model = to_parallel(args, model)

    steps = 0
    best_score = defaultdict(lambda:-1)
    best_preds = None

    for epoch in range(1,args.epoch+1):
        outputs = []
        tr_loss = []
        if epoch >= args.lr_dec_epoch:
            scheduler.step()
        if args.emb_unfreeze is not None and epoch >= args.emb_unfreeze:
            model.emb_freeze = False

        for batch in tqdm.tqdm(dataloader, desc=f"TRAIN {epoch}"):
            loss, logit, *_ = model(input_ids=batch["input_ids"].to(args.device),
                          pos1_ids=batch["pos1_ids"].to(args.device),
                          pos2_ids=batch["pos2_ids"].to(args.device),
                          labels=batch["label"].to(args.device))

            outputs += list(zip(batch["example_id"], logit.cpu().tolist()))

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss.append(loss.item())

            steps += 1
            if steps % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()
        print(f"|LOSS|{sum(tr_loss)/len(tr_loss)}|LR|{scheduler.get_lr()}|")

        score, preds = dataset.evaluate(outputs)
        print(f"|{'TRAIN':<7}|{score['precision']:>6.2f}|{score['recall']:>6.2f}|{score['f1']:>6.2f}|")

        dev_score, dev_preds = eval(args, dev_dataset, model)
        print(f"|{'DEV':<7}|{dev_score['precision']:>6.2f}|{dev_score['recall']:>6.2f}|{dev_score['f1']:>6.2f}|")
        if dev_score['f1'] > best_score['f1']:
            best_epoch = epoch
            best_score = dev_score
            best_preds = dev_preds
            best_model_param = model.state_dict()

        model.freeze = True

    model.load_state_dict(best_model_param)
    print(f"|{'BEST DEV':<7}|{best_score['precision']:>6.2f}|{best_score['recall']:>6.2f}|{best_score['f1']:>6.2f}|")

    score, preds = eval(args, dataset, model)
    print(f"|{'BEST(DEV) TRAIN':<7}|{score['precision']:>6.2f}|{score['recall']:>6.2f}|{score['f1']:>6.2f}|")

    return model, score, best_score, preds, best_preds, best_epoch

def eval(args, dataset, model):
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers = args.num_workers)

    outputs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="EVAL"):
            logit, *_ = model(input_ids=batch["input_ids"].to(args.device),
                         pos1_ids=batch["pos1_ids"].to(args.device),
                         pos2_ids=batch["pos2_ids"].to(args.device))

            outputs += list(zip(batch["example_id"], logit.cpu().tolist()))

    return dataset.evaluate(outputs)


def main(args=None):
    if args is None:
        args = load_arg()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    print_args(args)

    if args.glove is not None:
        embedding_vectors, word2id = load_word2vec(args.glove, vocab=args.vocab, use_gensim=False)
    else:
        embedding_vectors, word2id = load_word2vec(args.word2vec, vocab=args.vocab, use_gensim=True)
    train_dataset, dev_dataset, test_dataset = load_tacred_dataset(args, word2id)

    if args.entity_mask:
        mask_vectors = torch.randn(len(train_dataset.ner_tags)*2,embedding_vectors.size(1))
        embedding_vectors = torch.cat([embedding_vectors, mask_vectors], dim=0)

    label_weights = train_dataset.label_weights if args.label_weights else None
    model = CNNForRE(args, embedding_vectors, pad_id=train_dataset.pad_id,
                     num_labels=train_dataset.num_labels, label_weights=label_weights)

    model.to(args.device)

    preds, scores = {}, {}
    model, scores["train"], scores["dev"], preds["train"], preds["dev"], best_epoch = train(args, train_dataset, dev_dataset, model)

    test_score = None
    if args.do_eval:
        scores["test"], preds["test"] = eval(args, test_dataset, model)
        print(f"|{'TEST':<7}|{scores['test']['precision']:>6.2f}|{scores['test']['recall']:>6.2f}|{scores['test']['f1']:>6.2f}|")

    model.to("cpu")

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
        save_model(args.output, model)
        save_args(args.output, args)
        save_json(f"{args.output}/preds.json", preds)
        save_json(f"{args.output}/scores.json", scores)

    return model, scores, best_epoch

if __name__ == '__main__':
    main()
