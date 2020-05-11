import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

try:
    import apex
    from apex import amp
except ModuleNotFoundError:
    apex = None

def to_parallel(args, model):
    if args.n_gpu != 1:
        model = torch.nn.DataParallel(model)
    return model

def save_model(output_dir, model):
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), f"{output_dir}/pytorch_model.bin")

def to_fp16(args, model, optimizer=None):
    if args.fp16 and not args.no_cuda:
        if apex is None:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if optimizer is None:
            model = apex.amp.initialize(model, opt_level=args.fp16_opt_level)
        else:
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    if optimizer is None:
        return model
    return model, optimizer

class CNNForRE(nn.Module):
    def __init__(self, args, embedding_vectors, pad_id=None, num_labels=5, label_weights=None):
        super().__init__()
        self.emb_freeze = args.emb_freeze

        #add zero vector for pad_id.
        if pad_id is None:
            pad_id = embedding_vectors.size(0) - 1

        self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_vectors,
                                                           padding_idx=pad_id)
        self.pos1_embedding = nn.Embedding(args.seq_len*2, args.pos_emb_dim, padding_idx=args.seq_len*2-1)
        self.pos2_embedding = nn.Embedding(args.seq_len*2, args.pos_emb_dim, padding_idx=args.seq_len*2-1)

        self.dropout = nn.Dropout(args.dropout_ratio)
        self.embedding_dim = self.word_embedding.embedding_dim + self.pos1_embedding.embedding_dim * 2

        self.covns = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=args.filter_num,
                                              kernel_size=(k, self.embedding_dim),
                                              padding=0) for k in args.filters])

        self.tanh = nn.Tanh()

        self.label_weights = label_weights
        if label_weights is not None:
            self.label_weights = torch.tensor(label_weights).to(args.device)

        filter_dim = args.filter_num * len(args.filters)
        self.linear = nn.Linear(filter_dim, num_labels)

    def forward(self, input_ids, pos1_ids, pos2_ids, labels=None):
        if self.emb_freeze:
            with torch.no_grad():
                word_embs = self.word_embedding(input_ids)
        else:
            word_embs = self.word_embedding(input_ids)
        pos1_embs = self.pos1_embedding(pos1_ids)
        pos2_embs = self.pos2_embedding(pos2_ids)

        input_feature = torch.cat([word_embs, pos1_embs, pos2_embs], dim=2)
        x = input_feature.unsqueeze(1)
        x = self.dropout(x)

        x = [self.tanh(conv(x)).squeeze(3) for conv in self.covns]

        x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in x]
        sentence_features = torch.cat(x, dim=1)

        x = self.dropout(sentence_features)
        logits = self.linear(x)

        outputs = (logits.argmax(-1),)

        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.label_weights)
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs


if __name__ == '__main__':
    import argparse

    def load_arg():
        parser = argparse.ArgumentParser()
        parser.add_argument("--word2vec", type=str)


        parser.add_argument("--filter_num", type=int, default=150)
        parser.add_argument("--pos_emb_dim", type=int, default=50)
        parser.add_argument("--pos_dis_limit", type=int, default=50)
        parser.add_argument('--filters', nargs='*', default=[2,3,4,5])

        parser.add_argument("--emb_freeze", action="store_true")

        parser.add_argument("--dropout_ratio", type=float, default=0.5)
        return parser.parse_args()

    args = load_arg()
    model = CNNForRE(args)
