import os
import json
import gensim

from collections import Counter

import torch
from torch.utils.data import Dataset

OBJ_TOKEN = "[OBJ_NER]"
SUBJ_TOKEN = "[SUBJ_NER]"

def load_word2vec(file_path, vocab=64000, pad_token="[PAD]", unk_token="[UNK]", use_gensim=True):
    if use_gensim:
        model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
        words = model.wv.index2word
        vectors = model.wv.vectors
    else:
        words, vectors = [], []
        with open(file_path, "r") as f:
            for line in f:
                word, *vector = line.split(" ")
                words.append(word)
                try:
                    vectors.append(list(map(float, vector)))
                except :
                    assert False, vector[:10]
                if len(words) >= vocab:
                    break

    word2id = {word:i for i, word in enumerate(words[:vocab] + [pad_token, unk_token])}
    embedding_vectors = torch.FloatTensor(vectors[:vocab])
    pad_embed = torch.zeros(1, embedding_vectors.size(1)) #init by zero
    unk_embed = torch.randn(1, embedding_vectors.size(1)) #init by random
    embedding_vectors = torch.cat([embedding_vectors, pad_embed, unk_embed], dim=0)
    return embedding_vectors, word2id

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
            return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token

def compute_f1(preds, labels):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec*100, 'recall': recall*100, 'f1': f1*100}

class TacredDataset(Dataset):
    def __init__(self, data_dir, data_name, word2id=None, ner_tags=None, entity_mask=False,
                 seq_len=120, labels=None, pad_token="[PAD]", unk_token=["UNK"]):
        self.dataset = load_json(os.path.join(data_dir, data_name))
        self.examples = self._create_examples(self.dataset)
        self.entity_mask = entity_mask

        self.labels = labels
        if labels is None:
            self.labels = self._get_labels(self.examples, negative_label="no_relation")

        self.word2id = word2id
        if word2id is None:
            self.word2id = self._get_word2id(self.examples)

        self.ner_tags = ner_tags
        if ner_tags is None:
            self.ner_tags = self._get_ner_tags(self.examples)
            masks = [OBJ_TOKEN.replace("NER", tag) for tag in self.ner_tags]
            masks += [SUBJ_TOKEN.replace("NER", tag) for tag in self.ner_tags]
            vocab = len(self.word2id)
            for i, mask in enumerate(masks):
                self.word2id[mask] = vocab+i

        self.id2label = {idx:label for idx, (label, cnt) in enumerate(self.labels)}
        self.label2id = {v:k for k, v in self.id2label.items()}
        self.num_labels = len(self.labels)
        sum_exa = sum([cnt for _, cnt in self.labels])
        inv_label = [sum_exa-cnt for label, cnt in self.labels]
        self.label_weights = [cnt/sum(inv_label) for cnt in inv_label]

        self.seq_len = seq_len
        self.pad_id = self.word2id[pad_token]
        self.unk_id = self.word2id[unk_token]

        self.answers = self._get_answers(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        if self.entity_mask:
            example = self._entity_masking(example)

        token_ids = self._words2ids(example["sentence"])

        pos1_ids = self._get_pos_id(token_ids, example["subj_span"])
        pos2_ids = self._get_pos_id(token_ids, example["obj_span"])

        token_ids = self._padding(token_ids, self.pad_id, self.seq_len)
        pos1_ids = self._padding(pos1_ids, self.seq_len*2-1, self.seq_len)
        pos2_ids = self._padding(pos2_ids, self.seq_len*2-1, self.seq_len)

        d = {
            "input_ids":torch.LongTensor(token_ids),
            "pos1_ids":torch.LongTensor(pos1_ids),
            "pos2_ids":torch.LongTensor(pos2_ids),
            "label":self.label2id[example["label"]],
            "example_id":example["guid"]
        }
        return d

    def _get_pos_id(self, token_ids, span):
        pos_ids = []
        for i in range(len(token_ids)):
            if span[0] <= i and span[1] > i:
                pos_ids.append(self.seq_len - 1)
            elif span[0] > i:
                pos_ids.append(i - span[0] + self.seq_len - 1)
            else:
                pos_ids.append(i - span[1] + self.seq_len)
        return pos_ids

    def _entity_masking(self, example):
        subj_mask = SUBJ_TOKEN.replace("NER", example["subj_ner"])
        obj_mask = OBJ_TOKEN.replace("NER", example["obj_ner"])
        query = [list(example["subj_span"])+[subj_mask],
                 list(example["obj_span"])+[obj_mask]]
        for start, end, mask in sorted(query, key=lambda x:x[0], reverse=True):
            example["sentence"] = example["sentence"][:start] + [mask] + example["sentence"][end:]
        subj_idx = example["sentence"].index(subj_mask)
        obj_idx = example["sentence"].index(obj_mask)
        example["obj_span"] = (obj_idx, obj_idx+1)
        example["subj_span"] = (subj_idx, subj_idx+1)
        return example

    def _get_ner_tags(self, examples):
        ner_tags = set()
        for example in examples:
            ner_tags.add(example["obj_ner"])
            ner_tags.add(example["subj_ner"])
        return sorted(ner_tags)

    def __len__(self):
        return len(self.dataset)

    def _get_answers(self, examples):
        answers = {}
        for example in examples:
            answers[example["guid"]] = self.label2id[example["label"]]
        return answers

    def _get_word2id(self, examples):
        total_tokens = []
        for example in examples:
            total_tokens += list(map(lambda x:x.lower(), example["sentence"]))
        vocab = Counter(total_tokens)
        del vocab["unk"]
        vocab = sorted(vocab.items(), key=lambda x:(x[1],x[0]), reverse=True)
        vocab, cnt = zip(*vocab)
        vocab = ["unk"] + list(vocab)
        word2id = {word:idx for idx, word in enumerate(vocab)}
        return word2id

    def _words2ids(self, words):
        return [self.word2id.get(word, self.unk_id) for word in words]

    def _padding(self, array, pad, length):
        return array + [pad] * (length - len(array))

    def _get_labels(self, examples, negative_label="no_relation"):
        count = Counter()
        for example in examples:
            count[example['label']] += 1
        # Make sure the negative label is alwyas 0
        labels = [negative_label]
        for label, _ in count.most_common():
            if label not in labels:
                labels.append(label)
        labels = [(label, count[label]) for label in labels]
        return labels

    def _create_examples(self, dataset):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in dataset:
            sentence = [convert_token(token) for token in example['token']]
            assert example['subj_start'] >= 0 and example['subj_start'] <= example['subj_end'] \
                and example['subj_end'] < len(sentence)
            assert example['obj_start'] >= 0 and example['obj_start'] <= example['obj_end'] \
                and example['obj_end'] < len(sentence)
            examples.append({"guid":example['id'],
                             "sentence":sentence,
                             "subj_span":(example['subj_start'], example['subj_end']+1), #Add 1 to span to correspond to slice of the array.
                             "obj_span":(example['obj_start'], example['obj_end']+1),
                             "subj_ner":example['subj_type'],
                             "obj_ner":example['obj_type'],
                             "label":example['relation']})
        return examples

    def evaluate(self, outputs):
        outputs.sort(key=lambda x:x[0])
        example_ids, preds = list(map(list,zip(*outputs)))
        labels = [self.answers[idx] for idx in example_ids]
        outputs = [[example_id, self.id2label[pred]] for example_id, pred in outputs]
        return compute_f1(preds, labels), outputs

def load_tacred_dataset(args, word2id=None):
    train_dataset = TacredDataset(args.dataset, "train.json", word2id=word2id, seq_len=args.seq_len,
                                  pad_token=args.pad_token, unk_token=args.unk_token, entity_mask=args.entity_mask)
    dev_dataset = TacredDataset(args.dataset, "dev.json", word2id=train_dataset.word2id,
                                seq_len=args.seq_len, labels=train_dataset.labels, ner_tags=train_dataset.ner_tags,
                                pad_token=args.pad_token, unk_token=args.unk_token, entity_mask=args.entity_mask)
    test_dataset = None
    if args.do_eval:
        test_dataset = TacredDataset(args.dataset, "test.json", word2id=train_dataset.word2id,
                                     seq_len=args.seq_len, labels=train_dataset.labels, ner_tags=train_dataset.ner_tags,
                                     pad_token=args.pad_token, unk_token=args.unk_token, entity_mask=args.entity_mask)
    return train_dataset, dev_dataset, test_dataset

if __name__ == '__main__':
    args = load_arg()
