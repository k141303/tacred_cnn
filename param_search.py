import os
import csv
import itertools

from main import load_arg, main

LR = [5e-1, 1e-1, 5e-2]
BATCH = [16, 32, 64, 128]
WEIGHT_DECAY = [1e-3, 5e-4, 1e-4]

def save_csv(file_path, table):
    with open(file_path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(table)

if __name__ == '__main__':
    header = ["lr", "batch", "weight decay", "best_epoch", "train-prec", "train-rec", "train-f1",
             "dev-prec", "dev-rec", "dev-f1", "test-prec", "test-rec", "test-f1"]
    table = [header]
    for lr, batch, wd in itertools.product(LR, BATCH, WEIGHT_DECAY):
        args = load_arg()
        args.lr = lr
        args.batch_size = batch
        args.weight_decay = wd
        model, scores, best_epoch = main(args)
        row = [lr, batch, wd, best_epoch]
        row += [scores["train"][key] for key in ["precision", "recall", "f1"]]
        row += [scores["dev"][key] for key in ["precision", "recall", "f1"]]
        if scores.get("test") is not None:
            row += [scores["test"][key] for key in ["precision", "recall", "f1"]]
        table.append(row)
        os.makedirs("./scores", exist_ok=True)
        save_csv(f"./scores/param_search_unmask.csv", table)
