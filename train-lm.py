#!/usr/bin/env python3

import sys
import argparse
import math
import numpy as np
from collections import defaultdict

import chainer
import chainer.functions as F
from chainer import Chain, cuda, optimizers, Variable

from chainn import functions as UF
from chainn.model import RNNParallelSequence
from chainn.util import Vocabulary, ModelFile

def parse_args():
    parser = argparse.ArgumentParser("Program for POS-Tagging classification using RNN/LSTM-RNN")
    parser.add_argument("--hidden", type=int, help="Hidden unit size", default=200)
    parser.add_argument("--embed", type=int, help="Embedding vector size", default=200)
    parser.add_argument("--depth", type=int, help="Depth of the network", default=2)
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--epoch", type=int, help="Epoch", default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model_out", type=str, help="Where the model is saved", required=True)
    parser.add_argument("--init_model", type=str, help="Initialize model with the previous")
    parser.add_argument("--model", type=str, choices=["lstm", "rnn"], default="lstm")
    parser.add_argument("--dev", type=str)
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Variable
    batch_size = args.batch
    epoch_total = args.epoch

    # data
    UF.trace("Loading corpus + dictionary")
    word, next_word, X = load_data(sys.stdin, args.batch, Vocabulary())
    if args.dev:
        word_dev, next_word_dev, _ = load_data(args.dev, args.batch, X, replace_unk=True)

    # Setup model
    UF.trace("Setting up classifier")
    opt   = optimizers.SGD(lr=args.lr)
    model = RNNParallelSequence(args, X, X, opt, not args.use_cpu, activation=F.relu)
    
    # Hooking
    opt.add_hook(chainer.optimizer.GradientClipping(10))

    # Begin training
    UF.trace("Begin training Language Model")
    prev_loss = 1e10
    for ep in range(epoch_total):
        UF.trace("Epoch %d" % (ep+1))
        epoch_loss = 0
        for x_data, y_data in zip(word, next_word):
            accum_loss, accum_acc, output = model.train(x_data, y_data)
            epoch_loss += float(accum_loss)
        epoch_loss /= len(word)

        print("PPL Train:", math.exp(epoch_loss), file=sys.stderr)

        # Evaluate on Dev Set
        if args.dev:
            dev_loss = 0
            for x_data, y_data in zip(word_dev, next_word_dev):
                accum_loss, _, _ = model.train(x_data, y_data, update=False)
                dev_loss   += float(accum_loss)
            dev_loss /= len(word_dev)
            epoch_loss = dev_loss
            print("PPL Dev:", math.exp(dev_loss), file=sys.stderr)
            
        # Decaying Weight
        if prev_loss < epoch_loss:
            opt.lr *= 0.5
            UF.trace("Reducing LR:", opt.lr)
        prev_loss = epoch_loss

    UF.trace("Saving model to", args.model_out, "...")
    with ModelFile(open(args.model_out, "w")) as model_out:
        model.save(model_out)

def load_data(fp, batch_size, x_ids, replace_unk=False):
    count  = defaultdict(lambda:0)
    holder = defaultdict(lambda:[])
    # Reading and counting the data
    for sent_id, line in enumerate(fp):
        sent = ["<s>"] + line.strip().lower().split() + ["</s>"]
        words, next_w = [], []
        for i, tok in enumerate(sent):
            count[tok] += 1
            if i < len(sent)-1:
                words.append(sent[i])
                next_w.append(sent[i+1])
        holder[len(words)].append([sent_id, words, next_w])

    id_train = lambda x: x_ids[x] if count[x] > 3 else x_ids[x_ids.unk()]
    id_rep = lambda x: x_ids[x] if  x in x_ids else x_ids[x_ids.unk()]
    convert_to_id = id_rep if replace_unk else id_train
    # Convert to appropriate data structure
    X, Y = [], []
    for src_len, items in sorted(holder.items(), key=lambda x:x[0]):
        item_count = 0
        x_batch, y_batch = [], []
        for sent_id, words, next_words in items:
            word = list(map(convert_to_id, words))
            nw   = list(map(convert_to_id, next_words))
            x_batch.append(word)
            y_batch.append(nw)
            item_count += 1

            if item_count % batch_size == 0:
                X.append(x_batch)
                Y.append(y_batch)
                x_batch, y_batch = [], []
        if len(x_batch) != 0:
            X.append(x_batch)
            Y.append(y_batch)
    return X, Y, x_ids
 
if __name__ == "__main__":
    main()

