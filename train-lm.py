#!/usr/bin/env python3

import sys
import argparse
import math
import numpy as np
from collections import defaultdict

import chainer
from chainer import Chain, cuda, optimizers, Variable

from chainn import functions as UF
from chainn.model import RNNParallelSequence
from chainn.util import Vocabulary, ModelFile

def parse_args():
    parser = argparse.ArgumentParser("Program for POS-Tagging classification using RNN/LSTM-RNN")
    parser.add_argument("--hidden", type=int, help="Hidden unit size", default=100)
    parser.add_argument("--embed", type=int, help="Embedding vector size", default=100)
    parser.add_argument("--depth", type=int, help="Depth of the network", default=2)
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--epoch", type=int, help="Epoch", default=30)
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
        word_dev, next_word_dev, _ = load_data(args.dev, args.batch, X)

    # Setup model
    UF.trace("Setting up classifier")
    opt   = optimizers.SGD(lr=1.)
    model = RNNParallelSequence(args, X, X, opt, not args.use_cpu)
    
    # Hooking
    opt.add_hook(chainer.optimizer.GradientClipping(5))

    # Begin training
    UF.trace("Begin training RNN Pos Tagger")
    prev_loss = 1e10
    for ep in range(epoch_total):
        UF.trace("Epoch %d" % (ep+1))
        epoch_loss = 0
        for x_data, y_data in zip(word, next_word):
            accum_loss, accum_acc, output = model.train(x_data, y_data)
            epoch_loss += accum_loss
        epoch_loss /= len(word)

        print("PPL Train:", math.exp(epoch_loss), file=sys.stderr)

        # Evaluate on Dev Set
        if args.dev:
            dev_loss, total_sent = 0, 0
            for x_data, y_data in zip(word_dev, next_word_dev):
                accum_loss, _, _ = model.train(x_data, y_data, update=False)
                dev_loss   += accum_loss
                total_sent += len(x_data)
            dev_loss /= len(word_dev)
            epoch_loss = dev_loss
            print("PPL Dev:", math.exp(dev_loss), file=sys.stderr)
            
        # Decaying Weight
        if prev_loss < epoch_loss or ep > 5:
            opt.lr *= 0.5
            UF.trace("Reducing LR:", opt.lr)
        prev_loss = epoch_loss

    UF.trace("Saving model....")
    with ModelFile(open(args.model_out, "w")) as model_out:
        model.save(model_out)

def load_data(fp, batch_size, x_ids):
    holder        = defaultdict(lambda:[])
    # Reading in the data
    for sent_id, line in enumerate(fp):
        sent          = ["<s>"] + line.strip().lower().split() + ["</s>"]
        words, next_w = [], []
        for i in range(len(sent)-1):
            words.append(x_ids[sent[i]])
            next_w.append(x_ids[sent[i+1]])
        holder[len(words)].append((sent_id, words, next_w))

    # Convert to appropriate data structure
    X, Y = [], []
    for src_len, items in sorted(holder.items(), key=lambda x:x[0]):
        item_count = 0
        x_batch, y_batch = [], []
        for sent_id, words, next_words in items:
            x_batch.append(words)
            y_batch.append(next_words)
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

