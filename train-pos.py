#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from collections import defaultdict

import chainer
from chainer import Chain, cuda, optimizers, Variable

from chainn import functions as UF
from chainn.model import RNNParallelSequence
from chainn.util import Vocabulary, ModelFile

def parse_args():
    parser = argparse.ArgumentParser("Program for POS-Tagging classification using RNN/LSTM-RNN")
    parser.add_argument("--hidden", type=int, help="Hidden unit size", default=50)
    parser.add_argument("--embed", type=int, help="Embedding vector size", default=200)
    parser.add_argument("--depth", type=int, help="Depth of the network", default=1)
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--epoch", type=int, help="Epoch", default=100)
    parser.add_argument("--model_out", type=str, help="Where the model is saved", required=True)
    parser.add_argument("--init_model", type=str, help="Initialize model with the previous")
    parser.add_argument("--model", type=str, choices=["lstm", "rnn"], default="lstm")
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Variable
    batch_size = args.batch
    epoch_total = args.epoch

    # data
    UF.trace("Loading corpus + dictionary")
    train, label, X, Y = load_data(args.batch)

    # Setup model
    UF.trace("Setting up classifier")
    opt   = optimizers.AdaGrad(lr=0.01)
    model = RNNParallelSequence(args, X, Y, opt, not args.use_cpu)
    
    # Hooking
    opt.add_hook(chainer.optimizer.GradientClipping(10))

    # Begin training
    UF.trace("Begin training RNN Pos Tagger")
    prev_loss = 1e10
    for ep in range(epoch_total):
        UF.trace("Epoch %d" % (ep+1))
        epoch_loss = 0
        epoch_acc  = 0
        for x_data, y_data in zip(train, label):
            accum_loss, accum_acc, output = model.train(x_data, y_data)
            epoch_loss += accum_loss
            epoch_acc  += accum_acc
                   
        # Decaying Weight
        if prev_loss < epoch_loss:
            opt.lr *= 0.5
            UF.trace("Reducing LR:", opt.lr)
        prev_loss = epoch_loss

        print("Loss:", epoch_loss, file=sys.stderr)
        print("Accuracy:", epoch_acc, file=sys.stderr)

    UF.trace("Saving model....")
    with ModelFile(open(args.model_out, "w")) as model_out:
        model.save(model_out)

def load_data(batch_size):
    x_ids, y_ids  = Vocabulary(), Vocabulary()
    holder        = defaultdict(lambda:[])
    # Reading in the data
    for line in sys.stdin:
        sent          = line.strip().lower().split()
        words, labels = [], []
        for word in sent:
            word, tag = word.split("_")
            words.append(x_ids[word])
            labels.append(y_ids[tag])
        holder[len(words)].append((words,labels))

    # Convert to appropriate data structure
    X, Y = [], []
    for src_len, items in sorted(holder.items(), key=lambda x:x[0]):
        item_count = 0
        x_batch, y_batch = [], []
        for words, labels in items:
            x_batch.append(words)
            y_batch.append(labels)
            item_count += 1

            if item_count % batch_size == 0:
                X.append(x_batch)
                Y.append(y_batch)
                x_batch, y_batch = [], []
        if len(x_batch) != 0:
            X.append(x_batch)
            Y.append(y_batch)
    return X, Y, x_ids, y_ids
 
if __name__ == "__main__":
    main()

