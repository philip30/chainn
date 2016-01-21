#!/usr/bin/env python3

import sys
import argparse
import math
import numpy as np
from collections import defaultdict

import chainer
import chainer.functions as F
from chainer import Chain, optimizers, Variable

from chainn import functions as UF
from chainn.model import ParallelTextClassifier
from chainn.util import ModelFile, load_lm_data

def parse_args():
    parser = argparse.ArgumentParser("Program for POS-Tagging classification using RNN/LSTM-RNN")
    parser.add_argument("--hidden", type=int, help="Hidden unit size", default=200)
    parser.add_argument("--embed", type=int, help="Embedding vector size", default=200)
    parser.add_argument("--depth", type=int, help="Depth of the network", default=2)
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--epoch", type=int, help="Epoch", default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--model_out", type=str, help="Where the model is saved", required=True)
    parser.add_argument("--init_model", type=str, help="Initialize model with the previous")
    parser.add_argument("--model", type=str, choices=["lstm", "rnn"], default="lstm")
    parser.add_argument("--dev", type=str)
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Variable
    epoch_total = args.epoch

    # data
    UF.trace("Loading corpus + dictionary")
    X, train_data = load_lm_data(sys.stdin, batch_size=args.batch)
    if args.dev:
        with open(args.dev) as dev_fp:
            X, dev_data = load_lm_data(dev_fp, X, batch_size=args.batch)

    # Setup model
    UF.trace("Setting up classifier")
    opt   = optimizers.Adam()
    model = ParallelTextClassifier(args, X, X, opt, not args.use_cpu, activation=F.relu)
    
    # Hooking
    opt.add_hook(chainer.optimizer.GradientClipping(10))

    # Begin training
    UF.trace("Begin training Language Model")
    prev_loss = 1e10
    for ep in range(epoch_total):
        UF.trace("Epoch %d" % (ep+1))
        epoch_loss = 0
        for x_data, y_data in train_data:
            accum_loss, accum_acc, output = model.train(x_data, y_data)
            epoch_loss += float(accum_loss)
        epoch_loss /= len(train_data)

        print("PPL Train:", math.exp(epoch_loss), file=sys.stderr)

        # Evaluate on Dev Set
        if args.dev:
            dev_loss = 0
            for x_data, y_data in dev_data:
                accum_loss, _, _ = model.train(x_data, y_data, update=False)
                dev_loss   += float(accum_loss)
            dev_loss /= len(dev_data)
            epoch_loss = dev_loss
            print("PPL Dev:", math.exp(dev_loss), file=sys.stderr)
            
        # Decaying Weight
        if prev_loss < epoch_loss and hasattr(opt,'lr'):
            try:
                opt.lr *= 0.5
                UF.trace("Reducing LR:", opt.lr)
            except: pass
        prev_loss = epoch_loss

    UF.trace("Saving model to", args.model_out, "...")
    with ModelFile(open(args.model_out, "w")) as model_out:
        model.save(model_out)

if __name__ == "__main__":
    main()

