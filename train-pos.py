#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import chainer.links as L

from chainer import Chain, cuda, optimizers, Variable, serializers

from chainn import functions as UF
from chainn.model import RNNPosTagger
from chainn.util import Vocabulary

xp = None

def parse_args():
    parser = argparse.ArgumentParser("Program for POS-Tagging classification using RNN/LSTM-RNN")
    parser.add_argument("--hidden", type=int, help="Hidden unit size", default=200)
    parser.add_argument("--input", type=int, help="Input vocabulary size", default=25000)
    parser.add_argument("--depth", type=int, help="Depth of the network", default=2)
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--epoch", type=int, help="Epoch", default=100)
    parser.add_argument("--model_out", type=str, help="Where the model is saved", required=True)
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    init_program_state(args)
    
    # Variable
    batch_size = args.batch
    epoch_total = args.epoch

    # data
    UF.trace("Loading corpus + dictionary")
    train, label, X, Y = load_data(args.input)

    # Setup model
    UF.trace("Setting up classifier")
    model = L.Classifier(RNNPosTagger(hidden=args.hidden, depth=args.depth, input=args.input, output=len(Y)))
    if not args.use_cpu: model = model.to_gpu()
    opt   = optimizers.SGD()
    opt.setup(model)

    # Begin training
    UF.trace("Begin training MLP")
    for ep in range(epoch_total):
        UF.trace("Epoch %d" % (ep+1))
        for i in range(0, len(train), batch_size):
            x_data = Variable(train[i:i+batch_size])
            y_data = Variable(label[i:i+batch_size])
            opt.update(model, x_data, y_data) 
    serializers.save_hdf5(args.model_out, model)

def load_data():


def init_program_state(args):
    global xp
    xp = np if args.use_cpu else cuda.cupy

if __name__ == "__main__":
    main()

