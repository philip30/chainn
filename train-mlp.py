#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import mlp

from chainer import Chain, cuda, optimizers, Variable, serializers

from chainn import functions as UF
from chainn.model import MLP
from chainn.util import Vocabulary
from chainn.util import ModelFile

xp = None

def parse_args():
    parser = argparse.ArgumentParser("Program for multi-class classification using multi layered perceptron")
    parser.add_argument("--hidden", type=int, help="Hidden unit size", default=200)
    parser.add_argument("--depth", type=int, help="Depth of the network", default=2)
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--epoch", type=int, help="Epoch", default=100)
    parser.add_argument("--model_out", type=str, help="Where the model is saved", required=True)
    parser.add_argument("--init_model", type=str, help="Initiate the model from previous")
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
    train, label, X, Y = load_data()

    # Setup model
    UF.trace("Setting up classifier")
    model = L.Classifier(load_model(args, X, Y))
    if not args.use_cpu: model = model.to_gpu()
    opt   = optimizers.SGD()
    opt.setup(model)
    
    # Begin training
    UF.trace("Begin training MLP")
    for ep in range(epoch_total):
        UF.trace("Epoch %d" % (ep+1))
        accum_loss = 0
        for i in range(0, len(train), batch_size):
            x_data = Variable(train[i:i+batch_size])
            y_data = Variable(label[i:i+batch_size])
            model.zerograds()
            loss = model(x_data, y_data)
            accum_loss += loss
            loss.backward()
            opt.update()
        print("Loss:", loss.data, file=sys.stderr)
        print("Accuracy:", model.accuracy.data, file=sys.stderr)

    for i in range(0, len(train), batch_size):
        x_data = Variable(train[i:i+batch_size])
        y_data = model.predictor(x_data)
        UF.print_classification(y_data.data, Y)

    UF.trace("Saving model....")
    with ModelFile(open(args.model_out, "w")) as model_out:
        model.predictor.save(model_out)

def load_model(args, X, Y):
    if args.init_model:
        with ModelFile(open(args.init_model)) as model_in:
            return MLP.load(model_in)
    else:
        return MLP(X, Y, hidden=args.hidden, depth=args.depth, input=len(X), output=len(Y))

def load_data():
    x_ids, y_ids = Vocabulary(), Vocabulary()
    holder = []
    # Reading in the data
    for line in sys.stdin:
        label, sent = line.strip().lower().split("\t")
        sent = sent.split()
        for i, feat in enumerate(sent):
            feat_name, feat_score = feat.split("_:_")
            feat_score = float(feat_score)
            sent[i] = (x_ids[feat_name], feat_score)
        label = y_ids[label]
        holder.append((sent, label))

    # Creating an appropriate data structure
    num, dim = len(holder), len(x_ids)
    data     = xp.zeros(num * dim, dtype = np.float32).reshape((num, dim))
    target   = xp.zeros(num, dtype = np.int32).reshape((num, ))
    for i, (sent, label) in enumerate(holder):
        for feat_name, feat_score in sent:
            data[i][feat_name] += feat_score
        target[i] = label
    return data, target, x_ids, y_ids

def init_program_state(args):
    global xp
    xp = np if args.use_cpu else cuda.cupy

if __name__ == "__main__":
    main()

