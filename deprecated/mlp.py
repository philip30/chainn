#!/usr/bin/env python3 

import sys
import argparse
import numpy as np

from chainer import Chain, cuda, optimizers, Variable, serializers

from chainn import functions as UF
from chainn.model import MLP
from chainn.util import Vocabulary
from chainn.util import ModelFile

xp = None

def parse_args():
    parser = argparse.ArgumentParser("Program for multi-class classification using multi layered perceptron")
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--init_model", required=True, type=str, help="Initiate the model from previous")
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    init_program_state(args)
    
    # Variable
    batch_size = args.batch

    # Setup model
    UF.trace("Setting up classifier")
    model = load_model(args)
    if not args.use_cpu: model = model.to_gpu()

    # data
    UF.trace("Loading corpus + dictionary")
    test = load_data(sys.stdin, model._feat, model._input)
        
    for i in range(0, len(test), batch_size):
        x_data = Variable(test[i:i+batch_size])
        y_data = model(x_data)
        UF.print_classification(y_data.data, model._trg)

def load_model(args):
    with ModelFile(open(args.init_model)) as model_in:
        return MLP.load(model_in)

def load_data(fp, x_ids, input_size):
    holder = []
    # Reading in the data
    for line in fp:
        sent = line.strip().lower().split()
        for i, feat in enumerate(sent):
            feat_name, feat_score = feat.split("_:_")
            feat_score = float(feat_score)
            if feat_name in x_ids:
                sent[i] = (x_ids[feat_name], feat_score)
        holder.append(sent)

    # Creating an appropriate data structure
    num, dim = len(holder), input_size
    data     = xp.zeros(num * dim, dtype = np.float32).reshape((num, dim))
    for i, sent in enumerate(holder):
        for item in sent:
            if type(item) == tuple:
                feat_name, feat_score = item
                data[i][feat_name] += feat_score
    return data

def init_program_state(args):
    global xp
    xp = np if args.use_cpu else cuda.cupy

if __name__ == "__main__":
    main()

