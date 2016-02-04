#!/usr/bin/env python3 

import sys
import argparse
import numpy as np

from collections import defaultdict
from chainn import functions as UF
from chainn.model import ParallelTextClassifier
from chainn.util import load_pos_test_data

def parse_args():
    parser = argparse.ArgumentParser("Program for multi-class classification using multi layered perceptron")
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--init_model", required=True, type=str, help="Initiate the model from previous")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=-1)
    return parser.parse_args()

def check_args(args):
    if args.use_cpu:
        args.gpu = -1
    return args

def main():
    args = check_args(parse_args())
    
    # Setup model
    UF.trace("Setting up classifier")
    model = ParallelTextClassifier(args, use_gpu=args.gpu)
    X, Y  = model.get_vocabularies()

    # data
    UF.trace("Loading test data + dictionary from stdin")
    test, ids = load_pos_test_data(sys.stdin.readlines(), X, args.batch)
       
    # POS Tagging
    UF.trace("Start Tagging")
    out = {}
    for batch, batch_id in zip(test, ids):
        tag_result = model(batch)
        for o_id, inp, result in zip(batch_id, batch, tag_result):
            out[o_id] = result
            
            if args.verbose:
                inp    = [X.tok_rpr(x) for x in inp]
                result = [Y.tok_rpr(x) for x in result]
                print(" ".join(str(x) + "_" + str(y) for x, y in zip(inp, result)), file=sys.stderr)
    
    for _, result in sorted(out.items(), key=lambda x:x[0]):
        print(Y.str_rpr(result))
    
if __name__ == "__main__":
    main()

