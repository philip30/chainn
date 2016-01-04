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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup model
    UF.trace("Setting up classifier")
    model = ParallelTextClassifier(args, use_gpu=not args.use_cpu)
    X, Y  = model.get_vocabularies()

    # data
    UF.trace("Loading test data + dictionary from stdin")
    test, sent_ids = load_pos_test_data(sys.stdin.readlines(), X, args.batch)
       
    # POS Tagging
    output_collector = {}
    UF.trace("Start Tagging")
    for batch, batch_id in zip(test, sent_ids):
        tag_result = model(batch)
        for inp, result, id in zip(batch, tag_result, batch_id):
            output_collector[id] = Y.str_rpr(result)
            
            if args.verbose:
                inp    = [X.tok_rpr(x) for x in inp]
                result = [Y.tok_rpr(x) for x in result]
                print(" ".join(str(x) + "_" + str(y) for x, y in zip(inp, result)), file=sys.stderr)

    # Printing all output
    for _, result in sorted(output_collector.items(), key=lambda x:x[0]):
        print(result)

if __name__ == "__main__":
    main()

