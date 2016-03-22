#!/usr/bin/env python3 

import sys
import argparse
import numpy as np
import math

from collections import defaultdict
from chainn import functions as UF
from chainn.model import ParallelTextClassifier
from chainn.util.io import load_lm_data, batch_generator

def parse_args():
    parser = argparse.ArgumentParser("Program for multi-class classification using multi layered perceptron")
    parser.add_argument("--batch", type=int, help="Minibatch size", default=1)
    parser.add_argument("--init_model", required=True, type=str, help="Initiate the model from previous")
    parser.add_argument("--operation", choices=["sppl", "cppl"], help="sppl: Sentence-wise ppl\ncppl: Corpus-wise ppl", default="sppl")
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=-1)
    return parser.parse_args()

def main():
    args = check_args(parse_args())
    
    # Setup model
    UF.trace("Setting up classifier")
    model = ParallelTextClassifier(args, use_gpu=args.gpu, collect_output=True)
    X, Y  = model.get_vocabularies()

    # data
    UF.trace("Loading test data + dictionary from stdin")
    _, data = load_lm_data(sys.stdin, X)
       
    # Calculating PPL
    gen_fp = open(args.gen, "w") if args.gen else None
    UF.trace("Start Calculating PPL")
    corpus_loss = 0
    for x_data, y_data in batch_generator(data, (X,), batch_size=args.batch):
        accum_loss, output = model.train(x_data, y_data, update=False)
        
        accum_loss = accum_loss / len(x_data)
        for inp, result in zip(x_data, output):
            # Counting PPL
            if args.operation == "sppl":
                print(math.exp(accum_loss))
            else:
                corpus_loss += float(accum_loss) / len(x_data)
            
            # Printing some outputs
            inp    = X.str_rpr(inp)
            result = Y.str_rpr(result)
            if gen_fp is not None:
                print(" ".join(result), file=gen_fp)

            if args.verbose:
                print("INP:", inp, file=sys.stderr)
                print("OUT:", result, file=sys.stderr)
                print("PPL:", math.exp(accum_loss), file=sys.stderr)
    
    if args.operation == "cppl":
        corpus_loss /= len(data)
        print(math.exp(corpus_loss))

    if gen_fp is not None:
        gen_fp.close()

def check_args(args):
    if args.operation == "sppl" and args.batch != 1:
        raise ValueError("Currently sentence based perplexity not supports multi batching.")
    if args.use_cpu:
        args.gpu = -1
    return args

if __name__ == "__main__":
    main()

