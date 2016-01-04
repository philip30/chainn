#!/usr/bin/env python3 

import sys
import argparse
import numpy as np
import math

from collections import defaultdict
from chainn import functions as UF
from chainn.model import ParallelTextClassifier
from chainn.util import load_lm_data

def parse_args():
    parser = argparse.ArgumentParser("Program for multi-class classification using multi layered perceptron")
    parser.add_argument("--batch", type=int, help="Minibatch size", default=1)
    parser.add_argument("--init_model", required=True, type=str, help="Initiate the model from previous")
    parser.add_argument("--operation", choices=["sppl", "cppl"], help="sppl: Sentence-wise ppl\ncppl: Corpus-wise ppl", default="sppl")
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup model
    UF.trace("Setting up classifier")
    model = ParallelTextClassifier(args, use_gpu=not args.use_cpu, collect_output=True)
    X, Y  = model.get_vocabularies()

    # data
    UF.trace("Loading test data + dictionary from stdin")
    word, next_word, _, sent_ids = load_lm_data(sys.stdin, X, batch_size=args.batch)
       
    # POS Tagging
    output_collector = {}
    UF.trace("Start Calculating PPL")
    for x_data, y_data, batch_id in zip(word, next_word, sent_ids):
        accum_loss, _, output = model.train(x_data, y_data, update=False)
        
        accum_loss = accum_loss / len(x_data)
        for inp, result, id in zip(x_data, output, batch_id):
            output_collector[id] = (X.str_rpr(result), accum_loss)
            
            if args.verbose:
                inp    = [Y.tok_rpr(x) for x in inp]
                result = [X.tok_rpr(x) for x in result]
                print("INP:", " ".join(inp), file=sys.stderr)
                print("OUT:", " ".join(result), file=sys.stderr)
                print("PPL:", math.exp(accum_loss), file=sys.stderr)

    # Printing all output
    gen_fp = open(args.gen, "w") if args.gen else None
    operation = args.operation
    if operation == "sppl":
        for _, (result, accum_loss) in sorted(output_collector.items(), key=lambda x:x[0]):
            print(math.exp(accum_loss))
            if gen_fp is not None:
                print(result, file=gen_fp)
    elif operation == "cppl":
        total_loss = 0
        for _, (result, accum_loss) in output_collector.items():
            total_loss += accum_loss
        total_loss = float(total_loss) / len(output_collector)
        print(math.exp(total_loss))
    if gen_fp is not None:
        gen_fp.close()

if __name__ == "__main__":
    main()

