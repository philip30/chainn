#!/usr/bin/env python

import sys
import argparse

from chainn import functions as UF
from chainn import output

from chainn.model import EncDecNMT
from chainn.util import load_nmt_test_data

# default parameter
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", type=str, help="Directory to the model trained with train-nmt", required=True)
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--src", type=str)
    parser.add_argument("--gen_limit", type=int, default=50)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def main():
    # Preparations
    args  = parse_args()
    
    # Loading model
    UF.trace("Setting up classifier")
    model = EncDecNMT(args, use_gpu=not args.use_cpu)
    SRC, TRG  = model.get_vocabularies()

    # Decoding
    if args.src:
        # Batched decoding
        UF.trace("Loading test data...")
        with open(args.src) as src_fp:
            data = load_nmt_test_data(src_fp, SRC, batch_size=args.batch)
            UF.trace("Decoding started.")
            for src in data:
                trg = model(src, gen_limit=args.gen_limit)
                 
                for trg_out in trg:
                    print(TRG.str_rpr(trg_out))
    
                if args.verbose:
                    print_result(trg, TRG, src, SRC, sys.stderr)
    else:
        UF.trace("src is not specified, reading src from stdin.")
        # Line by line decoding
        for line in sys.stdin:
            line = list(load_nmt_test_data([line.strip()], SRC))
            trg = model(line[0], gen_limit=args.gen_limit)
            print_result(trg, TRG, line[0], SRC, sys.stdout)
    
def print_result(trg, TRG, src, SRC, fp=sys.stderr):
    for i, (sent, result) in enumerate(zip(src, trg)):
        print("SRC:", SRC.str_rpr(sent), file=fp)
        print("TRG:", TRG.str_rpr(result), file=fp)

if __name__ == "__main__":
    main()

