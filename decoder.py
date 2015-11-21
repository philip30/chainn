#!/usr/bin/env python

import sys
import argparse

import util.functions as UF
import util.generators as UG

# default parameter
def_batch    = 64
def_genlimit = 50

def main():
    # Preparations
    args  = parse_args()
    
    # Loading model
    UF.trace("Loading model:", args.init_model)
    model = UF.select_model(args.model)(use_gpu=not args.use_cpu)
    with open(args.init_model) as fp:
        model.load(fp)
    
    # Decoding
    SRC, TRG   = model.get_vocabularies()
    B          = args.batch
    PRE        = lambda x: x.strip().lower().split()
    POST       = lambda x: convert_to_id(x, SRC)
    if args.src:
        # Batched decoding
        # Note that to get a correct order the src must be sorted according to its length
        # ascendingly.
        for src in UG.same_len_batch((args.src,), B, PRE, POST):
            trg = model.decode(src)
            print_result(trg, TRG)
    else:
        UF.trace("src is not specified, reading src from stdin.")
        # Line by line decoding
        for line in sys.stdin:
            line = POST([PRE(line)])
            trg = model.decode(line)
            print_result(trg, TRG)

def print_result(trg, TRG):
    for result in trg:
        print(TRG.str_rpr(result))

def convert_to_id(batch, src_voc):
    SRC = src_voc
    max_src = max(len(src) for src in batch)
    src_batch = []
    EOS = SRC.get_eos()
    for src in batch:
        swids  = [SRC[x] for x in src]
        swids.append(SRC[EOS])
        src_batch.append(swids)
    return src_batch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--init_model", type=str, required=True)
    parser.add_argument("--model",choices=["encdec","att"], default="encdec")
    parser.add_argument("--batch", type=int, default=def_batch)
    parser.add_argument("--src", type=str)
    parser.add_argument("--gen_limit", type=int, default=def_genlimit)
    return parser.parse_args()

if __name__ == "__main__":
    main()

