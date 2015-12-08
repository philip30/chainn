#!/usr/bin/env python

import sys
import argparse

from chainn import functions as UF
from chainn import generators as UG
from chainn import output

# default parameter
def_batch    = 64
def_genlimit = 50

def main():
    # Preparations
    args  = parse_args()
    align_visualizer = output.AlignmentVisualizer(args.align_out)
    
    # Loading model
    UF.trace("Loading model:", args.init_model)
    model = UF.select_model(args.model)(use_gpu=not args.use_cpu, dictionary=args.dictionary, compile=False)
    with open(args.init_model) as fp:
        model.load(fp)
    
    # Decoding
    SRC, TRG   = model.get_vocabularies()
    B          = args.batch
    PRE        = lambda x: x.strip().lower().split()
    POST       = lambda x: convert_to_id(x, SRC)
    print_alignment = lambda x, y, z: align_visualizer.print(x, y, z, SRC, TRG)

    if args.src:
        # Batched decoding
        # Note that to get a correct order the src must be sorted according to its length
        # ascendingly.
        data_batch = list(UG.same_len_batch((args.src,), B, PRE, POST))
        for src in data_batch:
            trg = model.decode(src)
            print_result(trg["decode"], TRG)
            print_alignment(trg["alignment"], src, trg["decode"])
    else:
        UF.trace("src is not specified, reading src from stdin.")
        # Line by line decoding
        for line in sys.stdin:
            line = POST([[PRE(line)]])
            trg = model.decode(line)
            print_result(trg["decode"], TRG)
            print_alignment(trg["alignment"], line, trg["decode"])
    
    # Closing
    align_visualizer.close()

def print_result(trg, TRG):
    for i, result in enumerate(trg):
        print(TRG.str_rpr(result))

def convert_to_id(batch, src_voc):
    SRC = src_voc
    src_batch = []
    EOS = SRC.get_eos()
    for src in batch:
        swids  = [SRC[x] for x in src[0]]
        swids.append(SRC[EOS])
        src_batch.append(swids)
    return src_batch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--init_model", type=str, required=True)
    parser.add_argument("--model",choices=["encdec","attn"], default="attn")
    parser.add_argument("--batch", type=int, default=def_batch)
    parser.add_argument("--src", type=str)
    parser.add_argument("--gen_limit", type=int, default=def_genlimit)
    parser.add_argument("--align_out", type=str)
    parser.add_argument("--dictionary", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    main()

