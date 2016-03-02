#!/usr/bin/env python

import sys
import argparse

from chainn import functions as UF
from chainn import output

from chainn.model import EncDecNMT
from chainn.util import AlignmentVisualizer
from chainn.util.io import load_nmt_test_data, batch_generator

# default parameter
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model", type=str, help="Directory to the model trained with train-nmt", required=True)
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--src", type=str)
    parser.add_argument("--gen_limit", type=int, default=50)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to use (Negative for cpu)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--alignment_out", type=str)
    parser.add_argument("--eos_disc", type=float, default=0.0)
    return parser.parse_args()

def check_args(args):
    if args.use_cpu:
        args.gpu = -1
    return args

def main():
    # Preparations
    args  = check_args(parse_args())
    ao_fp = UF.load_stream(args.alignment_out)

    # Loading model
    UF.trace("Setting up classifier")
    model = EncDecNMT(args, use_gpu=args.gpu, collect_output=True)
    SRC, TRG = model.get_vocabularies()

    decoding_options = {"gen_limit": args.gen_limit, "eos_disc": args.eos_disc}
    # Decoding
    if args.src:
        # Batched decoding
        UF.trace("Loading test data...")
        with open(args.src) as src_fp:
            data = load_nmt_test_data(src_fp, SRC)
            ctr  = 0
            UF.trace("Decoding started.")

            output = {}
            for src, src_id in batch_generator(data, (SRC,), batch_size=args.batch):
                trg = model.decode(src, **decoding_options)
               
                for trg_i, id_i in zip(trg.y, src_id):
                    output[id_i] = trg_i
                
                if ao_fp is not None:
                    AlignmentVisualizer.print(trg.a, ctr, src, trg.y, SRC, TRG, fp=ao_fp)

                if args.verbose:
                    print_result(ctr, trg, TRG, src, SRC, sys.stderr)
                ctr += len(src)

            for _, out in sorted(output.items(), key=lambda x:x[0]):
                print(TRG.str_rpr(out))
    else:
        UF.trace("src is not specified, reading src from stdin.")
        # Line by line decoding
        for i, line in enumerate(sys.stdin):
            line = [x for x in batch_generator(load_nmt_test_data([line.strip()], SRC), (SRC,), 1)]
            src = line[0][0]
            trg = model.decode(src, **decoding_options)
            if args.verbose:
                print_result(i, trg, TRG, src, SRC, sys.stderr)
            if ao_fp is not None:
                AlignmentVisualizer.print(trg.a, i, src, trg.y, SRC, TRG, fp=ao_fp)
            print(TRG.str_rpr(trg.y[0]))
    
    if ao_fp is not None:
        ao_fp.close()

def print_result(ctr, trg, TRG, src, SRC, fp=sys.stderr):
    for i, (sent, result) in enumerate(zip(src, trg.y)):
        print(ctr + i, file=fp)
        print("SRC:", SRC.str_rpr(sent), file=fp)
        print("TRG:", TRG.str_rpr(result), file=fp)
   
    if trg.a is not None:
        AlignmentVisualizer.print(trg.a, ctr, src, trg.y, SRC, TRG, fp)

if __name__ == "__main__":
    main()

