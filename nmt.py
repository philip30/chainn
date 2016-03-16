#!/usr/bin/env python

import sys
import argparse

from chainn import functions as UF

from chainn.model import EncDecNMT
from chainn.util import AlignmentVisualizer
from chainn.util.io import load_nmt_test_data
from chainn.machine import Tester

""" Arguments """
parser = argparse.ArgumentParser()
positive = lambda x: UF.check_positive(x, int)
# Required
parser.add_argument("--init_model", type=str, help="Directory to the model trained with train-nmt.", required=True)
# Options
parser.add_argument("--batch", type=positive, default=512, help="Number of source word in the batch.")
parser.add_argument("--src", type=str, help="Specify this to do batched decoding, it has a priority than stdin.")
parser.add_argument("--gen_limit", type=positive, default=50)
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to use (Negative for cpu).")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--alignment_out", type=str)
parser.add_argument("--eos_disc", type=float, default=0.0, help="Give fraction positive discount to output longer sentence.")
args  = parser.parse_args()

""" Sanity Check """
if args.use_cpu:
    args.gpu = -1

""" Begin Testing """
ao_fp = UF.load_stream(args.alignment_out)
decoding_options = {"gen_limit": args.gen_limit, "eos_disc": args.eos_disc}

# Loading model
UF.trace("Setting up classifier")
model    = EncDecNMT(args, use_gpu=args.gpu, collect_output=True)
SRC, TRG = model.get_vocabularies()

# Testing callbacks
def print_result(ctr, trg, TRG, src, SRC, fp=sys.stderr):

    for i, (sent, result) in enumerate(zip(src, trg.y)):
        print(ctr + i, file=fp)
        print("SRC:", SRC.str_rpr(sent), file=fp)
        print("TRG:", TRG.str_rpr(result), file=fp)
   
    if trg.a is not None:
        AlignmentVisualizer.print(trg.a, ctr, src, trg.y, SRC, TRG, fp)

def onDecodingStart():
    UF.trace("Decoding started.")

def onBatchUpdate(ctr, src, trg):
    # Decoding
    if ao_fp is not None:
        AlignmentVisualizer.print(trg.a, ctr, src, trg.y, SRC, TRG, fp=ao_fp)
    if args.verbose:
        print_result(ctr, trg, TRG, src, SRC, sys.stderr)

def onSingleUpdate(ctr, src, trg):
    onBatchUpdate(ctr, src, trg)
    print(TRG.str_rpr(trg.y[0]))

def onDecodingFinish(output):
    for _, out in sorted(output.items(), key=lambda x:x[0]):
        print(TRG.str_rpr(out))

# Execute testing
tester = Tester(load_nmt_test_data, SRC, decoding_options, onDecodingStart, onBatchUpdate, onSingleUpdate, onDecodingFinish, batch=args.batch)
tester.test(args.src, model)

# Finishing up
if ao_fp is not None:
    ao_fp.close()

