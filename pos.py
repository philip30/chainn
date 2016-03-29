#!/usr/bin/env python3 

import sys
import argparse

from chainn import functions as UF
from chainn.model import ParallelTextClassifier
from chainn.util.io import load_pos_test_data
from chainn.machine import Tester

""" Arguments """
parser = argparse.ArgumentParser()
positive = lambda x: UF.check_positive(x, int)
# Required
parser.add_argument("--init_model", type=str, help="Directory to the model trained with train-nmt.", required=True)
# Options
parser.add_argument("--batch", type=positive, default=512, help="Number of source word in the batch.")
parser.add_argument("--src", type=str, help="Specify this to do batched decoding, it has a priority than stdin.")
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to use (Negative for cpu).")
parser.add_argument("--verbose", action="store_true")
args  = parser.parse_args()

""" Sanity Check """
if args.use_cpu:
    args.gpu = -1

# Loading model
UF.trace("Setting up classifier")
model    = ParallelTextClassifier(args, use_gpu=args.gpu, collect_output=True)
SRC, TRG = model.get_vocabularies()

# Testing callbacks
def print_result(ctr, trg, TRG, src, SRC, fp=sys.stderr):
    for i, (sent, result) in enumerate(zip(src, trg.y)):
        print(ctr + i, file=fp)
        print("INP:", SRC.str_rpr(sent), file=fp)
        print("TAG:", TRG.str_rpr(result), file=fp)
   
def onDecodingStart():
    UF.trace("Tagging started.")

def onBatchUpdate(ctr, src, trg):
    # Decoding
    if args.verbose:
        print_result(ctr, trg, TRG, src, SRC, sys.stderr)

def onSingleUpdate(ctr, src, trg):
    print(TRG.str_rpr(trg[0]))

def onDecodingFinish(data, output):
    for src_id, (inp, out) in sorted(output.items(), key=lambda x:x[0]):
        print(TRG.str_rpr(out))

# Execute testing
tester = Tester(load_pos_test_data, SRC, onDecodingStart, onBatchUpdate, onSingleUpdate, onDecodingFinish, batch=args.batch)
tester.test(args.src, model)
   
