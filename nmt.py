#!/usr/bin/env python3

import sys, argparse

from chainn import functions as UF
from chainn.machine import NMTTester

""" Arguments """
parser = argparse.ArgumentParser("A Neural Machine Translation Decoder.")
positive     = lambda x: UF.check_positive(x, int)
non_negative = lambda x: UF.check_non_negative(x, int)
non_negative_dec = lambda x: UF.check_non_negative(x, float)
# Required
parser.add_argument("--init_model", nargs="+", type=str, help="Directory to the model trained with train-nmt.", required=True)
# Options
parser.add_argument("--gen_limit", type=positive, default=50)
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to use (Negative for cpu).")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--align_out", type=str)
parser.add_argument("--beam", type=positive, default=1)
parser.add_argument("--eos_disc", type=non_negative_dec, default=0.0, help="Give fraction positive discount to output longer sentence.")
args  = parser.parse_args()

# Execute testing
tester = NMTTester(args)
tester.test()

