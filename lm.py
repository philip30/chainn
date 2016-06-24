#!/usr/bin/env python3 

import sys, argparse

from chainn import functions as UF
from chainn.util.io import load_lm_gen_data
from chainn.machine import LMTester

""" Arguments """
parser       = argparse.ArgumentParser("Language Model toolkit with LSTM.")
positive     = lambda x: UF.check_positive(x, int)
non_negative = lambda x: UF.check_non_negative(x, int)
non_negative_dec = lambda x: UF.check_non_negative(x, float)
# Required
parser.add_argument("--init_model", nargs="+", required=True, type=str, help="Directories of model trained using the training script.")
# Optional
parser.add_argument("--operation", choices=["sppl", "cppl", "gen"], help="sppl: Sentence-wise ppl\ncppl: Corpus-wise ppl\ngen: Read input, start generating random words.", default="sppl")
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--beam", type=positive, default=1, help="Beam size in beam search decoding.")
parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to use (Negative for cpu).")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--eos_disc", type=non_negative_dec, default=0.0, help="Give fraction positive discount to output longer sentence.")
args = parser.parse_args()

# Execute Testing
tester = LMTester(args, load_lm_gen_data)
tester.test()

