#!/usr/bin/env python3 

import sys, argparse

from chainn import functions as UF
from chainn.util.io import load_pos_test_data
from chainn.machine import POSTester

""" Arguments """
parser       = argparse.ArgumentParser("POS Tagger toolkit with LSTM")
positive     = lambda x: UF.check_positive(x, int)
non_negative = lambda x: UF.check_non_negative(x, int)
# Required
parser.add_argument("--init_model", nargs="+", type=str, required=True, help="Directories of model trained using the training script.")
# Options
parser.add_argument("--use_cpu", action="store_true", help="To Force use CPU.")
parser.add_argument("--beam", type=positive, default=1, help="Beam size in beam search decoding.")
parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to use (Negative for cpu).")
parser.add_argument("--verbose", action="store_true", help="Verbose!")
args  = parser.parse_args()

# Execute testing
tester = POSTester(args, load_pos_test_data)
tester.test()
   
