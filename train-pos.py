#!/usr/bin/env python3

import sys, argparse
import chainn.util.functions as UF

from chainn.machine import POSTrainer

""" Arguments """
parser              = argparse.ArgumentParser("POS Tagger training script.")
positive            = lambda x: UF.check_positive(x, int)
positive_decimal    = lambda x: UF.check_positive(x, float)
nonnegative         = lambda x: UF.check_non_negative(x, int)
nonnegative_decimal = lambda x: UF.check_non_negative(x, float)
# Required
parser.add_argument("--model_out", type=str, required=True)
# Parameters
parser.add_argument("--hidden", type=positive, default=128, help="Size of hidden layer.")
parser.add_argument("--embed", type=positive, default=128, help="Size of embedding vector.")
parser.add_argument("--batch", type=positive, default=64, help="Number of (src) sentences in batch.")
parser.add_argument("--epoch", type=positive, default=10, help="Number of max epoch to train the model.")
parser.add_argument("--depth", type=positive, default=1, help="Layers used for the network.")
parser.add_argument("--unk_cut", type=nonnegative, default=1, help="Threshold for words in corpora to be treated as unknown.")
parser.add_argument("--dropout", type=nonnegative_decimal, default=0.2, help="Dropout ratio for LSTM.")
parser.add_argument("--optimizer", type=str, default="", help="Optimizer used for training.")
# Configuration
parser.add_argument("--verbose", action="store_true", help="To output the training progress for every sentence in corpora.")
parser.add_argument("--use_cpu", action="store_true", help="Force to use CPU.")
parser.add_argument("--save_models", action="store_true", help="Save models for every iteration with auto enumeration.")
parser.add_argument("--gpu", type=int, default=-1, help="Specify GPU to be used, negative for using CPU.")
parser.add_argument("--init_model", type=str, help="Init the training weights with saved model.")
parser.add_argument("--model",type=str,choices=["lstm"], default="lstm", help="Type of model being trained.")
parser.add_argument("--seed", type=int, default=0, help="Seed for RNG. 0 for totally random seed.")
parser.add_argument("--one_epoch", action="store_true", help="Finish the training in 1 epoch")
parser.add_argument("--debug", type=bool, default=False, help="Whether to start the model in deubg mode")
# Development set
parser.add_argument("--dev", type=str, help="Development data")
args = parser.parse_args()

""" Training """
trainer   = POSTrainer(args)
trainer.print_details()
trainer.train()

