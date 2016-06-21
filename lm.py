#!/usr/bin/env python3 

import sys
import argparse
import math

from chainn import functions as UF
from chainn.classifier import RNNLM
from chainn.util.io import load_lm_data, load_lm_gen_data, batch_generator
from chainn.machine import Tester

""" Arguments """
parser = argparse.ArgumentParser("Language model Toolkit using LSTM RNN.")
positive = lambda x: UF.check_positive(x, int)
# Required
parser.add_argument("--init_model", nargs="+", required=True, type=str, help="Initiate the model from previous")
# Optional
parser.add_argument("--operation", choices=["sppl", "cppl", "gen"], help="sppl: Sentence-wise ppl\ncppl: Corpus-wise ppl\ngen: Read input, start generating random words.", default="sppl")
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to use (Negative for cpu).")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--eos_disc", type=float, default=0.0, help="Give fraction positive discount to output longer sentence.")
args = parser.parse_args()
op   = args.operation

if args.use_cpu:
    args.gpu = -1

# Loading model
UF.trace("Setting up classifier")
lm  = RNNLM(args, use_gpu=args.gpu, collect_output=True, operation=op)
vocabulary = lm.get_vocabularies()
decoding_options = {"eos_disc": args.eos_disc}
total_loss = 0
total_sent = 0

# Testing callbacks
def PPL(loss):
    try:
        return math.exp(loss.data)
    except:
        return math.exp(loss)

def onDecodingStart():
    if op == "gen":
        UF.trace("Sentence generation started.")
    elif op == "cppl":
        UF.trace("Corpus PPL calculation started.")
    elif op == "sppl":
        UF.trace("Sentence PPL calculation started.")

def onSingleUpdate(ctr, src, trg):
    global total_loss, total_sent, vocabulary
    if op == "gen":
        print(vocabulary.str_rpr(trg.y[0]))
    else:
        loss       = trg.loss / len(src[0])
        total_loss += loss
        total_sent += 1
        if op == "sppl":
            print(PPL(loss))

def onDecodingFinish():
    if op == "cppl":
        global total_loss, total_sent
        print(PPL(total_loss/total_sent)) 

tester = Tester(load_lm_gen_data, vocabulary, onDecodingStart, onSingleUpdate, onDecodingFinish, out_vocab=vocabulary, options=decoding_options)
tester.test(lm)

