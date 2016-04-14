#!/usr/bin/env python3 

import sys
import argparse
import math

from chainn import functions as UF
from chainn.classifier import LanguageModel
from chainn.util.io import load_lm_data, load_lm_gen_data, batch_generator
from chainn.machine import Tester

""" Arguments """
parser = argparse.ArgumentParser("Language model using LSTM RNN.")
positive = lambda x: UF.check_positive(x, int)
# Required
parser.add_argument("--init_model", required=True, type=str, help="Initiate the model from previous")
# Optional
parser.add_argument("--src", type=str, help="Specify this to do batched decoding, it has a priority than stdin.")
parser.add_argument("--batch", type=int, help="Minibatch size", default=1)
parser.add_argument("--operation", choices=["sppl", "cppl", "gen"], help="sppl: Sentence-wise ppl\ncppl: Corpus-wise ppl\ngen: Read input, start generating random words.", default="sppl")
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to use (Negative for cpu).")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--gen_limit", type=positive, default=50)
parser.add_argument("--eos_disc", type=float, default=0.0, help="Give fraction positive discount to output longer sentence.")
args = parser.parse_args()
op   = args.operation

if op == "sppl" and args.batch != 1:
    raise ValueError("Currently sentence based perplexity not supports multi batching.")
if args.use_cpu:
    args.gpu = -1

# Loading model
UF.trace("Setting up classifier")
model  = LanguageModel(args, use_gpu=args.gpu, collect_output=True)
VOC, _ = model.get_vocabularies()
decoding_options = {"gen_limit": args.gen_limit, "eos_disc": args.eos_disc}

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

def onBatchUpdate(ctr, src, trg):
    # Decoding
    if args.verbose:
        pass

def onSingleUpdate(ctr, src, trg):
    if op == "gen":
        print(VOC.str_rpr(trg[0]))
    elif op == "sppl":
        print(PPL(trg))

def onDecodingFinish(data, output):
    if op == "gen":
        for src_id, (inp, out) in sorted(output.items(), key=lambda x:x[0]):
            print(TRG.str_rpr(out))
    elif op == "cppl":
        UF.trace("Corpus PPL:", PPL(output))
        print(PPL(output))

tester = Tester(load_lm_gen_data, VOC, onDecodingStart, onBatchUpdate, onSingleUpdate, onDecodingFinish, batch=args.batch, out_vocab=VOC, options=decoding_options)
if op == "sppl" or op == "cppl":
    if not args.src:
        _, data = load_lm_data(sys.stdin, VOC)
    else:
        with open(args.src) as src_fp:
            _, data = load_lm_data(src_fp, VOC)
    data = list(batch_generator(data, (VOC, VOC), args.batch))
    tester.eval(data, model)
elif op == "gen":
    tester.test(args.src, model)
else:
    raise NotImplementedError("Undefined operation:", op)
    
