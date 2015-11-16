#!/usr/bin/env python

import sys
import argparse
import math
import random
import numpy as np

import util.functions as UF
import util.generators as UG

from collections import defaultdict
from util.vocabulary import Vocabulary as Vocab
from util.settings import DecoderSettings as DS
from model.encdec import EncoderDecoder
from model.attentional import Attentional
from chainer import optimizers

# Default parameters
def_embed      = 256
def_hidden     = 512
def_batchsize  = 64
def_input      = 32768
def_output     = 32768
def_epoch      = 100
def_lr         = 0.01
def_decay      = 1.2     # LR = 80% * LR

# Constants
grad_clip      = 5
bp_len         = 1
EOS            = "</s>"
STUFF          = "*"

# Global var
xp         = None

def main():
    # Preparation
    args      = parse_args()
    init_program_state(args)
    optimizer = optimizers.AdaGrad(lr=args.lr)
    SRC, TRG  = DS.src_voc, DS.trg_voc
    SRC.fill_from_file(args.src)
    TRG.fill_from_file(args.trg)
    model     = select_model(args.model).new(optimizer=optimizer, \
            gradient_clip=grad_clip)

    # Begin Training
    SC, TC, B  = args.src, args.trg, args.batch
    PRE        = lambda x: x.strip().lower().split()
    POST       = lambda x: convert_to_id(x, SRC, TRG)
    EP         = args.epoch
    accum_loss = 0  # Accumulated loss
    bp_ctr     = 0  # counter
    model.init_params()
    for epoch in range(EP):
        trained = 0
        # Start Epoch
        UF.trace("Starting Epoch", epoch)
        for src, trg in UG.parallel_batch(SC, TC, B, PRE, POST):
            output, loss = model.train(src, trg)
            accum_loss  += loss
            report(output, src, trg, SRC, TRG, trained, epoch+1, EP)
                
            # Run truncated BPTT
            if (bp_ctr+1) % bp_len == 0:
                model.update(accum_loss)
                accum_loss = 0
            bp_ctr  += 1
            trained += B
            UF.trace("Trained %d: %f" % (trained, loss.data))

        # Decaying learning rate
        if epoch > 8:
            model.decay_lr(args.decay_factor)
        model.save(sys.stdout)

"""
Utility functions
"""
def init_program_state(args):
    global xp
    xp         = UF.select_wrapper(not args.use_cpu)
    DS.use_gpu = not args.use_cpu
    DS.xp      = xp
    DS.hidden  = args.hidden
    DS.embed   = args.embed
    DS.batch   = args.batch
    DS.input   = args.input
    DS.output  = args.output
    DS.src_voc = Vocab()
    DS.trg_voc = Vocab()
    DS.src_voc[EOS], DS.trg_voc[EOS]
    DS.src_voc[STUFF], DS.trg_voc[STUFF]

def select_model(model):
    if model == "encdec":
        return EncoderDecoder
    else:
        return Attentional

def convert_to_id(batch, src_voc, trg_voc):
    SRC, TRG = src_voc, trg_voc
    max_src = max(len(src) for src, trg in batch)
    max_trg = max(len(trg) for src, trg in batch)
    src_batch = []
    trg_batch = []
    for src, trg in batch:
        swids  = [SRC[x] for x in src]
        swids += [SRC[EOS]] * (max_src - len(swids))
        swids.append(SRC[EOS])
        src_batch.append(swids)
        twids  = [TRG[x] for x in trg]
        twids += [TRG[EOS]] * (max_trg - len(twids))
        twids.append(TRG[EOS])
        trg_batch.append(twids)
    return src_batch, trg_batch

def report(output, src, trg, src_voc, trg_voc, trained, epoch, max_epoch):
    SRC, TRG = src_voc, trg_voc
    for index in range(len(src)):
        source   = SRC.str_rpr(src[index], EOS)
        ref      = TRG.str_rpr(trg[index], EOS)
        out      = TRG.str_rpr(output[index], EOS)
        UF.trace("Epoch (%d/%d) sample %d:\n\tSRC: %s\n\tOUT: %s\n\tREF: %s" % (epoch, max_epoch,\
                index+trained, source, out, ref))

"""
Arguments
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--trg", type=str, required=True)
    parser.add_argument("--hidden", type=int, default=def_hidden)
    parser.add_argument("--embed", type=int, default=def_embed)
    parser.add_argument("--batch", type=int, default=def_batchsize)
    parser.add_argument("--input", type=int, default=def_input)
    parser.add_argument("--output", type=int, default=def_output)
    parser.add_argument("--epoch", type=int, default=def_epoch)
    parser.add_argument("--lr", type=float, default=def_lr)
    parser.add_argument("--decay_factor", type=float,default=def_decay)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--model",choices=["encdec","att"], default="encdec")
    return parser.parse_args()

if __name__ == "__main__":
    main()

