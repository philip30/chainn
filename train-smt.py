#!/usr/bin/env python

import sys
import argparse
import math
import random
import gc
import numpy as inp

import util.functions as UF
import util.generators as UG

from collections import defaultdict
from util.vocabulary import Vocabulary as Vocab
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
def_savelen    = 1

# Constants
grad_clip      = 5
bp_len         = 1
EOS            = "</s>"
STUFF          = "*"

def main():
    # Preparation
    args      = parse_args()
    model     = UF.select_model(args.model)
    optimizer = optimizers.AdaGrad(lr=args.lr)
    model = UF.select_model(args.model)(optimizer=optimizer, gc=grad_clip,\
            hidden=args.hidden, embed=args.embed, input=args.input,\
            use_gpu=not args.use_cpu,\
            output=args.output, src_voc=Vocab(EOS), trg_voc=Vocab(EOS))

    if args.init_model is not None:
        UF.trace("Loading model:", args.init_model)
        with open(args.init_model) as fp:
            model.load(fp)
    else:
        model.init_params()

    # Begin Training
    SRC, TRG   = model.get_vocabularies()
    SC, TC, B  = args.src, args.trg, args.batch
    PRE        = lambda x: x.strip().lower().split()
    POST       = lambda x: convert_to_id(x, SRC, TRG)
    EP         = args.epoch
    accum_loss = 0  # Accumulated loss
    bp_ctr     = 0  # counter
    save_ctr   = 0  # save counter
    save_len   = args.save_len
    SRC.fill_from_file(args.src)
    TRG.fill_from_file(args.trg)
    for epoch in range(EP):
        model.setup_optimizer()
        trained = 0
        # Training from the corpus
        UF.trace("Starting Epoch", epoch+1)
        for src, trg in UG.same_len_batch((SC, TC), B, PRE, POST):
            output, loss = model.train(src, trg)
            accum_loss  += loss
            report(output, src, trg, SRC, TRG, trained, epoch+1, EP)
                
            # Run truncated BPTT
            if (bp_ctr+1) % bp_len == 0:
                model.update(accum_loss)
                accum_loss = 0
            bp_ctr  += 1
            trained += len(src)
            UF.trace("Trained %d: %f" % (trained, loss.data))

        # Decaying learning rate
        #if epoch > 8:
        #    model.decay_lr(args.decay_factor)

        # saving model
        if (save_ctr + 1) % save_len == 0:
            UF.trace("saving model...")
            with open(args.model_out, "w") as fp:
                model.save(fp)
        
        gc.collect()
        save_ctr += 1

"""
Utility functions
"""
def convert_to_id(batch, src_voc, trg_voc):
    SRC, TRG = src_voc, trg_voc
    max_src = max(len(src) for src, trg in batch)
    max_trg = max(len(trg) for src, trg in batch)
    src_batch = []
    trg_batch = []

    # No stuffing
    assert(max_src == len(x) and max_trg == len(y) for x, y in batch)
    
    for src, trg in batch:
        swids  = [SRC[x] for x in src]
        swids.append(SRC[EOS])
        src_batch.append(swids)
        twids  = [TRG[x] for x in trg]
        twids.append(TRG[EOS])
        trg_batch.append(twids)
    return src_batch, trg_batch

def report(output, src, trg, src_voc, trg_voc, trained, epoch, max_epoch):
    SRC, TRG = src_voc, trg_voc
    for index in range(len(src)):
        source   = SRC.str_rpr(src[index])
        ref      = TRG.str_rpr(trg[index])
        out      = TRG.str_rpr(output[index])
        UF.trace("Epoch (%d/%d) sample %d:\n\tSRC: %s\n\tOUT: %s\n\tREF: %s" % (epoch, max_epoch,\
                index+trained, source, out, ref))

"""
Arguments
"""
def parse_args():
    parser = argparse.ArgumentParser()
    positive = lambda x: UF.check_positive(x, int)
    positive_decimal = lambda x: UF.check_positive(x, float)
    # Required
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--trg", type=str, required=True)
    parser.add_argument("--model_out", type=str, required=True)
    # Options
    parser.add_argument("--hidden", type=positive, default=def_hidden)
    parser.add_argument("--embed", type=positive, default=def_embed)
    parser.add_argument("--batch", type=positive, default=def_batchsize)
    parser.add_argument("--input", type=positive, default=def_input)
    parser.add_argument("--output", type=positive, default=def_output)
    parser.add_argument("--epoch", type=positive, default=def_epoch)
    parser.add_argument("--lr", type=positive, default=def_lr)
    parser.add_argument("--decay_factor", type=positive_decimal,default=def_decay)
    parser.add_argument("--save_len", type=positive_decimal, default=def_savelen)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model",choices=["encdec","attn"], default="attn")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

