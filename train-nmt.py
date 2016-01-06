#!/usr/bin/env python

import sys
import argparse
import gc
import chainer
import chainer.functions as F
import chainn.util.functions as UF
import chainn.util.generators as UG

from collections import defaultdict
from chainn.util import Vocabulary as Vocab, load_nmt_train_unsorted_data, ModelFile
from chainn.model import EncDecNMT
from chainer import optimizers

def parse_args():
    parser = argparse.ArgumentParser()
    positive = lambda x: UF.check_positive(x, int)
    positive_decimal = lambda x: UF.check_positive(x, float)
    # Required
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--trg", type=str, required=True)
    parser.add_argument("--model_out", type=str, required=True)
    # Options
    parser.add_argument("--hidden", type=positive, default=128)
    parser.add_argument("--embed", type=positive, default=128)
    parser.add_argument("--batch", type=positive, default=64)
    parser.add_argument("--epoch", type=positive, default=100)
    parser.add_argument("--depth", type=positive, default=1)
    parser.add_argument("--lr", type=positive, default=0.01)
    parser.add_argument("--save_len", type=positive_decimal, default=1)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model",type=str,choices=["encdec","attn","efattn"], default="attn")
    parser.add_argument("--debug",action="store_true")
    return parser.parse_args()

def main():
    # Preparation
    args      = parse_args()
    
    # data
    UF.trace("Loading corpus + dictionary")
    with open(args.src) as src_fp:
        with open(args.trg) as trg_fp:
            cut = 1 if not args.debug else 0
            x_data, y_data, SRC, TRG = load_nmt_train_unsorted_data(src_fp, trg_fp, batch_size=args.batch, cut_threshold=cut)
   
    # Setup model
    UF.trace("Setting up classifier")
    opt   = optimizers.AdaGrad(lr=args.lr)
    model = EncDecNMT(args, SRC, TRG, opt, not args.use_cpu, collect_output=True)

    # Hooking
    opt.add_hook(chainer.optimizer.GradientClipping(10))

    # Begin Training
    UF.trace("Begin training NMT")
    EP         = args.epoch
    save_ctr   = 0  # save counter
    save_len   = args.save_len
    prev_loss  = 1e10
    for epoch in range(EP):
        trained = 0
        epoch_loss = 0
        # Training from the corpus
        UF.trace("Starting Epoch", epoch+1)
        for src, trg in zip(x_data, y_data):
            accum_loss, accum_acc, output = model.train(src, trg)
            epoch_loss += accum_loss

            # Reporting
            report(output, src, trg, SRC, TRG, trained, epoch+1, EP)
            trained += len(src)
            UF.trace("Trained %d: %f" % (trained, accum_loss))

        # Decaying learning rate
        if (prev_loss < epoch_loss or epoch > 10) and hasattr(opt,'lr'):
            opt.lr *= 0.5
            UF.trace("Reducing LR:", opt.lr)
        prev_loss = epoch_loss

        # saving model
        if (save_ctr + 1) % save_len == 0:
            UF.trace("saving model to " + args.model_out + "...")
            with ModelFile(open(args.model_out, "w")) as model_out:
                model.save(model_out)

        gc.collect()
        save_ctr += 1

def report(output, src, trg, src_voc, trg_voc, trained, epoch, max_epoch):
    SRC, TRG = src_voc, trg_voc
    for index in range(len(src)):
        source   = SRC.str_rpr(src[index])
        ref      = TRG.str_rpr(trg[index])
        out      = TRG.str_rpr(output[index])
        UF.trace("Epoch (%d/%d) sample %d:\n\tSRC: %s\n\tOUT: %s\n\tREF: %s" % (epoch, max_epoch,\
                index+trained, source, out, ref))

if __name__ == "__main__":
    main()

