#!/usr/bin/env python

import sys, argparse, math, gc, chainer
import chainer.functions as F
import chainn.util.functions as UF

from collections import defaultdict
from chainn.util import Vocabulary as Vocab, load_nmt_train_data, ModelFile, AlignmentVisualizer
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model",type=str,choices=["encdec","attn","dictattn"], default="attn")
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("--unk_cut", type=int, default=1)
    # DictAttn
    parser.add_argument("--dict",type=str)
    return parser.parse_args()

def main():
    # Preparation
    args      = check_args(parse_args())
    
    # data
    UF.trace("Loading corpus + dictionary")
    with open(args.src) as src_fp:
        with open(args.trg) as trg_fp:
            SRC, TRG, data = load_nmt_train_data(src_fp, trg_fp, batch_size=args.batch, cut_threshold=args.unk_cut, debug=args.debug)

    # Setup model
    UF.trace("Setting up classifier")
    opt   = optimizers.Adam()
    model = EncDecNMT(args, SRC, TRG, opt, args.gpu, collect_output=args.verbose)

    # Begin Training
    UF.trace("Begin training NMT")
    EP         = args.epoch
    save_ctr   = 0  # save counter
    save_len   = args.save_len
    prev_loss  = 1e10
    for epoch in range(EP):
        trained = 0
        epoch_loss = 0
        epoch_accuracy = 0
        # Training from the corpus
        UF.trace("Starting Epoch", epoch+1)
        for src, trg in data:
            accum_loss, accum_acc, output = model.train(src, trg)
            epoch_loss += accum_loss
            epoch_accuracy += accum_acc
            # Reporting
            if args.verbose:
                report(output, src, trg, SRC, TRG, trained, epoch+1, EP)
            trained += len(src)
            UF.trace("Trained %d: %f, col_size=%d" % (trained, accum_loss, len(trg[0])-1)) # minus the </s>
            model.report()
        epoch_loss /= len(data)
        epoch_accuracy /= len(data)

        # Decaying learning rate
        if (prev_loss < epoch_loss or epoch > 10) and hasattr(opt,'lr'):
            try:
                opt.lr *= 0.5
                UF.trace("Reducing LR:", opt.lr)
            except: pass
        prev_loss = epoch_loss
       
        UF.trace("Epoch Loss:", float(epoch_loss))
        UF.trace("Epoch Accuracy:", float(epoch_accuracy))
        UF.trace("PPL:", math.exp(float(epoch_loss)))

        # saving model
        if (save_ctr + 1) % save_len == 0:
            UF.trace("saving model to " + args.model_out + "...")
            with ModelFile(open(args.model_out, "w")) as model_out:
                model.save(model_out)

        gc.collect()
        save_ctr += 1
   
    if (save_ctr +1) % save_len != 0:
        with ModelFile(open(args.model_out, "w")) as model_out:
            model.save(model_out)

def report(output, src, trg, src_voc, trg_voc, trained, epoch, max_epoch):
    SRC, TRG = src_voc, trg_voc
    for index in range(len(src)):
        source   = SRC.str_rpr(src[index])
        ref      = TRG.str_rpr(trg[index])
        out      = TRG.str_rpr(output.y[index])
        UF.trace("Epoch (%d/%d) sample %d:\n\tSRC: %s\n\tOUT: %s\n\tREF: %s" % (epoch, max_epoch,\
                index+trained, source, out, ref))
   
    if output.a is not None:
        AlignmentVisualizer.print(output.a, trained, src, output.y, SRC, TRG, sys.stderr)

def check_args(args):
    if args.model == "dictattn":
        if not args.dict:
            raise ValueError("When using dict attn, you need to specify the (--dict) lexical dictionary files.")
    else:
        if args.dict:
            raise ValueError("When not using dict attn, you do not need to specify the dictionary.")
    if args.model == "attn":
        if args.depth > 1:
            raise ValueError("Currently depth is not supported for both of these models")
    
    if args.use_cpu:
        args.gpu = -1

    return args

if __name__ == "__main__":
    main()

