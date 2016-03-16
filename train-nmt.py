#!/usr/bin/env python3

import argparse, math
import chainn.util.functions as UF

from chainer import optimizers
from chainn.util import AlignmentVisualizer
from chainn.util.io import ModelFile, load_nmt_train_data
from chainn.model import EncDecNMT
from chainn.machine import ParallelTrainer

""" Arguments """
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
parser.add_argument("--batch", type=positive, default=512)
parser.add_argument("--epoch", type=positive, default=100)
parser.add_argument("--depth", type=positive, default=1)
parser.add_argument("--save_len", type=positive_decimal, default=1)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--init_model", type=str)
parser.add_argument("--model",type=str,choices=["encdec","attn","dictattn"], default="attn")
parser.add_argument("--unk_cut", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
# DictAttn
parser.add_argument("--dict",type=str)
args = parser.parse_args()

""" Sanity Check """
if args.model == "dictattn":
    if not args.dict:
        raise ValueError("When using dict attn, you need to specify the (--dict) lexical dictionary files.")
else:
    if args.dict:
        raise ValueError("When not using dict attn, you do not need to specify the dictionary.")

if args.use_cpu:
    args.gpu = -1

""" Training """
trainer   = ParallelTrainer(args.seed)
 
# data
UF.trace("Loading corpus + dictionary")
SRC, TRG, train_data = trainer.load_data(args.src, args.trg, load_nmt_train_data, args.batch, args.unk_cut)
UF.trace("SRC size:", len(SRC))
UF.trace("TRG size:", len(TRG))
UF.trace("Data loaded.")

# Setup model
UF.trace("Setting up classifier")
opt   = optimizers.Adam()
model = EncDecNMT(args, SRC, TRG, opt, args.gpu, collect_output=args.verbose)

# Training Callback
def onEpochStart(epoch):
    UF.trace("Starting Epoch", epoch+1)

def report(output, src, trg, trained, epoch):
    for index in range(len(src)):
        source   = SRC.str_rpr(src[index])
        ref      = TRG.str_rpr(trg[index])
        out      = TRG.str_rpr(output.y[index])
        UF.trace("Epoch (%d/%d) sample %d:\n\tSRC: %s\n\tOUT: %s\n\tREF: %s" % (epoch+1, args.epoch, index+trained, source, out, ref))

def onBatchUpdate(output, src, trg, trained, epoch, accum_loss):
    if args.verbose:
        report(output, src, trg, trained, epoch)
    UF.trace("Trained %d: %f, col_size=%d" % (trained, accum_loss, len(trg[0])-1)) # minus the last </s>

def save_model():
    UF.trace("saving model to " + args.model_out + "...")
    with ModelFile(open(args.model_out, "w")) as model_out:
        model.save(model_out)

def onEpochUpdate(epoch_loss, epoch_accuracy, prev_loss, epoch):
    UF.trace("Train Loss:", float(prev_loss), "->", float(epoch_loss))
    UF.trace("Train PPL:", math.exp(float(prev_loss)), "->", math.exp(float(epoch_loss)))
    UF.trace("Train Accuracy:", float(epoch_accuracy))

    # saving model
    if (epoch + 1) % args.save_len == 0:
        save_model()        

def onTrainingFinish(epoch):
    if epoch % args.save_len != 0:
        save_model()
    UF.trace("training complete!")

# Execute Training loop
trainer.train(train_data, model, args.epoch, onEpochStart, onBatchUpdate, onEpochUpdate, onTrainingFinish)

