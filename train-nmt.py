#!/usr/bin/env python3

import argparse, math
import chainn.util.functions as UF

from chainer import optimizers
from chainn.util import AlignmentVisualizer
from chainn.util.io import ModelFile, load_nmt_train_data, batch_generator
from chainn.classifier import EncDecNMT
from chainn.machine import ParallelTrainer

""" Arguments """
parser = argparse.ArgumentParser("A trainer for NMT model.")
positive = lambda x: UF.check_positive(x, int)
positive_decimal = lambda x: UF.check_positive(x, float)
# Required
parser.add_argument("--src", type=str, required=True)
parser.add_argument("--trg", type=str, required=True)
parser.add_argument("--model_out", type=str, required=True)
# Options
parser.add_argument("--hidden", type=positive, default=128, help="Size of hidden layer.")
parser.add_argument("--embed", type=positive, default=128, help="Size of embedding vector.")
parser.add_argument("--batch", type=positive, default=512, help="Number of (src) words in batch.")
parser.add_argument("--epoch", type=positive, default=10, help="Number of epoch to train the model.")
parser.add_argument("--depth", type=positive, default=1, help="Depth of the network.")
parser.add_argument("--save_len", type=positive, default=1, help="Number of iteration being done for ")
parser.add_argument("--verbose", action="store_true", help="To output the training progress for every sentence in corpora.")
parser.add_argument("--use_cpu", action="store_true", help="Force to use CPU.")
parser.add_argument("--save_models", action="store_true", help="Save models for every iteration with auto enumeration.")
parser.add_argument("--gpu", type=int, default=-1, help="Specify GPU to be used, negative for using CPU.")
parser.add_argument("--init_model", type=str, help="Init the training weights with saved model.")
parser.add_argument("--model",type=str,choices=["encdec","attn","dictattn"], default="attn", help="Type of model being trained.")
parser.add_argument("--unk_cut", type=int, default=1, help="Threshold for words in corpora to be treated as unknown.")
parser.add_argument("--dropout", type=positive_decimal, default=0.2, help="Dropout ratio for LSTM.")
parser.add_argument("--seed", type=int, default=0, help="Seed for RNG. 0 for totally random seed.")
parser.add_argument("--src_dev", type=str)
parser.add_argument("--trg_dev", type=str)
# DictAttn
parser.add_argument("--dict",type=str, help="Tab separated trg give src dictionary")
parser.add_argument("--dict_caching",action="store_true", help="Whether to cache the whole dictionary densely")
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

if args.save_models:
    args.save_len = 1

""" Training """
trainer   = ParallelTrainer(args.seed, args.gpu)
 
# data
UF.trace("Loading corpus + dictionary")
with open(args.src) as src_fp:
    with open(args.trg) as trg_fp:
        SRC, TRG, train_data = load_nmt_train_data(src_fp, trg_fp, cut_threshold=args.unk_cut)
        train_data = list(batch_generator(train_data, (SRC, TRG), args.batch))
UF.trace("SRC size:", len(SRC))
UF.trace("TRG size:", len(TRG))
UF.trace("Data loaded.")

# dev data
dev_data = None
if args.src_dev and args.trg_dev:
    with open(args.src_dev) as src_fp:
        with open(args.trg_dev) as trg_fp:
            UF.trace("Loading dev data")
            _, _, dev_data = load_nmt_train_data(src_fp, trg_fp, SRC, TRG)
            dev_data = list(batch_generator(dev_data, (SRC, TRG), args.batch))
            UF.trace("Dev data loaded")

""" Setup model """
UF.trace("Setting up classifier")
opt   = optimizers.Adam()
model = EncDecNMT(args, SRC, TRG, opt, args.gpu, collect_output=args.verbose)

""" Training Callback """
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

def save_model(epoch):
    out_file = args.model_out
    if args.save_models:
        out_file += "-" + str(epoch)
    UF.trace("saving model to " + out_file + "...")
    with ModelFile(open(out_file, "w")) as model_out:
        model.save(model_out)

def onEpochUpdate(epoch_loss, prev_loss, epoch):
    UF.trace("Train Loss:", float(prev_loss), "->", float(epoch_loss))
    UF.trace("Train PPL:", math.exp(float(prev_loss)), "->", math.exp(float(epoch_loss)))

    if dev_data is not None:
        dev_loss = trainer.eval(dev_data, model)
        UF.trace("Dev Loss:", float(dev_loss))
        UF.trace("Dev PPL:", math.exp(float(dev_loss)))

    # saving model
    if args.save_models and (epoch + 1) % args.save_len == 0:
        save_model(epoch)        

def onTrainingFinish(epoch):
    if not args.save_models or epoch % args.save_len != 0:
        save_model(epoch)
    UF.trace("training complete!")

""" Execute Training loop """
trainer.train(train_data, model, args.epoch, onEpochStart, onBatchUpdate, onEpochUpdate, onTrainingFinish)

