#!/usr/bin/env python3

import sys, argparse, math
import chainer.functions as F
import chainn.util.functions as UF

from chainer import optimizers
from chainn.util import AlignmentVisualizer
from chainn.util.io import ModelFile, load_lm_data, batch_generator
from chainn.model import LanguageModel
from chainn.machine import ParallelTrainer

parser = argparse.ArgumentParser("Program to train POS Tagger model using LSTM")
positive = lambda x: UF.check_positive(x, int)
positive_decimal = lambda x: UF.check_positive(x, float)
# Required
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
parser.add_argument("--model",type=str,choices=["lstm"], default="lstm", help="Type of model being trained.")
parser.add_argument("--seed", type=int, default=0, help="Seed for RNG. 0 for totally random seed.")
parser.add_argument("--dev", type=str, help="Development data.")
args = parser.parse_args()

if args.use_cpu:
    args.gpu = -1

""" Training """
trainer   = ParallelTrainer(args.seed, args.gpu)

# data
UF.trace("Loading corpus + dictionary")
X, data    = load_lm_data(sys.stdin)
data       = list(batch_generator(data, (X, X), args.batch))
UF.trace("INPUT size:", len(X))
UF.trace("Data loaded.")

# dev data
dev_data = None
if args.dev:
    with open(args.dev) as dev_fp:
        UF.trace("Loading dev data")
        _, dev_data = load_lm_data(dev_fp, X)
        dev_data = list(batch_generator(dev_data, (X, X), args.batch))
        UF.trace("Dev data loaded")

""" Setup model """
UF.trace("Setting up classifier")
opt   = optimizers.Adam()
model = LanguageModel(args, X, X, opt, args.gpu, activation=F.relu, collect_output=args.verbose)

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
    UF.trace("Trained %d: %f, col_size=%d" % (trained, accum_loss, len(trg[0])))

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
    if (epoch + 1) % args.save_len == 0:
        save_model(epoch)        

def onTrainingFinish(epoch):
    if epoch % args.save_len != 0:
        save_model(epoch)
    UF.trace("training complete!")

""" Execute Training loop """
trainer.train(data, model, args.epoch, onEpochStart, onBatchUpdate, onEpochUpdate, onTrainingFinish)

