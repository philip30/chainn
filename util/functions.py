
import sys
import datetime
import numpy as np
from . import globalvars
from chainer import cuda
from numpy.random import RandomState

# Utility
def trace(*args, debug_level=0):
    if debug_level <= globalvars.DEBUG_LEVEL:
        print(datetime.datetime.now(), '...', *args, file=sys.stderr)
        sys.stderr.flush()

def print_argmax(data, file=sys.stdout):
    data = cuda.to_cpu(data).argmax(1)
    for x in data:
        print(x, file=file)

def print_classification(data, trg, file=sys.stdout):
    data = cuda.to_cpu(data).argmax(1)
    for x in data:
        print(trg.tok_rpr(x), file=file)

def argmax(data):
    data = cuda.to_cpu(data).argmax(1)
    return [x for x in data]

# SMT decoder model
def select_model(name, all_models):
    for pot_model in all_models:
        if name == pot_model.name:
            return pot_model
    raise NotImplementedError(name)

# Argparse
def check_positive(value, cast=float):
    ivalue = cast(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive value" % value)
    return ivalue

