
import argparse
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

def load_stream(fp):
    if fp is None or len(fp) == 0:
        return None
    elif fp == "STDOUT":
        return sys.stdout
    elif fp == "STDERR":
        return sys.stderr
    else:
        return open(fp, "w")

def print_argmax(data, file=sys.stdout):
    data = cuda.to_cpu(data).argmax(1)
    for x in data:
        print(x, file=file)

def setup_gpu(use_gpu):
    ret = None
    if not hasattr(cuda, "cupy"):
        use_gpu  = -1
        ret = np
    else:
        if use_gpu >= 0:
            ret = cuda.cupy
            cuda.get_device(use_gpu).use()
        else:
            ret = np
    return ret, use_gpu

def print_classification(data, trg, file=sys.stdout):
    data = cuda.to_cpu(data).argmax(1)
    for x in data:
        print(trg.tok_rpr(x), file=file)

def argmax(data, number=1):
    data = cuda.to_cpu(data).argmax(number)
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

