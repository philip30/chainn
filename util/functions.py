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

def convert_to_GPU(use_gpu, model):
    if use_gpu:
        cuda.check_cuda_available()
        cuda.get_device(0).use()
        return model.to_gpu()
    else:
        return model

def to_cpu(use_gpu, array):
    if use_gpu:
        return cuda.to_cpu(array)
    else:
        return array

def select_wrapper(use_gpu):
    if not use_gpu:
        return np
    else:
        return cuda.cupy

# SMT decoder model
def init_model_parameters(model, minimum=-0.1, maximum=0.1, seed=0):
    prng = RandomState(seed)
    for param in model.parameters:
        param[:] = prng.uniform(minimum, maximum, param.shape)
    print(model.w_E.W.data)


def select_model(model):
    from model.encdec import EncoderDecoder
    from model.attentional import Attentional
    
    if model == "encdec":
        return EncoderDecoder
    else:
        return Attentional

# Serialization
def vtos(v, fmt='%.8e'):
    return ' '.join(fmt % x for x in v)

def stov(s, tp=float):
    return [tp(x) for x in s.split()]

# Argparse
def check_positive(value, cast=float):
    ivalue = cast(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive value" % value)
    return ivalue

