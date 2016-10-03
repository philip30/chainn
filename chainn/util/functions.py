
import argparse, sys, random, datetime
import numpy as np
from chainer import cuda

# Utility
def init_global_environment(seed, gpu_num, use_cpu):
    if use_cpu or not hasattr(cuda, "cupy"):
        gpu_num = -1
    if gpu_num >= 0 and hasattr(cuda, "cupy"):
        cuda.get_device(gpu_num).use()
    # Init seed and use GPU
    if seed != 0:
        np.random.seed(seed)
        if gpu_num >= 0 and hasattr(cuda, "cupy"):
            cuda.cupy.random.seed(seed)
        random.seed(seed)
    return gpu_num

def trace(*args):
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

def argmax(data):
    data = cuda.to_cpu(data).argmax(axis=1)
    return [x for x in data]

def nargmax(data, top=1):
    data = cuda.to_cpu(data)
    top = min(top, len(data))
    return np.argpartition(data, -top)[-top:]
    
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

def check_non_negative(value, cast=float):
    ivalue = cast(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s is an invalid non-negative value" % value)
    return ivalue


