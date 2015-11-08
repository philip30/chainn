import sys
import datetime
import numpy as np
from . import globalvars
from chainer import cuda

def trace(*args, debug_level=0):
    if debug_level <= globalvars.DEBUG_LEVEL:
        print(datetime.datetime.now(), '...', *args, file=sys.stderr)
        sys.stderr.flush()

def init_model_parameters(model):
    for param in model.parameters:
        param[:] = np.random.uniform(-0.1, 0.1, param.shape)

def convert_to_GPU(use_gpu, model):
    if use_gpu:
        cuda.check_cuda_available()
        cuda.get_device(0).use()
        return model.to_gpu()
    else:
        return model

def select_wrapper(use_gpu):
    if not use_gpu:
        return np
    else:
        return cuda.cupy
