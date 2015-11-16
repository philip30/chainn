import sys
import datetime
import numpy as np
from . import globalvars
from chainer import cuda

def trace(*args, debug_level=0):
    if debug_level <= globalvars.DEBUG_LEVEL:
        print(datetime.datetime.now(), '...', *args, file=sys.stderr)
        sys.stderr.flush()

def init_model_parameters(model, minimum=-0.1, maximum=0.1):
    for param in model.parameters:
        param[:] = np.random.uniform(minimum, maximum, param.shape)

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
