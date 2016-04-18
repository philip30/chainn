import chainer
import chainer.functions as F
import math

from chainer import cuda
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class LinearInterpolationFunction(chainer.function.Function):
    def _check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        W, x, y = inputs
        W  = sigmoid(W)
        yp = W * x + (1-W) * y
        return yp,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        W, x, y =inputs
        W = sigmoid(W)
        out = grad_outputs[0]
        gw = out * (x * W - y * W)
        gx = out * W
        gy = out * (1- W)
        ret = gw.sum().sum().reshape((1,))
        return ret, gx, gy

def linear_interpolation(W, x, y):
    return LinearInterpolationFunction()(W, x, y)

class LinearInterpolation(chainer.Link):
    def __init__(self, init=0.0):
        super(LinearInterpolation, self).__init__()
        self.add_param("W", 1)
        self.W.data[...] = init

    def __call__(self, x, y):
        return linear_interpolation(self.W, x, y)

