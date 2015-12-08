import chainer
import chainer.functions as F

import numpy as np

class LinearInterpolation(chainer.Link):
    def __init__(self):
        super(LinearInterpolation, self).__init__(W=(1,))

    def __call__(self, x):
        return LI(x, self.W)

class LI(chainer.function.Function):
    def forward(self, inputs):
        pass

    def backward(self, inputs, grad_outputs):
        pass

