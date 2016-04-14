import unittest

import math
import mock
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from chainn.chainer_component.links.linear_interpolation import LinearInterpolation


class TestLinearInterpolation(unittest.TestCase):


    def setUp(self):
        self.link = LinearInterpolation()

        shape  = (4,5)
        self.x1 = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.x2 = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.w = self.link.W.data.reshape(())
        self.gy = numpy.random.uniform(
            -1, 1, (shape)).astype(numpy.float32)
        self.y = self.w * self.x1 + (1-self.w) * self.x2

    def check_forward(self, x1_data, x2_data):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y  = self.link(x1, x2)
        self.assertEqual(y.data.dtype, numpy.float32)
        gradient_check.assert_allclose(self.y, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x1, self.x2)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2))

    def check_backward(self, x1_data, x2_data, gy_data):
        gradient_check.check_backward(self.link, (x1_data, x2_data), gy_data, eps=1e-2, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x1), cuda.to_gpu(self.x2), cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)
