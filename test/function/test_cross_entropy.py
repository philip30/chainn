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

from chainn.chainer_component.functions.cross_entropy import CrossEntropy, cross_entropy

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sf = numpy.exp(x)
    sf = sf/numpy.sum(sf, axis=0)
    return sf

class TestCrossEntropy(unittest.TestCase):

    shape = (4, 3)
    backward_atol = 1e-2
    cache_score = True

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        out_shape = (self.shape[0],) + self.shape[2:]
        self.t = numpy.random.randint(0, 3, out_shape).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = functions.softmax(chainer.Variable(x_data))
        t = chainer.Variable(t_data)
        loss = cross_entropy(
            x, t, use_cudnn=use_cudnn, cache_score=self.cache_score)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        self.assertEqual(hasattr(loss.creator, 'y'), self.cache_score)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0.0
        count = 0
        x = numpy.rollaxis(self.x, 1, self.x.ndim).reshape(
            (self.t.size, self.x.shape[1]))
        t = self.t.ravel()
        for xi, ti in six.moves.zip(x, t):
            if ti == -1:
                continue
            log_z = numpy.ufunc.reduce(numpy.logaddexp, xi)
            loss_expect -= (xi - log_z)[ti]
            count += 1

        if count == 0:
            loss_expect = 0.0
        else:
            loss_expect /= count
        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)

    def check_backward(self, x_data, t_data, use_cudnn=True):
        x_data = functions.softmax(chainer.Variable(x_data)).data
        gradient_check.check_backward(
            CrossEntropy(
                use_cudnn=use_cudnn, cache_score=self.cache_score),
            (x_data, t_data), None, eps=0.01, atol=self.backward_atol)

    @condition.retry(1)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.cudnn
    @condition.retry(1)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(1)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)


class TestCrossEntropyRemoveForward(TestCrossEntropy):

    cache_score = False

@testing.parameterize(
    {'t_value': -2, 'valid': False},
    {'t_value': 3,  'valid': False},
    {'t_value': -1, 'valid': True},  # -1 is ignore_label
)
class TestCrossEntropyValueCheck(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 2)).astype(numpy.float32)
        # `0` is required to avoid NaN
        self.t = numpy.array([self.t_value, 0], dtype=numpy.int32)
        self.original_debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.original_debug)

    def check_value_check(self, x_data, t_data, use_cudnn):
        x = functions.softmax(chainer.Variable(x_data))
        t = chainer.Variable(t_data)

        if self.valid:
            # Check if it throws nothing
            cross_entropy(x, t, use_cudnn)
        else:
            with self.assertRaises(ValueError):
                cross_entropy(x, t, use_cudnn)

    def test_value_check_cpu(self):
        self.check_value_check(self.x, self.t, False)

    @attr.gpu
    def test_value_check_gpu(self):
        self.check_value_check(self.x, self.t, False)

    @attr.cudnn
    def test_value_check_gpu_cudnn(self):
        self.check_value_check(cuda.to_gpu(self.x), cuda.to_gpu(self.t), True)


testing.run_module(__name__, __file__)
