import unittest
import chainn
import numpy as np

from chainer import ChainList
from chainer.links.connection.linear import Linear
from chainer.links.connection.embed_id import EmbedID
from chainer.links.connection.lstm import LSTM

class TestCase(unittest.TestCase):
    def assertVocEqual(self, x, y):
        self.assertEqual(x._data, y._data)
        self.assertEqual(x._back, y._back)

    def assertModelEqual(self, x, y):
        self.assertEqual(x._input, y._input)
        self.assertEqual(x._output, y._output)
        self.assertVocEqual(x._src_voc, y._src_voc)
        self.assertVocEqual(x._trg_voc, y._trg_voc)
        self.assertEqual(x._embed, y._embed)
        self.assertEqual(x._hidden, y._hidden)
        self.assertEqual(str(x._activation), str(y._activation))
        self.assertChainListEqual(x, y)

    def assertChainListEqual(self, x, y):
        self.assertTrue(issubclass(x.__class__, ChainList))
        self.assertEqual(type(x), type(y))
        self.assertEqual(len(x), len(y))

        for i in range(len(x)):
            with self.subTest(i=i, t=type(x[i])):
                xi, yi = x[i], y[i]
                self.assertLinkEqual(xi, yi)

    def assertLinkEqual(self, xi, yi):
        self.assertEqual(type(xi), type(yi))
        if type(xi) == Linear:
            self.assertLinearEqual(xi, yi)
        elif type(xi) == EmbedID:
            self.assertEmbedEqual(xi, yi)
        elif type(xi) == LSTM or type(xi) == chainn.link.component.lstm.LSTM:
            self.assertLSTMEqual(xi, yi)
        elif issubclass(xi.__class__, ChainList):
            self.assertChainListEqual(xi, yi)
        else:
            raise NotImplementedError()
 

    def assertLinearEqual(self, x, y):
        self.assertMatrixEqual(x.W.data, y.W.data)
        self.assertVectorEqual(x.b.data, y.W.data)
    
    def assertEmbedEqual(self, x, y):
        self.assertMatrixEqual(x.W.data, y.W.data)

    def assertLSTMEqual(self, x, y):
        self.assertLinearEqual(x.upward, y.upward)
        self.assertEmbedEqual(x.lateral, y.lateral)

    def assertVectorEqual(self, x, y):
        self.assertEqual(len(x), len(y))
        for xi, yi in zip(x, y):
            self.assertTrue(np.equal(xi, yi).data)

    def assertMatrixEqual(self, x, y):
        self.assertEqual(len(x), len(y))
        for xrow, yrow in zip(x, y):
            self.assertVectorEqual(xrow, yrow)
        
    def assertReturn0(self, signal):
        self.assertEqual(signal, 0)

