
import chainn
import chainer.links.connection.linear
import chainer.links.connection.embed_id
import chainer.links.connection.lstm
import numpy as np
import unittest

class NMTArgs:
    def __init__(self, **kwargs):
        self.align_out    = None
        self.batch        = 1
        self.beam         = 10
        self.debug        = True
        self.dict_caching = True
        self.dict_method  = "bias"
        self.eos_disc     = 0
        self.gen_limit    = 10 
        self.gpu          = 1
        self.init_model   = ""
        self.one_epoch    = False
        self.optimizer    = "adam"
        self.save_models  = False
        self.seed         = 1
        self.src_dev      = ""
        self.trg_dev      = ""
        self.unk_cut      = 0
        self.use_cpu      = False
        self.verbose      = True

        for key, value in kwargs.items():
            setattr(self, key, value)

class TestCase(unittest.TestCase):
    def setUp(self):
        chainn.util.functions.init_global_environment(38, 0, False)
    
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
        self.assertChainListEqual(x, y)

    def assertChainListEqual(self, x, y):
        self.assertTrue(issubclass(x.__class__, chainer.ChainList))
        self.assertEqual(type(x), type(y))
        self.assertEqual(len(x), len(y))

        for i in range(len(x)):
            with self.subTest(i=i, t=type(x[i])):
                xi, yi = x[i], y[i]
                self.assertLinkEqual(xi, yi)

    def assertLinkEqual(self, xi, yi):
        self.assertEqual(type(xi), type(yi))
        if type(xi) == linear.Linear:
            self.assertLinearEqual(xi, yi)
        elif type(xi) == embed_id.EmbedID:
            self.assertEmbedEqual(xi, yi)
        elif type(xi) == lstm.LSTM:
            self.assertLSTMEqual(xi, yi)
        elif issubclass(xi.__class__, chainer.ChainList):
            self.assertChainListEqual(xi, yi)
        else:
            raise NotImplementedError()
 
    def assertDctEqual(self, x, y):
        self.assertEqual(len(x), len(y))
        for xi in x:
            self.assertTrue(xi in y)
            self.assertEqual(list(x[xi]), list(y[xi]))

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

