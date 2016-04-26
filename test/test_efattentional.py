import unittest
import numpy as np

from chainer import optimizers, Variable

from chainn.util import Vocabulary
from chainn.util.io import ModelSerializer, load_nmt_train_data, batch_generator
from chainn.test import TestCase
from chainn.model.nmt import Attentional

class Args:
    def __init__(self, X, Y):
        self.input = len(X)
        self.output = len(Y)
        self.hidden = 3
        self.embed = 1
        self.depth = 2

class TestEfAttn(TestCase):
    def setUp(self):
        src=["I am Philip .", "I am a student ."]
        trg=["私 は フィリップ です .", "私 は 学生 です ."]
        SRC, TRG, data = load_nmt_train_data(src, trg, cut_threshold=0)
        self.model = Attentional(SRC, TRG, Args(SRC,TRG))
        self.data = batch_generator(data, (SRC, TRG), 1)

    def test_efattn_encode(self):
        model = self.model
        for src, trg in self.data:
            h = model.reset_state(src, trg)
    
    def test_efattn_call(self):
        model = self.model
        for src, trg in self.data:
            model.reset_state(src, trg)
            for j in range(len(trg[0])):
                trg_j = Variable(np.array([trg[i][j] for i in range(len(trg))], dtype=np.int32))
                model(src, trg_j)

if __name__ == "__main__":
    unittest.main()
