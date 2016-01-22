import unittest
import numpy as np

from chainer import optimizers, Variable

from chainn.util import Vocabulary, ModelFile, load_nmt_train_data
from chainn.test import TestCase
from chainn.model.nmt import EffectiveAttentional

class Args:
    def __init__(self, X, Y):
        self.input = len(X)
        self.output = len(Y)
        self.hidden = 5
        self.embed = 5
        self.depth = 5

class TestEfAttn(TestCase):
    def setUp(self):
        src=["I am Philip", "I am a student"]
        trg=["私 は フィリップ です", "私 は 学生 です"]
        SRC, TRG, data = load_nmt_train_data(src, trg, cut_threshold=0, batch_size=len(src))
        self.model = EffectiveAttentional(SRC, TRG, Args(SRC,TRG))
        self.data = data

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
