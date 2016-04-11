import unittest
from os import path
import numpy as np

from chainer import optimizers, Variable, cuda

from chainn.util import Vocabulary
from chainn.util.io import ModelFile, load_nmt_train_data, batch_generator
from chainn.test import TestCase
from chainn.classifier import EncDecNMT
from chainn.model.nmt import DictAttentional

class Args:
    def __init__(self, X, Y, method="bias"):
        self.input = len(X)
        self.output = len(Y)
        self.hidden = 5
        self.embed = 5
        self.depth = 5
        self.dict = path.join(path.dirname(__file__), "data/dict.txt")
        self.dict_caching = True
        self.dict_method = method

class InitArgs(object):
    def __init__(self, init):
        self.init_model = init

class TestDictAttn(TestCase):
    def setUp(self):
        src=["I am Philip", "I am a student"]
        trg=["私 は フィリップ です", "私 は 学生 です"]
        SRC, TRG, data = load_nmt_train_data(src, trg, cut_threshold=0)
        self.model = DictAttentional(SRC, TRG, Args(SRC,TRG))
        self.model_lin = DictAttentional(SRC, TRG, Args(SRC, TRG, "linear"))
        self.data = data
        self.SRC = SRC
        self.TRG = TRG

    def exec_dict(self, model):
        for src, trg in batch_generator(self.data, (self.SRC, self.TRG)):
            model.reset_state(src, trg)
            for j in range(len(trg[0])):
                trg_j = Variable(np.array([trg[i][j] for i in range(len(trg))], dtype=np.int32))
                model(src, trg_j)
        return True
    
    def test_dictattn_call(self):
        return self.exec_dict(self.model)
   
    def test_dictattn_linear_call(self):
        return self.exec_dict(self.model_lin)


    def test_dictattn_readwrite(self):
        model = self.model

        model_out = "/tmp/model-dictattn.temp"

        with ModelFile(open(model_out, "w")) as fp:
            model.save(fp)
       
        args = InitArgs(model_out)

        with ModelFile(open(model_out)) as fp:
            name = fp.read()
            model1 = model.load(fp, DictAttentional, args, np)
    
        self.assertModelEqual(model, model1)
        self.assertDctEqual(model._dict, model1._dict)

if __name__ == "__main__":
    unittest.main()
