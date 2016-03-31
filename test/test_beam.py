import unittest
import numpy as np
from chainer import optimizers
from chainn.test import TestCase
from chainn.util import Vocabulary
from chainn.util.io import load_nmt_train_data, ModelFile
from chainn.classifier import EncDecNMT

class Args(object):
    def __init__(self, model):
        self.hidden = 5
        self.use_cpu = True
        self.embed = 6
        self.model = model
        self.depth = 2
        self.init_model = False

class InitArgs(object):
    def __init__(self, init):
        self.init_model = init

class TestBeam(TestCase):
    def test_beam(self):
        src_voc = Vocabulary()
        trg_voc = Vocabulary()
        for tok in "</s> I am Philip You are a".split():
            src_voc[tok]
        for tok in "</s> 私 は フィリップ です 1 2 3".split():
            trg_voc[tok]
        model = EncDecNMT(Args("attn"), src_voc, trg_voc, optimizer=optimizers.SGD())

        model_out = "/tmp/model-nmt.temp"
        X, Y  = src_voc, trg_voc
        
        # Train with 1 example
        src = np.array([[X["I"], X["am"], X["Philip"]]], dtype=np.int32)
        trg = np.array([[Y["私"], Y["は"], Y["フィリップ"], Y["です"]]], dtype=np.int32)
        
        model.train(src, trg)
            
        # Save
        with ModelFile(open(model_out, "w")) as fp:
            model.save(fp)

        # Load
        model1 = EncDecNMT(InitArgs(model_out))
        model.classify(src, beam=10, beam_pick=5)

if __name__ == "__main__":
    unittest.main()
