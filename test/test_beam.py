import unittest
import numpy as np
from chainer import optimizers
from chainn.test import TestCase
from chainn.util import Vocabulary
from chainn.util.io import load_nmt_train_data, ModelSerializer
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
        self.init_model = [init]

class TestBeam(TestCase):
    def test_beam(self):
        src_voc = Vocabulary()
        trg_voc = Vocabulary()
        for tok in "</s> I am Philip You are a".split():
            src_voc[tok]
        for tok in "</s> 私 は フィリップ です 1 2 3".split():
            trg_voc[tok]
        model = EncDecNMT(Args("attn"), src_voc, trg_voc, optimizer=optimizers.Adam())

        model_out = "/tmp/model-nmt.temp"
        X, Y  = src_voc, trg_voc
        
        # Train with 1 example
        src = np.array([[X["I"], X["am"], X["Philip"]]], dtype=np.int32)
        trg = np.array([[Y["私"], Y["は"], Y["フィリップ"], Y["です"]]], dtype=np.int32)
        
        model.train(src, trg)
            
        # Save
        serializer = ModelSerializer(model_out)
        serializer.save(model)

        # Load
        model1 = EncDecNMT(InitArgs(model_out))
        k      = model1.classify(src, beam=2, allow_empty=False)
        self.assertEqual(trg_voc.str_rpr(k.y[0]), "1 1 私")
        
        for elem in k.prob:
            self.assertAlmostEqual(1.0, np.sum(elem, axis=1)[0], places=5)

if __name__ == "__main__":
    unittest.main()
