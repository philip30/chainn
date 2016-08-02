import unittest
import numpy as np
from chainer import optimizers
from chainn.test import TestCase
from chainn.util import Vocabulary
from chainn.util.io import load_nmt_train_data, ModelSerializer
from chainn.classifier import EncDecNMT

class Args(object):
    def __init__(self, model, seed):
        self.hidden = 5
        self.use_cpu = True
        self.embed = 6
        self.model = model
        self.depth = 2
        self.seed = seed
        self.init_model = False

class InitArgs(object):
    def __init__(self, *model):
        self.init_model = [*model]

class TestEnsemble(TestCase):
    def test_ensemble(self):
        src_voc = Vocabulary()
        trg_voc = Vocabulary()
        for tok in "</s> I am Philip You are a".split():
            src_voc[tok]
        for tok in "</s> 私 は フィリップ です 1 2 3".split():
            trg_voc[tok]
        model1 = EncDecNMT(Args("attn", 13), src_voc, trg_voc, optimizer=optimizers.SGD())
        model2 = EncDecNMT(Args("attn", 5), src_voc, trg_voc, optimizer=optimizers.SGD())
        
        model_out1 = "/tmp/model1-nmt.temp"
        model_out2 = "/tmp/model2-nmt.temp"
        X, Y  = src_voc, trg_voc
        
        # Train with 1 example
        src = np.array([[X["I"], X["am"], X["Philip"]]], dtype=np.int32)
        trg = np.array([[Y["私"], Y["は"], Y["フィリップ"], Y["です"]]], dtype=np.int32)
        
        model1.train(src, trg)
        model2.train(src, trg)
            
        # Save
        serializer = ModelSerializer(model_out1)
        serializer.save(model1)
        serializer = ModelSerializer(model_out2)
        serializer.save(model2)

        # Load
        ens    = EncDecNMT(InitArgs(model_out1, model_out2))._model
        model1 = EncDecNMT(InitArgs(model_out1))._model[0]
        model2 = EncDecNMT(InitArgs(model_out2))._model[0]
        
        ens.reset_state(src)
        model1.reset_state(src)
        model2.reset_state(src)

        y, _ = ens.classify(src)
        y1   = model1(src, is_train=False).y
        y2   = model2(src, is_train=False).y

        np.testing.assert_array_equal(y.data, (y1.data + y2.data)/2)

if __name__ == "__main__":
    unittest.main()
