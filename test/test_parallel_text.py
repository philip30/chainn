import unittest
import numpy as np

from chainer import optimizers

from chainn.util import Vocabulary
from chainn.util.io import ModelSerializer
from chainn.test import TestCase
from chainn.classifier import ParallelTextClassifier

class Args(object):
    def __init__(self):
        self.hidden = 10
        self.use_cpu = True
        self.embed = 20
        self.model = "lstm"
        self.depth = 2
        self.init_model = False

class InitArgs(object):
    def __init__(self, init):
        self.init_model = init

class TestParallelTextClassifier(TestCase):
    def setUp(self):
        src_voc = Vocabulary()
        trg_voc = Vocabulary()
        for tok in "I am Philip".split():
            src_voc[tok]
        for tok in "私 は フィリップ です".split():
            trg_voc[tok]
        self.model = ParallelTextClassifier(Args(), src_voc, trg_voc, optimizer=optimizers.SGD())
        self.src_voc = src_voc
        self.trg_voc = trg_voc

    def test_read_write(self):
        model = "/tmp/chainer-test/text/model.temp"
        X, Y  = self.src_voc, self.trg_voc
        
        # Train with 1 example
        inp = np.array([[X["Philip"], X["I"]]], dtype=np.int32)
        out = np.array([[Y["フィリップ"], Y["私"]]], dtype=np.int32)
        self.model.train(inp, out)
        
        # Save
        serializer = ModelSerializer(model)
        serializer.save(self.model)

        # Load
        model1 = ParallelTextClassifier(InitArgs(model))
            
        # Check
        self.assertModelEqual(self.model._model, model1._model)

if __name__ == "__main__":
    unittest.main()
