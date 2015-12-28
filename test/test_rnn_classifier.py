import unittest
import numpy as np

from chainer import optimizers

from chainn.util import Vocabulary, ModelFile
from chainn.test import TestCase
from chainn.model import RNNParallelSequence

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

class TestRNNClassifier(TestCase):
    def setUp(self):
        src_voc = Vocabulary()
        trg_voc = Vocabulary()
        for tok in "I am Philip".split():
            src_voc[tok]
        for tok in "私 は フィリップ です".split():
            trg_voc[tok]
        self.model = RNNParallelSequence(Args(), src_voc, trg_voc, optimizer=optimizers.SGD())
        self.src_voc = src_voc
        self.trg_voc = trg_voc

    def test_read_write(self):
        model = "/tmp/model.temp"
        X, Y  = self.src_voc, self.trg_voc
        
        # Train with 1 example
        inp = np.array([[X["Philip"], X["I"]]], dtype=np.int32)
        out = np.array([[Y["フィリップ"], Y["私"]]], dtype=np.int32)
        self.model.train(inp, out)
        
        # Save
        with ModelFile(open(model, "w")) as fp:
            self.model.save(fp)

        # Load
        model1 = RNNParallelSequence(InitArgs(model))
            
        # Check
        self.assertRNNEqual(self.model._model.predictor, model1._model.predictor)

if __name__ == "__main__":
    unittest.main()
