import unittest
import numpy as np

from os import path
from subprocess import check_call
from chainer import optimizers

from chainn.util import Vocabulary, ModelFile
from chainn.test import TestCase
from chainn.model import EncDecNMT

class Args(object):
    def __init__(self):
        self.hidden = 10
        self.use_cpu = True
        self.embed = 20
        self.model = "encdec"
        self.depth = 2
        self.init_model = False

class InitArgs(object):
    def __init__(self, init):
        self.init_model = init

class TestEncDecClassifier(TestCase):
    def setUp(self):
        src_voc = Vocabulary()
        trg_voc = Vocabulary()
        for tok in "I am Philip".split():
            src_voc[tok]
        for tok in "私 は フィリップ です".split():
            trg_voc[tok]
        self.model = EncDecNMT(Args(), src_voc, trg_voc, optimizer=optimizers.SGD())
        self.src_voc = src_voc
        self.trg_voc = trg_voc
        self.data = path.join(path.dirname(__file__), "data")
        self.script = path.join(path.dirname(__file__),"script")
 
    def test_encdec_read_write(self):
        model = "/tmp/model-nmt.temp"
        X, Y  = self.src_voc, self.trg_voc
        
        # Train with 1 example
        src = np.array([[X["I"], X["am"], X["Philip"]]], dtype=np.int32)
        trg = np.array([[Y["私"], Y["は"], Y["フィリップ"], Y["です"]]], dtype=np.int32)
        
        self.model.train(src, trg)
        
        # Save
        with ModelFile(open(model, "w")) as fp:
            self.model.save(fp)

        # Load
        model1 = EncDecNMT(InitArgs(model))
            
        # Check
        self.assertModelEqual(self.model._model.predictor, model1._model.predictor)

    def test_nmt_run(self):
        print("----- Testing train+using nmt -----")
        script    = path.join(self.script, "execute_nmt.sh")
        src       = path.join(self.data, "nmt.en")
        trg       = path.join(self.data, "nmt.ja")
        test      = path.join(self.data, "nmt-test.en")
        train_nmt = path.join("train-nmt.py")
        test_nmt  = path.join("nmt.py")
        check_call([script, src, trg, test, train_nmt, test_nmt])


if __name__ == "__main__":
    unittest.main()
