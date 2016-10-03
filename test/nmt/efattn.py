import unittest
from chainn.test import TestCase, Args
from chainn.model import EnsembleModel
from chainn.machine import NMTTrainer, NMTTester
from chainn.util.io import load_nmt_test_data

class TestAttentional(TestCase):
    def setUp(self):
        self.train_args = Args(\
                src       = "test/data/nmt.en", \
                trg       = "test/data/nmt.ja", \
                model_out = ".tmp/chainn/nmt/model/attn", \
                model     = "attn",\
                hidden    = 100, \
                epoch     = 30, \
                embed     = 100, \
                dropout   = 0.1, \
                depth     = 1)
        self.test_args = Args(init_model=[".tmp/chainn/nmt/model/attn"])
        self.trainer = NMTTrainer(self.train_args)
            
    def test_1train_attentional(self):
        self.trainer.train()

    def test_2test_attentional(self):
        self.tester = NMTTester(self.test_args, load_nmt_test_data)

        m1 = self.trainer.classifier
        m2 = self.tester.classifier
       
        # Check if they are equal
        self.assertEqual(type(m1), type(m2))
        self.assertEqual(type(m2._model), EnsembleModel)
        self.assertEqual(len(m2._model), 1)
        self.assertModelEqual(m2._model[0], m1._model)
            
        # Check if it works
        with open("test/data/nmt-test.en") as inp_file:
            self.tester.test(inp_file)

if __name__ == "__main__":
    unittest.main()
