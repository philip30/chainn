import unittest
from chainn.test import TestCase, Args
from chainn.machine import NMTTrainer, NMTTester
from chainn.util.io import load_nmt_test_data

class TestDictAttn(TestCase):
    def setUp(self):
        self.train_args = Args(\
                src        = "test/data/nmt.en", \
                trg        = "test/data/nmt.ja", \
                model_out  = "/tmp/chainn/nmt/model/dictattn", \
                model      = "dictattn",\
                hidden     = 100, \
                epoch      = 5, \
                embed      = 100, \
                depth      = 2, \
                dict       = "test/data/dict.txt")
        self.test_args = Args(\
                init_model = ["/tmp/chainn/nmt/model/dictattn"])
    
    def test_1train_encdec(self):
        self.trainer = NMTTrainer(self.train_args)
        self.trainer.train()

    def test_2test_encdec(self):
        self.tester = NMTTester(self.test_args, load_nmt_test_data)
        with open("test/data/nmt-test.en") as inp_file:
            self.tester.test(inp_file)
        
if __name__ == "__main__":
    unittest.main()
