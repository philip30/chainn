import unittest
import numpy as np
from chainn.model.basic import RNN
from chainn.util import Vocabulary
from chainn.util.io import ModelFile
from chainn.test import TestCase

class Args(object):
    pass

class TestRNN(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRNN, self).__init__(*args, **kwargs)
        self.Model = RNN

    def setUp(self):
        src_voc = Vocabulary()
        trg_voc = Vocabulary()

        for word in ["my", "name", "is", "philip", "."]:
            src_voc[word]

        for tag in ["PRP", "NN", "VBZ", "NNP", "."]:
            trg_voc[word]
        args = Args()
        args.input  = len(src_voc)
        args.output = len(trg_voc)
        args.hidden = 5
        args.depth  = 1
        args.embed  = 5
        self.model = self.Model(src_voc, trg_voc, args)
  
    def test_read_write(self):
        model = "/tmp/rnn.temp"
        with ModelFile(open(model, "w")) as fp:
            self.model.save(fp)

        with ModelFile(open(model)) as fp:
            fp.read()
            model1 = self.model.load(fp, self.model.__class__, Args(), np)
        
        self.assertModelEqual(self.model, model1)

    def test_init_size(self):
        self.assertEqual(len(self.model), 4) # Input, Embed, Hidden, Output

    def test_depth_size(self):
        args = Args()
        args.depth = 5
        args.input = 1
        args.output = 1
        args.embed = 1
        args.hidden = 1
        model = self.Model(Vocabulary(), Vocabulary(), args)
        self.assertEqual(len(model), 8) # Input, Embed, 5*Hidden, Output

if __name__ == "__main__":
    unittest.main()

