import unittest
import numpy as np
from chainn.model.text import RecurrentLSTM
from chainn.util import Vocabulary
from chainn.util.io import ModelSerializer
from chainn.test import TestCase

class Args(object):
    pass

class TestRecurrentLSTM(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRecurrentLSTM, self).__init__(*args, **kwargs)
        self.Model = RecurrentLSTM

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
        serializer = ModelSerializer(model)
        serializer._init_dir()
        serializer._write_model(self.model)
        model1 = serializer._read_model(self.Model)
        
        self.assertModelEqual(self.model, model1)

    def test_init_size(self):
        self.assertEqual(len(self.model.inner) + 2, 3) # embed, hiddenx1, output

    def test_depth_size(self):
        args = Args()
        args.depth = 5
        args.input = 1
        args.output = 1
        args.embed = 1
        args.hidden = 1
        model = self.Model(Vocabulary(), Vocabulary(), args)
        self.assertEqual(len(model.inner) + 2, 7) # Embed, 5*Hidden, Output

if __name__ == "__main__":
    unittest.main()

