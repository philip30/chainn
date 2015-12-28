import unittest
from chainn.model import RNN
from chainn.util import ModelFile, Vocabulary
from chainn.test import TestCase

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

        self.model = self.Model(src_voc=src_voc, trg_voc=trg_voc, \
                input=len(src_voc), output=len(trg_voc), \
                hidden=5, depth=1, embed=5)
  
    def test_read_write(self):
        model = "/tmp/rnn.temp"
        with ModelFile(open(model, "w")) as fp:
            self.model.save(fp)

        with ModelFile(open(model)) as fp:
            fp.read()
            model1 = self.model.load(fp, self.model.__class__)
        
        self.assertRNNEqual(self.model, model1)

    def test_init_size(self):
        self.assertEqual(len(self.model), 4) # Input, Embed, Hidden, Output

    def test_depth_size(self):
        model = self.Model(Vocabulary(), Vocabulary(), depth=5, input=1,output=1, embed=1, hidden=1)
        self.assertEqual(len(model), 8) # Input, Embed, 5*Hidden, Output

if __name__ == "__main__":
    unittest.main()

