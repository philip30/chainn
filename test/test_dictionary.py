import unittest
import tempfile

from chainn import Vocabulary
from chainn.util.io import ModelFile
from chainn.test import TestCase

class TestDictionary(TestCase):
    def test_read_write(self):
        voc = Vocabulary()
        voc["i"], voc["am"], voc["philip"]
        
        # writing
        model = "/tmp/dict.temp"
        with ModelFile(open(model,"w")) as fp:
            voc.save(fp)

        with ModelFile(open(model)) as fp:
            voc_read = Vocabulary.load(fp)
        
        self.assertVocEqual(voc, voc_read)

if __name__ == "__main__":
    unittest.main()
