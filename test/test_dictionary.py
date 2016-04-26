import unittest
import tempfile

from chainn import Vocabulary
from chainn.util.io import ModelSerializer
from chainn.test import TestCase

class TestDictionary(TestCase):
    def test_read_write(self):
        voc = Vocabulary()
        voc["i"], voc["am"], voc["philip"]
        
        # writing
        model = "/tmp/dictionary"
        serializer = ModelSerializer(model)
        with open(model, "w") as voc_fp:
            serializer._write_vocabulary(voc, voc_fp)
        
        with open(model) as voc_fp:
            voc_read = serializer._read_vocabulary(voc_fp)
        
        self.assertVocEqual(voc, voc_read)

if __name__ == "__main__":
    unittest.main()
