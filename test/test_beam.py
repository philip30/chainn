import numpy as np
import unittest

from chainn.test import TestCase, Args
from chainn.classifier import EncDecNMT

class TestBeam(TestCase):
    def setUp(self):
        self.model = EncDecNMT(Args(\
                init_model = "test/model/attn-model",\
                beam       = 5), \
                collect_output = True)
        

if __name__ == "__main__":
    unittest.main()
