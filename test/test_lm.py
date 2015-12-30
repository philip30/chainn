import unittest

from chainn.test import TestCase
from chainn.util import load_lm_data, Vocabulary

class TestLM(TestCase):
    def setUp(self):
        pass

    def test_read_train(self):
        train=["I am Philip", "I am student"]
        word, next_word, X, ids = load_lm_data(train,cut_threshold=1)
        print(word)
        print(next_word)
        print(X)
        print(ids)
if __name__ == '__main__':
    unittest.main()
