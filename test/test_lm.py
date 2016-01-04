import unittest

from chainn.test import TestCase
from chainn.util import load_lm_data, Vocabulary

class TestLM(TestCase):
    def setUp(self):
        pass

    def test_read_train(self):
        train=["I am Philip", "I am student"]
        word, next_word, X, ids = load_lm_data(train,cut_threshold=1)
        x_exp = Vocabulary()
        for w in "<s> </s> i am".split():
            x_exp[w]

        word_exp = [\
                [[x_exp["<s>"], x_exp["i"], x_exp["am"], x_exp.unk_id()]], \
                [[x_exp["<s>"], x_exp["i"], x_exp["am"], x_exp.unk_id()]] \
        ]

        next_word_exp = [\
                [[x_exp["i"], x_exp["am"], x_exp.unk_id(), x_exp["</s>"]]], \
                [[x_exp["i"], x_exp["am"], x_exp.unk_id(), x_exp["</s>"]]] \
        ]

        self.assertVocEqual(X, x_exp)
        self.assertEqual(word, word_exp)
        self.assertEqual(next_word, next_word_exp)

if __name__ == '__main__':
    unittest.main()

