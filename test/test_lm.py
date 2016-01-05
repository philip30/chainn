import unittest
from os import path
from subprocess import check_call

from chainn.test import TestCase
from chainn.util import load_lm_data, Vocabulary

class TestLM(TestCase):
    def setUp(self):
        self.data = path.join(path.dirname(__file__), "data")
        self.script = path.join(path.dirname(__file__),"script")

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

    def test_lm_run(self):
        print("----- Testing train+using lm -----")
        script   = path.join(self.script, "execute_lm.sh")
        inp      = path.join(self.data, "lm.train")
        dev      = path.join(self.data, "lm.dev")
        test     = path.join(self.data, "lm.test")
        train_lm = path.join("train-lm.py")
        test_lm  = path.join("lm.py")
        check_call([script, inp, dev, test, train_lm, test_lm])


if __name__ == '__main__':
    unittest.main()

