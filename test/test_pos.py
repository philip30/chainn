import unittest

from os import path
from subprocess import check_call

from chainn.test import TestCase
from chainn.util import Vocabulary
from chainn.util.io import load_pos_train_data, load_pos_test_data

class TestPOS(TestCase):

    def setUp(self):
        self.data = path.join(path.dirname(__file__), "data")
        self.script = path.join(path.dirname(__file__),"script")

    def test_read_train(self):
        train = ["I_NNP am_VBZ Philip_NNP", "I_NNP am_VBZ student_NN"]
        X, Y, data = load_pos_train_data(train)
       
        data = list(data)
        # Check Vocabulary
        x_exp, y_exp = Vocabulary(), Vocabulary(unk=False)
        x_exp["I"], x_exp["am"]
        y_exp["NNP"], y_exp["VBZ"], y_exp["NNP"], y_exp["NN"]

        self.assertVocEqual(X, x_exp)
        self.assertVocEqual(Y, y_exp)
        
        # Check data
        word_exp = [\
                [x_exp["I"], x_exp["am"], x_exp.unk_id()],\
                [x_exp["I"], x_exp["am"], x_exp.unk_id()]\
        ]

        label_exp = [\
                [y_exp["NNP"], y_exp["VBZ"], y_exp["NNP"]],\
                [y_exp["NNP"], y_exp["VBZ"], y_exp["NN"]]\
        ]

        data_exp = [(x,y) for x, y in zip(word_exp, label_exp)]

        self.assertEqual(data, data_exp)

    def test_read_test(self):
        test = ["I live in Japan"]
        X = Vocabulary()
        X["I"], X["live"], X["in"]

        data = list(load_pos_test_data(test, X))[0][0]

        data_exp = [\
                X["I"], X["live"], X["in"], X.unk_id()\
        ]
        self.assertEqual(data, data_exp)

    def test_pos_run(self):
        print("----- Testing train+using pos -----")
        script    = path.join(self.script, "execute_pos.sh")
        inp       = path.join(self.data, "pos.train")
        test      = path.join(self.data, "pos.test")
        train_pos = path.join("train-pos.py")
        test_pos  = path.join("pos.py")
        check_call([script, inp, test, train_pos, test_pos])

if __name__ == "__main__":
    unittest.main()

