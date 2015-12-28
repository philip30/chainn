import unittest

from chainn.test import TestCase
from chainn.util import load_pos_train_data, load_pos_test_data, Vocabulary

class TestPOS(TestCase):

    def setUp(self):
        pass

    def test_read_train(self):
        train = ["I_NNP am_VBZ Philip_NNP", "I_NNP am_VBZ student_NN"]
        data, labels, X, Y = load_pos_train_data(train)
        
        # Check Vocabulary
        x_exp, y_exp = Vocabulary(), Vocabulary(unk=False)
        x_exp["I"], x_exp["am"]
        y_exp["NNP"], y_exp["VBZ"], y_exp["NNP"], y_exp["NN"]

        self.assertVocEqual(X, x_exp)
        self.assertVocEqual(Y, y_exp)
        
        # Check data
        data_exp = [\
                [[x_exp["I"], x_exp["am"], x_exp.unk_id()]],\
                [[x_exp["I"], x_exp["am"], x_exp.unk_id()]]\
        ]

        label_exp = [\
                [[y_exp["NNP"], y_exp["VBZ"], y_exp["NNP"]]],\
                [[y_exp["NNP"], y_exp["VBZ"], y_exp["NN"]]]\
        ]

        self.assertEqual(data, data_exp)
        self.assertEqual(labels, label_exp)

    def test_read_test(self):
        test = ["I live in Japan"]
        X = Vocabulary()
        X["I"], X["live"], X["in"]

        data, _ = load_pos_test_data(test, X)

        data_exp = [\
                [[X["I"], X["live"], X["in"], X.unk_id()]]\
        ]
        self.assertEqual(data, data_exp)

if __name__ == "__main__":
    unittest.main()

