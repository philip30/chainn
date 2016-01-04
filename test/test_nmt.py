import unittest

from chainn.test import TestCase
from chainn.util import load_nmt_train_data, Vocabulary

class TestNMT(TestCase):
    def setUp(self):
        pass

    def test_NMT_read_train(self):
        src=["I am Philip", "I am a student"]
        trg=["私 は フィリップ です", "私 は 学生 です"]
        x_data, y_data, SRC, TRG = load_nmt_train_data(src, trg, cut_threshold=1)
        x_exp = Vocabulary(unk=True, eos=True)
        y_exp = Vocabulary(unk=True, eos=True)
        
        for w in "i am".split():
            x_exp[w]

        for w in "私 は です".split():
            y_exp[w]
        x_data_exp = [\
                [[x_exp["i"], x_exp["am"], x_exp.unk_id(), x_exp.eos_id()]], \
                [[x_exp["i"], x_exp["am"], x_exp.unk_id(), x_exp.unk_id(), x_exp.eos_id()]] \
        ]

        y_data_exp = [\
                [[y_exp["私" ], y_exp["は" ], y_exp.unk_id(), y_exp["です"], y_exp.eos_id()]], \
                [[y_exp["私" ], y_exp["は" ], y_exp.unk_id(), y_exp["です"], y_exp.eos_id()]] \
        ]

        self.assertVocEqual(SRC, x_exp)
        self.assertVocEqual(TRG, y_exp)
        self.assertEqual(x_data, x_data_exp)
        self.assertEqual(y_data, y_data_exp)

if __name__ == '__main__':
    unittest.main()

