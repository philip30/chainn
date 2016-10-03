#!/usr/bin/env python3

import chainn
import chainn.eval
import unittest

class TestBLEU(chainn.test.TestCase):
    def test_bleu(self):
        hyp = ['a', 'b', 'c', 'd', 'e']
        ref = ['a', 'c', 'd']

        print(eval.calcluate_blue_corpus(hyp,ref))

if __name__ == '__main__':
    unittest.main()
