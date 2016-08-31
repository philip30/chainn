import unittest
from chainn.test import TestCase
from chainn.util import Vocabulary

def add_words(vocab):
    vocab.add_word("this")
    vocab.add_word("is")
    vocab.add_word("a")
    vocab.add_word("test")
    return vocab

class TestVocabulary(TestCase):
    def setUp(self):
        self.vocab = add_words(Vocabulary())

    def test_add_word(self):
        for word_id, word in enumerate(["this", "is", "a", "test"]):
            self.assertEqual(self.vocab[word], word_id)

    def test_get_sentence(self):
        sentence = self.vocab.sentence([0, 1, 2, 3])
        self.assertEqual(sentence, "this is a test")
    
    def test_get_word(self):
        self.assertEqual(self.vocab.word(0), "this")

class TestVocabularyPlus(TestCase):
    def setUp(self):
        self.vocab = add_words(Vocabulary(\
                add_eos=True, add_unk=True, add_stuff=True))

    def test_unk_exists(self):
        self.assertEqual(self.vocab.unk_id(), 0)
   
    def test_eos_exists(self):
        self.assertEqual(self.vocab.eos_id(), 1)

    def test_stuff_exists(self):
        self.assertEqual(self.vocab.stuff_id(), 2)

    def test_get_sentence(self):
        word_ids = [3, 4, 5, 6] # 0, 1, 2 are reserved for eos, unk, stuff
        sentence = self.vocab.sentence(word_ids)
        self.assertEqual(sentence, "this is a test")
    
    def test_eos_in_middle(self):
        word_ids = [3, 4, self.vocab.eos_id(), 2]
        sentence = self.vocab.sentence(word_ids)
        self.assertEqual(sentence, "this is")

if __name__ == "__main__":
    unittest.main()
