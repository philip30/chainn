import unittest
import random

from chainn.test import TestCase
from chainn.util import BatchManager

class TestBatchManager(TestCase):
    def setUp(self):
        random.seed(17)
        self.manager = BatchManager()

    def test_load_batch_data(self):
        data = ["1st", "2nd", "3rd", "4th", "5th"]
        self.manager.load(data, n_items = 2)
        batches = self.manager
        self.assertEqual(len(batches[0].data), 2)
        self.assertEqual(len(batches[1].data), 2)
        self.assertEqual(len(batches[2].data), 1)

        self.assertEqual(batches[0].data[0], "1st")
        self.assertEqual(batches[0].data[1], "2nd")
        self.assertEqual(batches[1].data[0], "3rd")
        self.assertEqual(batches[1].data[1], "4th")
        self.assertEqual(batches[2].data[0], "5th")
    
    def test_shuffle(self):
        data = ["1st", "2nd", "3rd", "4th", "5th"]
        self.manager.load(data, n_items = 1)
        self.manager.shuffle()
        batches = self.manager
        self.assertEqual(batches[0].data[0], "1st")
        self.assertEqual(batches[1].data[0], "3rd")
        self.assertEqual(batches[2].data[0], "2nd")
        self.assertEqual(batches[3].data[0], "4th")
        self.assertEqual(batches[4].data[0], "5th")

    def test_iter(self):
        data = ["1st", "2nd", "3rd", "4th", "5th"]
        expected = ["1st", "3rd", "2nd", "4th", "5th"]
        self.manager.load(data, n_items = 1)
        self.manager.shuffle()
        for i, batch in enumerate(self.manager):
            self.assertEqual(batch.data[0], expected[i])

    def test_arrange(self):
        data = ["1st", "2nd", "3rd", "4th", "5th"]
        expected = ["5th", "1st", "2nd", "4th", "3rd"]
        self.manager.load(data, n_items = 1)
        self.manager.arrange([4, 0, 1, 3, 2])
        for i, batch in enumerate(self.manager):
            self.assertEqual(batch.data[0], expected[i])
    
    def test_even_data(self):
        data = ["1st", "2nd", "3rd", "4th"]
        self.manager.load(data, n_items = 2)
        self.assertEqual(len(self.manager), 2)
    
    def test_transform_integer(self):
        class TransformInteger:
            def transform(self, s):
                if s[0] == "1st":
                    return 1
                elif s[0] == "2nd":
                    return 2
                elif s[0] == "3rd":
                    return 3
                elif s[0] == "4th":
                    return 4
                else:
                    return 5
        self.manager.transformer = TransformInteger()
        data = ["1st", "2nd", "3rd", "4th", "5th"]
        expected = [1, 2, 3, 4, 5]
        self.manager.load(data, n_items = 1)
        for i, batch in enumerate(self.manager):
            self.assertEqual(batch.data, expected[i])

if __name__ == "__main__":
    unittest.main()
