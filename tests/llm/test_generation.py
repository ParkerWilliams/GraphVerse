import unittest

import torch

from graphverse.data.preparation import WalkVocabulary
from graphverse.llm.generation import generate_walk
from graphverse.llm.model import WalkTransformer


class TestLLMGeneration(unittest.TestCase):
    def setUp(self):
        self.vocab = WalkVocabulary([[1, 2, 3], [2, 3, 4]])
        self.model = WalkTransformer(
            len(self.vocab), d_model=128, nhead=4, num_layers=2, dim_feedforward=512
        )

    def test_generate_walk(self):
        start_sequence = [1, 2]
        max_length = 10
        generated_walk = generate_walk(
            self.model, start_sequence, max_length, self.vocab
        )

        self.assertIsInstance(generated_walk, list)
        self.assertTrue(len(generated_walk) <= max_length)
        self.assertEqual(generated_walk[:2], start_sequence)


if __name__ == "__main__":
    unittest.main()
