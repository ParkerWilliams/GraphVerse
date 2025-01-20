import unittest

import networkx as nx
import torch

from graphverse.data.preparation import WalkVocabulary, prepare_training_data
from graphverse.graph.rules import (
    define_ascenders,
    define_descenders,
    define_evens_odds,
)


class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from([(i, i + 1) for i in range(99)])
        self.ascenders = define_ascenders(self.graph, 100)
        self.descenders = define_descenders(self.graph, 100)
        self.evens, self.odds = define_evens_odds(self.graph, 100)

    def test_prepare_training_data(self):
        training_data, vocab = prepare_training_data(
            self.graph,
            10,
            5,
            10,
            self.ascenders,
            self.descenders,
            self.evens,
            self.odds,
        )
        self.assertIsInstance(training_data, torch.Tensor)
        self.assertIsInstance(vocab, WalkVocabulary)
        self.assertEqual(training_data.size(0), 10)  # 10 samples

    def test_walk_vocabulary(self):
        walks = [[1, 2, 3], [2, 3, 4]]
        vocab = WalkVocabulary(walks)
        self.assertIn("1", vocab.token2idx)
        self.assertIn("2", vocab.token2idx)
        self.assertIn("3", vocab.token2idx)
        self.assertIn("4", vocab.token2idx)


if __name__ == "__main__":
    unittest.main()
