import unittest

import networkx as nx

from graphverse.graph.rules import (
    define_ascenders,
    define_descenders,
    define_evens_odds,
)
from graphverse.graph.walk import generate_multiple_walks, generate_valid_walk


class TestGraphWalk(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from([(i, i + 1) for i in range(99)])
        self.ascenders = define_ascenders(self.graph, 100)
        self.descenders = define_descenders(self.graph, 100)
        self.evens, self.odds = define_evens_odds(self.graph, 100)

    def test_generate_valid_walk(self):
        walk = generate_valid_walk(
            self.graph, 0, 5, 10, self.ascenders, self.descenders, self.evens, self.odds
        )
        self.assertIsNotNone(walk)
        self.assertTrue(5 <= len(walk) <= 10)

    def test_generate_multiple_walks(self):
        walks = generate_multiple_walks(
            self.graph,
            10,
            5,
            10,
            self.ascenders,
            self.descenders,
            self.evens,
            self.odds,
        )
        self.assertEqual(len(walks), 10)
        for walk in walks:
            self.assertTrue(5 <= len(walk) <= 10)


if __name__ == "__main__":
    unittest.main()
