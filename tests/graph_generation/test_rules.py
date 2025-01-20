import unittest

import networkx as nx

from graphverse.graph.rules import (
    check_rule_compliance,
    define_ascenders,
    define_descenders,
    define_evens_odds,
)


class TestGraphRules(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(100))

    def test_define_ascenders(self):
        ascenders = define_ascenders(self.graph, 100)
        self.assertTrue(all(40 <= v <= 60 for v in ascenders))

    def test_define_descenders(self):
        descenders = define_descenders(self.graph, 100)
        self.assertTrue(all(40 <= v <= 60 for v in descenders))

    def test_define_evens_odds(self):
        evens, odds = define_evens_odds(self.graph, 100)
        self.assertTrue(all(v % 2 == 0 for v in evens))
        self.assertTrue(all(v % 2 != 0 for v in odds))

    def test_check_rule_compliance(self):
        ascenders = {50}
        descenders = {60}
        evens = {2, 4}
        odds = {1, 3}

        # Compliant walk
        walk = [1, 2, 3, 4, 50, 51, 52]
        self.assertTrue(
            check_rule_compliance(walk, self.graph, ascenders, descenders, evens, odds)
        )

        # Non-compliant walk (violates ascender rule)
        walk = [1, 2, 50, 49]
        self.assertFalse(
            check_rule_compliance(walk, self.graph, ascenders, descenders, evens, odds)
        )


if __name__ == "__main__":
    unittest.main()
