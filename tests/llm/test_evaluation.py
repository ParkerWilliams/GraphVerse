import unittest

import networkx as nx

from graphverse.data.preparation import WalkVocabulary
from graphverse.graph.rules import (
    define_ascenders,
    define_descenders,
    define_evens_odds,
)
from graphverse.llm.evaluation import count_rule_violations, evaluate_model
from graphverse.llm.model import WalkTransformer


class TestLLMEvaluation(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from([(i, i + 1) for i in range(99)])
        self.vocab = WalkVocabulary([[i, i + 1] for i in range(99)])
        self.model = WalkTransformer(
            len(self.vocab), d_model=128, nhead=4, num_layers=2, dim_feedforward=512
        )
        self.ascenders = define_ascenders(self.graph, 100)
        self.descenders = define_descenders(self.graph, 100)
        self.evens, self.odds = define_evens_odds(self.graph, 100)

    def test_evaluate_model(self):
        results = evaluate_model(
            self.model,
            self.graph,
            self.vocab,
            num_samples=10,
            min_start_length=1,
            max_start_length=5,
            ascenders=self.ascenders,
            descenders=self.descenders,
            evens=self.evens,
            odds=self.odds,
        )

        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIn("start_length", result)
            self.assertIn("generated_length", result)
            self.assertIn("rule_violations", result)

    def test_count_rule_violations(self):
        walk = [1, 2, 3, 4, 5]
        violations = count_rule_violations(
            walk, self.graph, self.ascenders, self.descenders, self.evens, self.odds
        )
        self.assertIsInstance(violations, int)


if __name__ == "__main__":
    unittest.main()
