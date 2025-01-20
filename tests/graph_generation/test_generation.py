import unittest

import networkx as nx

from graphverse.graph.generation import calculate_edge_density, generate_graph


class TestGraphGeneration(unittest.TestCase):
    def test_generate_graph(self):
        n = 100
        c = 1.1
        graph = generate_graph(n, c)

        self.assertEqual(graph.number_of_nodes(), n)
        self.assertTrue(nx.is_connected(graph))

        expected_density = c * nx.Graph.average_degree(graph) / (2 * (n - 1))
        actual_density = calculate_edge_density(graph)
        self.assertAlmostEqual(actual_density, expected_density, places=2)

    def test_calculate_edge_density(self):
        G = nx.complete_graph(5)
        density = calculate_edge_density(G)
        self.assertEqual(density, 1.0)


if __name__ == "__main__":
    unittest.main()
