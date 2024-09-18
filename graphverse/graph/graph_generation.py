import networkx as nx
import math
import random


def generate_random_graph(n, rules, num_walks, min_walk_length, max_walk_length):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(range(n))

    # Generate walks and add edges based on the walks
    walks = generate_multiple_walks(G, num_walks, min_walk_length, max_walk_length, rules)

    # Assign random probability distributions to outgoing edges
    for node in G.nodes():
        out_edges = list(G.out_edges(node))
        if out_edges:
            probabilities = [random.random() for _ in range(len(out_edges))]
            total = sum(probabilities)
            normalized_probabilities = [p / total for p in probabilities]
            for (u, v), prob in zip(out_edges, normalized_probabilities):
                G[u][v]['probability'] = prob

    return G


def calculate_edge_density(G):
    """
    Calculate the actual edge density of the graph.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    return 2 * m / (n * (n - 1))


def save_graph(G, path='my_graph.gml'):
    """
    Save the graph to disk.
    """
    nx.write_gml(G, path)
    return True


def load_graph(path='my_graph.gml'):
    """
    Load the Graph from disk.
    """
    G = nx.read_gml(path)
    return G
