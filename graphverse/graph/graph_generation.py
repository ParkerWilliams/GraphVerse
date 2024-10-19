import networkx as nx
import random
from .walk import generate_multiple_walks


def generate_random_graph(n, rules, num_walks, min_walk_length, max_walk_length, verbose=False):
    if verbose:
        print("Generating random graph...")

    # Create an empty directed graph
    if verbose:
        print("Creating an empty directed graph...")
    G = nx.DiGraph()

    # Add nodes
    if verbose:
        print(f"Adding {n} nodes...")
    G.add_nodes_from(range(n))

    # Assign rule designations to vertices
    if verbose:
        print("Assigning rule designations to vertices...")
    ascenders, descenders, evens, odds, repeaters = rules
    for node in G.nodes():
        if node in ascenders:
            G.nodes[node]['rule'] = 'ascender'
        elif node in descenders:
            G.nodes[node]['rule'] = 'descender'
        elif node in evens:
            G.nodes[node]['rule'] = 'even'
        elif node in odds:
            G.nodes[node]['rule'] = 'odd'
        elif node in repeaters:
            G.nodes[node]['rule'] = 'repeater'
            G.nodes[node]['repetitions'] = repeaters[node]
        else:
            G.nodes[node]['rule'] = 'none'  # Assign 'none' rule for nodes without a specific rule

    # Build the graph by adding edges that satisfy the rules
    if verbose:
        print("Building the graph by adding edges that satisfy the rules...")
    for node in G.nodes():
        if verbose:
            print(f"Processing node {node}...")
        rule = G.nodes[node]['rule']
        if rule == 'ascender':
            # Add edges to higher-numbered nodes
            candidates = [v for v in G.nodes() if v > node]
            if candidates:
                num_edges = random.randint(1, len(candidates))
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == 'descender':
            # Add edges to lower-numbered nodes
            candidates = [v for v in G.nodes() if v < node]
            if candidates:
                num_edges = random.randint(1, len(candidates))
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == 'even':
            # Add edges to even-numbered nodes
            candidates = [v for v in G.nodes() if v % 2 == 0]
            if candidates:
                num_edges = random.randint(1, len(candidates))
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == 'odd':
            # Add edges to odd-numbered nodes
            candidates = [v for v in G.nodes() if v % 2 != 0]
            if candidates:
                num_edges = random.randint(1, len(candidates))
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == 'repeater':
            # Add edges to satisfy the repeater rule
            repetitions = G.nodes[node]['repetitions']
            candidates = [v for v in G.nodes() if v != node and G.nodes[v]['rule'] != 'even' and G.nodes[v]['rule'] != 'odd']
            if candidates:
                for _ in range(repetitions):
                    v = random.choice(candidates)
                    G.add_edge(node, v)

    # Assign random probability distributions to outgoing edges
    if verbose:
        print("Assigning random probability distributions to outgoing edges...")
    for node in G.nodes():
        if verbose:
            print(f"Processing node {node}...")
        out_edges = list(G.out_edges(node))
        if out_edges:
            probabilities = [random.random() for _ in range(len(out_edges))]
            total = sum(probabilities)
            normalized_probabilities = [p / total for p in probabilities]
            for (u, v), prob in zip(out_edges, normalized_probabilities):
                G[u][v]['probability'] = prob

    if verbose:
        print("Random graph generated successfully.")
    return G


def calculate_edge_density(graph, verbose=False):
    if verbose:
        print("Calculating edge density...")
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    max_possible_edges = num_nodes * (num_nodes - 1)
    edge_density = num_edges / max_possible_edges
    if verbose:
        print(f"Edge density: {edge_density}")
    return edge_density


def save_graph(G, path='my_graph.gml', verbose=False):
    """
    Save the graph to disk.
    """
    if verbose:
        print(f"Saving graph to {path}...")
    nx.write_gml(G, path)
    if verbose:
        print("Graph saved successfully.")
    return True


def load_graph(path='my_graph.gml', verbose=False):
    """
    Load the Graph from disk.
    """
    if verbose:
        print(f"Loading graph from {path}...")
    G = nx.read_gml(path)
    if verbose:
        print("Graph loaded successfully.")
    return G
