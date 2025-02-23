import random

import networkx as nx

from .walk import generate_multiple_walks


def generate_random_graph(
    n, rules, verbose=False
):
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
    
    ascender_rule, descender_rule, even_rule, odd_rule, repeater_rule = rules
    
    # Debugging: Print the initial state of repeater_rule.member_nodes
    if verbose:
        print(f"Initial repeater_rule.member_nodes: {repeater_rule.member_nodes}")
    
    for node in G.nodes():
        if verbose:
            print(f"node {node} being inspected")
        # Start with no rule
        G.nodes[node]["rule"] = "none"
        
        # Check and assign the correct rule
        if node in ascender_rule.member_nodes:
            G.nodes[node]["rule"] = "ascender"
        elif node in descender_rule.member_nodes:
            G.nodes[node]["rule"] = "descender"
        elif node in even_rule.member_nodes:
            G.nodes[node]["rule"] = "even"
        elif node in odd_rule.member_nodes:
            G.nodes[node]["rule"] = "odd"
        elif node in repeater_rule.member_nodes:
            if verbose:
                print(f"node {node} is a repeater, the repeater set is {repeater_rule.member_nodes}")
            G.nodes[node]["rule"] = "repeater"
            G.nodes[node]["repetitions"] = repeater_rule.members_nodes_dict[node]
    
    # Debugging: Print the final state of repeater_rule.member_nodes
    if verbose:
        print(f"Final repeater_rule.member_nodes: {repeater_rule.member_nodes}")

    # Build the graph by adding edges that satisfy the rules
    if verbose:
        print("Building the graph by adding edges that satisfy the rules...")
    for node in G.nodes():
        if verbose:
            print(f"Processing node {node}...")
        rule = G.nodes[node]["rule"]
        if rule == "ascender":
            # Add edges to higher-numbered nodes
            candidates = [v for v in G.nodes() if v > node]
            if candidates:
                num_edges = random.randint(1, len(candidates))
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == "descender":
            # Add edges to lower-numbered nodes
            candidates = [v for v in G.nodes() if v < node]
            if candidates:
                num_edges = random.randint(1, len(candidates))
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == "even":
            # Add edges to even-numbered nodes
            candidates = [v for v in G.nodes() if v % 2 == 0]
            if candidates:
                num_edges = random.randint(1, len(candidates))
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == "odd":
            # Add edges to odd-numbered nodes
            candidates = [v for v in G.nodes() if v % 2 != 0]
            if candidates:
                num_edges = random.randint(1, len(candidates))
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == "repeater":
            # Add edges to satisfy the repeater rule
            repetitions = G.nodes[node]["repetitions"]
            candidates = [
                v
                for v in G.nodes()
                if v != node
                and G.nodes[v]["rule"] != "even"
                and G.nodes[v]["rule"] != "odd"
            ]
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
                G[u][v]["probability"] = prob

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


def save_graph(G, path="my_graph.gml", verbose=False):
    """
    Save the graph to disk.
    """
    if verbose:
        print(f"Saving graph to {path}...")
    nx.write_gml(G, path)
    if verbose:
        print("Graph saved successfully.")
    return True


def load_graph(path="my_graph.gml", verbose=False):
    """
    Load the Graph from disk.
    """
    if verbose:
        print(f"Loading graph from {path}...")
    G = nx.read_gml(path)
    if verbose:
        print("Graph loaded successfully.")
    return G
