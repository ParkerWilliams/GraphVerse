import random
import json
from pathlib import Path
from tqdm import tqdm
import networkx as nx

from .walk import generate_multiple_walks
from .rules import RepeaterRule, AscenderRule, DescenderRule, EvenRule, OddRule


def save_walks_to_files(walks, output_dir, max_file_size=50*1024*1024, verbose=False):
    """
    Save walks to multiple files, ensuring each file is under max_file_size (in bytes).
    
    Args:
        walks: List of walks to save
        output_dir: Directory to save the files
        max_file_size: Maximum file size in bytes (default 50MB)
        verbose: Whether to print progress
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    walks_per_file = len(walks) // 10  # Start with rough estimate of 10 files
    current_file = 0
    start_idx = 0
    
    while start_idx < len(walks):
        # Try to write a chunk of walks
        test_walks = walks[start_idx:start_idx + walks_per_file]
        test_json = json.dumps(test_walks)
        
        # If file would be too large, reduce walks_per_file
        if len(test_json.encode('utf-8')) > max_file_size:
            walks_per_file = walks_per_file // 2
            continue
            
        # Save the walks
        output_file = Path(output_dir) / f"walks_{current_file}.json"
        with open(output_file, 'w') as f:
            json.dump(test_walks, f)
            
        if verbose:
            print(f"Saved {len(test_walks)} walks to {output_file}")
            
        start_idx += walks_per_file
        current_file += 1


def generate_random_graph(
    n, rules, num_walks=1000, min_walk_length=5, max_walk_length=20, 
    verbose=False, save_walks=False, output_dir="walks"
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

    # Find the correct rule instances
    repeater_rule = next(rule for rule in rules if isinstance(rule, RepeaterRule))
    ascender_rule = next(rule for rule in rules if isinstance(rule, AscenderRule))
    descender_rule = next(rule for rule in rules if isinstance(rule, DescenderRule))
    even_rule = next(rule for rule in rules if isinstance(rule, EvenRule))
    odd_rule = next(rule for rule in rules if isinstance(rule, OddRule))
    
    if verbose:
        print(f"Rule types after finding:")
        print(f"repeater_rule: {type(repeater_rule)}")
        print(f"odd_rule: {type(odd_rule)}")
        print(f"ascender_rule: {type(ascender_rule)}")
        print(f"even_rule: {type(even_rule)}")
        print(f"descender_rule: {type(descender_rule)}")
    
    # Use tqdm for the node processing loop
    for node in tqdm(G.nodes(), desc="Processing nodes", disable=not verbose):
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
                print(f"\nnode {node} is a repeater")
            G.nodes[node]["rule"] = "repeater"
            G.nodes[node]["repetitions"] = repeater_rule.members_nodes_dict[node]

    # Build the graph by adding edges that satisfy the rules
    if verbose:
        print("\nBuilding the graph by adding edges that satisfy the rules...")
    for node in tqdm(G.nodes(), desc="Adding edges", disable=not verbose):
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

    # Generate and save walks if requested
    if save_walks:
        if verbose:
            print("\nGenerating walks...")
        walks = generate_multiple_walks(
            G, num_walks, min_walk_length, max_walk_length, 
            rules, verbose=verbose
        )
        if verbose:
            print("\nSaving walks...")
        save_walks_to_files(walks, output_dir, verbose=verbose)

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
