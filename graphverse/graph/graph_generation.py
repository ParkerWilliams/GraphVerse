import random
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .walk import generate_multiple_walks
from .rules import RepeaterRule, AscenderRule, EvenRule
from .base import Graph


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

    # Create graph with adjacency matrix
    G = Graph(n)

    # Find the correct rule instances
    repeater_rule = next(rule for rule in rules if isinstance(rule, RepeaterRule))
    ascender_rule = next(rule for rule in rules if isinstance(rule, AscenderRule))
    even_rule = next(rule for rule in rules if isinstance(rule, EvenRule))
    
    # Assign rules to nodes
    for node in tqdm(range(n), desc="Assigning rules to nodes"):
        G.node_attributes[node] = {"rule": "none"}
        
        if node in ascender_rule.member_nodes:
            G.node_attributes[node]["rule"] = "ascender"
        elif node in even_rule.member_nodes:
            G.node_attributes[node]["rule"] = "even"
        elif node in repeater_rule.member_nodes:
            G.node_attributes[node]["rule"] = "repeater"
            G.node_attributes[node]["repetitions"] = repeater_rule.members_nodes_dict[node]

    # Add edges based on rules
    for node in tqdm(range(n), desc="Adding edges to graph"):
        rule = G.node_attributes[node]["rule"]
        
        if rule == "ascender":
            candidates = [v for v in range(n) if v > node]
            if candidates:
                num_edges = random.randint(1, len(candidates))
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == "even":
            candidates = [v for v in range(n) if v % 2 == 0]
            if candidates:
                num_edges = random.randint(1, len(candidates))
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == "repeater":
            repetitions = G.node_attributes[node]["repetitions"]
            candidates = [
                v for v in range(n)
                if v != node
                and G.node_attributes[v]["rule"] != "even"
            ]
            if candidates:
                for _ in range(repetitions):
                    v = random.choice(candidates)
                    G.add_edge(node, v)

    # Assign random edge weights (probabilities)
    for node in tqdm(range(n), desc="Assigning edge probabilities"):
        neighbors = G.get_neighbors(node)
        if len(neighbors) > 0:
            weights = np.random.random(len(neighbors))
            weights = weights / np.sum(weights)
            for neighbor, weight in zip(neighbors, weights):
                G.adjacency[node, neighbor] = weight

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

    return G


def calculate_edge_density(graph, verbose=False):
    if verbose:
        print("Calculating edge density...")
    num_nodes = graph.n
    num_edges = np.sum(graph.adjacency > 0)
    max_possible_edges = num_nodes * (num_nodes - 1)
    edge_density = num_edges / max_possible_edges
    if verbose:
        print(f"Edge density: {edge_density}")
    return edge_density
