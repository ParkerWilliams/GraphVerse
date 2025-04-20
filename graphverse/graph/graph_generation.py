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
    verbose=False, save_walks=False, output_dir="walks",
    min_edge_density=0.1  # Add minimum edge density parameter
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

    # First handle repeater nodes to ensure valid paths exist
    for node in tqdm(range(n), desc="Setting up repeater paths"):
        if G.node_attributes[node]["rule"] == "repeater":
            repetitions = G.node_attributes[node]["repetitions"]
            
            # Find valid intermediate nodes (not even-rule nodes)
            valid_nodes = [
                v for v in range(n)
                if v != node
                and G.node_attributes[v]["rule"] != "even"
                and v not in repeater_rule.member_nodes  # Avoid other repeaters
            ]
            
            if len(valid_nodes) < repetitions:
                raise ValueError(f"Not enough valid nodes for repeater {node} with {repetitions} repetitions")
            
            # Select nodes for the path
            path_nodes = random.sample(valid_nodes, k=repetitions)
            
            # Add edges to create the path
            for path_node in path_nodes:
                G.add_edge(node, path_node)
                # Add return edge to allow getting back to repeater
                G.add_edge(path_node, node)

    # Now add edges for other rules
    for node in tqdm(range(n), desc="Adding remaining edges"):
        rule = G.node_attributes[node]["rule"]
        
        if rule == "ascender":
            candidates = [v for v in range(n) if v > node]
            if candidates:
                num_edges = random.randint(1, min(len(candidates), 5))  # Limit max edges
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == "even":
            candidates = [v for v in range(n) if v % 2 == 0]
            if candidates:
                num_edges = random.randint(1, min(len(candidates), 5))  # Limit max edges
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)

    # Ensure minimum edge density is met
    current_density = calculate_edge_density(G, verbose=False)
    if current_density < min_edge_density:
        if verbose:
            print(f"\nCurrent edge density {current_density:.3f} below minimum {min_edge_density}")
            print("Adding additional edges to meet minimum density...")
        
        # Calculate how many edges we need to add
        num_nodes = G.n
        max_possible_edges = num_nodes * (num_nodes - 1)
        target_edges = int(min_edge_density * max_possible_edges)
        current_edges = np.sum(G.adjacency > 0)
        edges_to_add = target_edges - current_edges
        
        # Add edges between non-rule nodes first
        non_rule_nodes = [n for n in range(n) if G.node_attributes[n]['rule'] == 'none']
        edges_added = 0
        
        # Try to add edges while respecting rules
        attempts = 0
        max_attempts = edges_to_add * 2  # Allow some failed attempts
        
        with tqdm(total=edges_to_add, desc="Adding density edges") as pbar:
            while edges_added < edges_to_add and attempts < max_attempts:
                # Prefer non-rule nodes but occasionally use rule nodes
                if random.random() < 0.8 and non_rule_nodes:
                    source = random.choice(non_rule_nodes)
                else:
                    source = random.randint(0, n-1)
                
                # For rule nodes, respect their constraints
                rule = G.node_attributes[source]['rule']
                if rule == 'ascender':
                    candidates = [v for v in range(n) if v > source]
                elif rule == 'even':
                    candidates = [v for v in range(n) if v % 2 == 0]
                elif rule == 'repeater':
                    # Skip repeater nodes as their paths are already set
                    attempts += 1
                    continue
                else:
                    candidates = list(range(n))
                    candidates.remove(source)
                
                # Remove existing neighbors
                existing_neighbors = set(np.where(G.adjacency[source] > 0)[0])
                candidates = [v for v in candidates if v not in existing_neighbors]
                
                if candidates:
                    target = random.choice(candidates)
                    G.add_edge(source, target)
                    edges_added += 1
                    pbar.update(1)
                
                attempts += 1
        
        if verbose:
            final_density = calculate_edge_density(G, verbose=False)
            print(f"Final edge density: {final_density:.3f}")
            if final_density < min_edge_density:
                print("Warning: Could not reach target density while respecting rules")

    # Assign random edge weights (probabilities)
    for node in tqdm(range(n), desc="Assigning edge probabilities"):
        neighbors = G.get_neighbors(node)
        if len(neighbors) > 0:
            weights = np.random.random(len(neighbors))
            weights = weights / np.sum(weights)
            for neighbor, weight in zip(neighbors, weights):
                G.adjacency[node, neighbor] = weight

    # Calculate and report edge density statistics
    if verbose:
        print("\nEdge Density Statistics:")
        # Overall edge density
        overall_density = calculate_edge_density(G, verbose=False)
        print(f"Overall edge density: {overall_density:.3f}")
        
        # Edge density by rule type
        rule_types = ['ascender', 'even', 'repeater', 'none']
        for rule_type in rule_types:
            # Get nodes of this type
            type_nodes = [n for n in range(G.n) if G.node_attributes[n]['rule'] == rule_type]
            if not type_nodes:
                continue
                
            # Count edges from these nodes
            num_edges = sum(np.sum(G.adjacency[n] > 0) for n in type_nodes)
            max_possible = len(type_nodes) * (G.n - 1)  # Each node could connect to all others
            density = num_edges / max_possible if max_possible > 0 else 0
            
            # Count edges between nodes of this type
            internal_edges = sum(
                np.sum(G.adjacency[n1, n2] > 0)
                for i, n1 in enumerate(type_nodes)
                for n2 in type_nodes[i+1:]
            )
            max_internal = (len(type_nodes) * (len(type_nodes) - 1)) // 2
            internal_density = internal_edges / max_internal if max_internal > 0 else 0
            
            print(f"\n{rule_type.capitalize()} nodes:")
            print(f"  Count: {len(type_nodes)}")
            print(f"  Outgoing edge density: {density:.3f}")
            print(f"  Internal edge density: {internal_density:.3f}")

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
