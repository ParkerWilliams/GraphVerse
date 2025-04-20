import random
import numpy as np
from tqdm import tqdm

from .rules import Rule


def check_rule_compliance(graph, walk, rules, verbose=False):
    """Check if a walk complies with all rules."""
    for rule in rules:
        if not rule.apply(graph, walk):
            return False
    return True


def generate_valid_walk(
    graph, start_vertex, min_length, max_length, rules, max_attempts=10, verbose=False
):
    """Generate a single valid walk starting from start_vertex."""
    # Input validation
    if start_vertex is None or not isinstance(start_vertex, (int, np.integer)):
        if verbose:
            print(f"Invalid start vertex: {start_vertex}")
        return None
        
    if start_vertex < 0 or start_vertex >= graph.n:
        if verbose:
            print(f"Start vertex {start_vertex} out of bounds [0, {graph.n-1}]")
        return None

    target_length = random.randint(min_length, max_length)
    attempts = 0
    
    while attempts < max_attempts:
        # Start a new walk
        walk = [start_vertex]
        
        # Try to extend the walk to target length
        while len(walk) < target_length:
            current_node = walk[-1] 
            neighbors, probs = graph.get_edge_probabilities(current_node)
            
            if len(neighbors) == 0:
                # Dead end - abandon this walk
                break
            
            # Sample next node based on probabilities
            next_vertex = np.random.choice(neighbors, p=probs)
            walk.append(next_vertex)
            
            # Check if the new walk segment violates any rules
            if not check_rule_compliance(graph, walk, rules, verbose):
                walk.pop()
                break
        
        # Check if we got a valid walk
        if len(walk) >= min_length and check_rule_compliance(graph, walk, rules, verbose):
            return walk
            
        attempts += 1
        
    if verbose:
        print(f"Failed to generate valid walk after {max_attempts} attempts")
    return None


def generate_multiple_walks(
    graph, num_walks, min_length, max_length, rules, verbose=False
):
    """Generate multiple valid walks on the graph."""
    walks = []
    attempts = 0
    max_attempts = 10
    total_attempts = 0

    pbar = tqdm(total=num_walks, desc="Generating walks")
    current_walks = 0

    while len(walks) < num_walks:
        start_vertex = random.randint(0, graph.n - 1)
        walk = generate_valid_walk(
            graph, start_vertex, min_length, max_length, rules, max_attempts, verbose
        )

        if walk:
            walks.append(walk)
            attempts = 0
            new_walks = len(walks) - current_walks
            pbar.update(new_walks)
            current_walks = len(walks)
        else:
            attempts += 1
            total_attempts += 1

            if attempts >= max_attempts:
                attempts = 0

    pbar.close()
    return walks
