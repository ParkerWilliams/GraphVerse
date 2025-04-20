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
    
    # Track which vertex types cause failures
    failure_points = []
    
    while attempts < max_attempts:
        # Start a new walk
        walk = [start_vertex]
        
        # Try to extend the walk to target length
        while len(walk) < target_length:
            current_node = walk[-1]
            neighbors, probs = graph.get_edge_probabilities(current_node)
            
            if len(neighbors) == 0:
                # Dead end - record the failing vertex type
                failure_points.append({
                    'node': current_node,
                    'type': graph.node_attributes[current_node]['rule'],
                    'reason': 'no_neighbors'
                })
                break
            
            # Sample next node based on probabilities
            next_vertex = np.random.choice(neighbors, p=probs)
            walk.append(next_vertex)
            
            # Check if the new walk segment violates any rules
            if not check_rule_compliance(graph, walk, rules, verbose):
                # Record which vertex caused the rule violation
                failure_points.append({
                    'node': next_vertex,
                    'type': graph.node_attributes[next_vertex]['rule'],
                    'reason': 'rule_violation'
                })
                walk.pop()
                break
        
        # Check if we got a valid walk
        if len(walk) >= min_length and check_rule_compliance(graph, walk, rules, verbose):
            return walk
            
        attempts += 1
    
    if verbose:
        print(f"Failed to generate valid walk after {max_attempts} attempts")
        if failure_points:
            # Count failures by type
            failure_counts = {}
            for failure in failure_points:
                key = (failure['type'], failure['reason'])
                failure_counts[key] = failure_counts.get(key, 0) + 1
            
            print("\nFailure analysis:")
            for (vertex_type, reason), count in failure_counts.items():
                print(f"  {vertex_type} vertices: {count} failures due to {reason}")
            
            # Show some example failing nodes
            print("\nExample failing nodes:")
            for failure in failure_points[-3:]:  # Show last 3 failures
                print(f"  Node {failure['node']} ({failure['type']}): {failure['reason']}")
    
    return None


def generate_multiple_walks(
    graph, num_walks, min_length, max_length, rules, verbose=False
):
    """Generate multiple valid walks on the graph."""
    walks = []
    total_attempts = 0
    failure_stats = {
        'ascender': {'attempts': 0, 'successes': 0},
        'even': {'attempts': 0, 'successes': 0},
        'repeater': {'attempts': 0, 'successes': 0},
        'none': {'attempts': 0, 'successes': 0}
    }

    pbar = tqdm(total=num_walks, desc="Generating walks")
    current_walks = 0

    while len(walks) < num_walks:
        start_vertex = random.randint(0, graph.n - 1)
        vertex_type = graph.node_attributes[start_vertex]['rule']
        failure_stats[vertex_type]['attempts'] += 1
        
        walk = generate_valid_walk(
            graph, start_vertex, min_length, max_length, rules, max_attempts=10, verbose=verbose
        )

        if walk:
            walks.append(walk)
            failure_stats[vertex_type]['successes'] += 1
            new_walks = len(walks) - current_walks
            pbar.update(new_walks)
            current_walks = len(walks)
        
        total_attempts += 1

    pbar.close()

    if verbose:
        print("\nWalk Generation Statistics:")
        print(f"Total attempts needed: {total_attempts}")
        print("\nSuccess rate by vertex type:")
        for vertex_type, stats in failure_stats.items():
            if stats['attempts'] > 0:
                success_rate = (stats['successes'] / stats['attempts']) * 100
                print(f"  {vertex_type}: {success_rate:.1f}% success rate "
                      f"({stats['successes']}/{stats['attempts']} attempts)")

    return walks
