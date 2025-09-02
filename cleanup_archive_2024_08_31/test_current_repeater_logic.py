#!/usr/bin/env python3
"""
Test walks with repeaters using the current RepeaterRule implementation
to verify the logic is working as coded (even if it has the off-by-one issue).
"""

import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
from graphverse.graph.walk import generate_multiple_walks


def create_test_graph_with_repeaters():
    """Create a small test graph with known repeater nodes."""
    n = 20
    adjacency = np.zeros((n, n))
    
    # Create a connected graph with edges
    for i in range(n):
        for j in range(n):
            if i != j:  # No self-loops
                adjacency[i][j] = np.random.random() * 0.3  # Sparse connections
    
    # Define some repeaters with different k values
    repeaters = {
        5: 2,   # k=2
        10: 3,  # k=3  
        15: 4   # k=4
    }
    
    # Create attributes
    node_attributes = {}
    for i in range(n):
        if i in repeaters:
            node_attributes[str(i)] = {
                'rule': 'repeater',
                'repetitions': repeaters[i]
            }
        else:
            node_attributes[str(i)] = {'rule': 'none'}
    
    attrs_data = {
        'node_attributes': node_attributes,
        'graph_type': 'test',
        'n_nodes': n
    }
    
    return adjacency, attrs_data, repeaters


def test_current_repeater_implementation():
    """Generate walks and test them against the current RepeaterRule implementation."""
    print("🧪 TESTING CURRENT REPEATER RULE IMPLEMENTATION")
    print("=" * 70)
    
    # Create test graph
    adjacency, attrs_data, repeaters_dict = create_test_graph_with_repeaters()
    node_attributes = attrs_data['node_attributes']
    
    print("🔄 Test repeaters defined:")
    for node, k in repeaters_dict.items():
        print(f"  Node {node}: k={k}")
    
    # Create graph object
    graph = Graph(adjacency.shape[0])
    graph.adjacency_matrix = adjacency
    
    # Create rule objects
    repeater_rule = RepeaterRule(repeaters_dict)
    rules = [repeater_rule]
    
    print(f"\n📊 Generating walks with current RepeaterRule implementation...")
    
    # Generate walks that should follow the current (buggy) rule
    walks = generate_multiple_walks(
        graph=graph,
        num_walks=100,
        min_length=10,
        max_length=20,
        rules=rules,
        verbose=False
    )
    
    print(f"✓ Generated {len(walks)} walks")
    
    # Find walks that contain repeaters
    repeater_walks = []
    for walk in walks:
        has_repeater = False
        repeater_visits = defaultdict(int)
        
        for node in walk:
            if node in repeaters_dict:
                repeater_visits[node] += 1
                has_repeater = True
        
        if has_repeater:
            # Check if any repeater appears multiple times
            max_visits = max(repeater_visits.values())
            if max_visits > 1:
                repeater_walks.append((walk, repeater_visits))
    
    print(f"\n🎯 Found {len(repeater_walks)} walks with repeater cycles")
    
    if not repeater_walks:
        print("❌ No walks with repeater cycles found.")
    
    # Always test synthetic examples for verification
    print(f"\n" + "=" * 70)
    print("🧪 SYNTHETIC EXAMPLES VERIFICATION")
    print("=" * 70)
    generate_synthetic_examples(repeaters_dict)
    
    # Analyze the walks
    analyzed = 0
    valid_count = 0
    
    for walk, repeater_visits in repeater_walks[:10]:  # Analyze first 10
        print(f"\n" + "🔍 " + "=" * 60)
        print(f"WALK #{analyzed + 1}")
        print(f"Walk: {walk}")
        print(f"Length: {len(walk)}")
        print(f"Repeater visits: {dict(repeater_visits)}")
        
        # Test against current rule implementation
        is_valid_current = repeater_rule.is_satisfied_by(walk, graph)
        
        # Manual analysis of what the current rule expects
        print(f"\n📋 CURRENT RULE ANALYSIS:")
        walk_valid = True
        
        for repeater_node in repeater_visits:
            if repeater_visits[repeater_node] > 1:
                k_value = repeaters_dict[repeater_node]
                positions = [i for i, x in enumerate(walk) if x == repeater_node]
                
                print(f"\n🔄 Repeater {repeater_node} (k={k_value}):")
                print(f"   Positions: {positions}")
                
                # Check what the current implementation expects
                for i in range(len(positions) - 1):
                    pos1 = positions[i]
                    pos2 = positions[i + 1]
                    distance = pos2 - pos1
                    
                    print(f"   Cycle {i+1}: pos {pos1} → pos {pos2}")
                    print(f"   Distance: {distance} (current rule expects: {k_value})")
                    
                    sequence = walk[pos1:pos2+1]
                    nodes_between = len(sequence) - 2  # Actual nodes between
                    print(f"   Sequence: {sequence}")
                    print(f"   Nodes between: {nodes_between}")
                    
                    if distance == k_value:
                        print(f"   ✅ Current rule satisfied (distance = k)")
                    else:
                        print(f"   ❌ Current rule violated (distance ≠ k)")
                        walk_valid = False
        
        print(f"\n🎯 Manual analysis: {'✅ VALID' if walk_valid else '❌ INVALID'}")
        print(f"🎯 Current rule.is_satisfied_by(): {'✅ VALID' if is_valid_current else '❌ INVALID'}")
        
        if walk_valid == is_valid_current:
            print("✅ Manual analysis matches current implementation")
        else:
            print("❌ MISMATCH between manual analysis and current implementation!")
        
        if is_valid_current:
            valid_count += 1
        
        analyzed += 1
        
        if analyzed >= 10:
            break
    
    print(f"\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total walks analyzed: {analyzed}")
    print(f"Valid under current rule: {valid_count}")
    print(f"Invalid under current rule: {analyzed - valid_count}")
    print(f"Success rate: {valid_count/analyzed:.1%}" if analyzed > 0 else "N/A")


def generate_synthetic_examples(repeaters_dict):
    """Generate synthetic examples to test the current rule."""
    print(f"\n🧪 GENERATING SYNTHETIC EXAMPLES FOR CURRENT RULE")
    print(f"=" * 60)
    
    for repeater_node, k_value in repeaters_dict.items():
        print(f"\n🔄 Examples for Node {repeater_node} (k={k_value}):")
        
        # Example that should satisfy current rule (distance = k)
        # This means k-1 nodes between visits
        intermediate_nodes = list(range(0, k_value - 1))  # k-1 nodes
        valid_walk = [repeater_node] + intermediate_nodes + [repeater_node]
        
        print(f"  Current rule compliant: {valid_walk}")
        print(f"  Distance: {len(valid_walk) - 1} (should equal k={k_value})")
        print(f"  Nodes between: {len(intermediate_nodes)} (k-1 = {k_value-1})")
        
        # Test it
        graph = Graph(20)
        rule = RepeaterRule(repeaters_dict)
        is_valid = rule.is_satisfied_by(valid_walk, graph)
        print(f"  Current rule validation: {'✅ VALID' if is_valid else '❌ INVALID'}")
        
        # Example that should violate current rule (distance ≠ k)
        wrong_intermediate = list(range(0, k_value))  # k nodes instead of k-1
        invalid_walk = [repeater_node] + wrong_intermediate + [repeater_node]
        
        print(f"  Current rule violating: {invalid_walk}")
        print(f"  Distance: {len(invalid_walk) - 1} (should equal k={k_value})")
        print(f"  Nodes between: {len(wrong_intermediate)} (k = {k_value})")
        
        is_invalid = rule.is_satisfied_by(invalid_walk, graph)
        print(f"  Current rule validation: {'✅ VALID' if is_invalid else '❌ INVALID'}")


def main():
    test_current_repeater_implementation()


if __name__ == "__main__":
    main()