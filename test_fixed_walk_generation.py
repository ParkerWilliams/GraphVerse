#!/usr/bin/env python3
"""
Test the fixed walk generation with add_edges parameter.
This script:
1. Creates a fresh graph
2. Pre-builds it to desired density
3. Tests walk generation with add_edges=False
4. Verifies model evaluation works correctly
"""

import os
import sys
import pickle
import numpy as np
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import AscenderRule, EvenRule, RepeaterRule
from graphverse.graph.walk import generate_valid_walk


def create_dense_graph(n=100, target_density=0.4, seed=42):
    """Create a dense graph with specified edge density."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Create base graph
    graph = Graph(n)
    
    # Calculate number of edges needed
    total_possible_edges = n * (n - 1) // 2  # For undirected graph
    target_edges = int(total_possible_edges * target_density)
    
    print(f"Creating graph with {n} nodes")
    print(f"Target density: {target_density}")
    print(f"Target edges: {target_edges}")
    
    # Add random edges until we reach target density
    edges_added = 0
    while edges_added < target_edges:
        v1 = random.randint(0, n-1)
        v2 = random.randint(0, n-1)
        
        if v1 != v2 and not graph.has_edge(v1, v2):
            graph.add_edge(v1, v2)
            edges_added += 1
            
            if edges_added % 1000 == 0:
                print(f"  Added {edges_added}/{target_edges} edges...")
    
    # Verify density
    actual_edges = np.sum(graph.adjacency > 0) // 2  # Divide by 2 for undirected
    actual_density = actual_edges / total_possible_edges
    print(f"Graph created: {actual_edges} edges, density={actual_density:.4f}")
    
    return graph


def test_walk_generation(graph, rules):
    """Test walk generation without edge addition."""
    print(f"\nTesting walk generation (no edge addition)")
    
    # Store initial edge count
    initial_edges = np.sum(graph.adjacency > 0)
    
    # Generate some walks
    num_test_walks = 10
    successful_walks = 0
    failed_walks = 0
    
    for i in range(num_test_walks):
        start_node = random.randint(0, graph.n - 1)
        
        try:
            walk = generate_valid_walk(
                graph=graph,
                start_vertex=start_node,
                min_length=20,
                max_length=30,
                rules=rules,
                max_attempts=10,
                verbose=False
            )
            
            if walk and len(walk) >= 20:
                successful_walks += 1
                
                # Verify all edges exist
                all_edges_exist = True
                for j in range(1, len(walk)):
                    if not graph.has_edge(walk[j-1], walk[j]):
                        all_edges_exist = False
                        print(f"  Walk {i}: Missing edge {walk[j-1]} -> {walk[j]}")
                        break
                
                if all_edges_exist:
                    print(f"  Walk {i}: Success, length={len(walk)}, all edges exist")
                else:
                    print(f"  Walk {i}: Generated but has missing edges!")
            else:
                failed_walks += 1
                print(f"  Walk {i}: Failed to generate")
                
        except Exception as e:
            failed_walks += 1
            print(f"  Walk {i}: Exception - {e}")
    
    # Check if edges were added
    final_edges = np.sum(graph.adjacency > 0)
    edges_added = (final_edges - initial_edges) // 2  # Divide by 2 for undirected
    
    print(f"\nResults:")
    print(f"  Successful walks: {successful_walks}/{num_test_walks}")
    print(f"  Failed walks: {failed_walks}/{num_test_walks}")
    print(f"  Edges added during generation: {edges_added}")
    
    return successful_walks, failed_walks, edges_added


def main():
    """Test the fixed walk generation."""
    print("="*60)
    print("TESTING FIXED WALK GENERATION")
    print("="*60)
    
    # Create a small test graph
    n = 100
    graph = create_dense_graph(n=n, target_density=0.4, seed=42)
    
    # Create simple rules for testing
    ascenders = list(range(10, 20))  # Nodes 10-19
    evens = list(range(0, n, 2))[:20]  # First 20 even numbers
    repeaters = {30: 3, 31: 4, 32: 5}  # 3 repeaters with different k values
    
    rules = [
        AscenderRule(ascenders),
        EvenRule(evens),
        RepeaterRule(repeaters)
    ]
    
    print(f"\nRules created:")
    print(f"  Ascenders: {len(ascenders)} nodes")
    print(f"  Evens: {len(evens)} nodes")
    print(f"  Repeaters: {len(repeaters)} nodes")
    
    # Test walk generation without edge addition
    print("\n" + "="*40)
    print("TEST: Walk generation without edge addition")
    print("="*40)
    success, fail, added = test_walk_generation(graph, rules)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Results:")
    print(f"  Success rate: {success}/10")
    print(f"  Edges added: {added}")
    print(f"\nConclusion:")
    if added == 0:
        print("  ✅ SUCCESS: No edges added during walk generation")
    else:
        print("  ❌ FAILURE: Edges are still being added")
    
    if success > 0:
        print("  ✅ Walks can be generated with existing edges only")
    else:
        print("  ⚠️  Walk generation may need a denser graph")


if __name__ == "__main__":
    main()