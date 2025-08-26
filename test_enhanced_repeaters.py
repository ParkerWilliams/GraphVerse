#!/usr/bin/env python3
"""
Test the enhanced repeater system with multiple k-cycles and near-uniform edge weights.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import AscenderRule, EvenRule, RepeaterRule
from graphverse.graph.graph_generation import generate_random_graph
from graphverse.graph.walk import generate_valid_walk
import json
import numpy as np

def test_enhanced_repeaters():
    """Test the enhanced repeater system."""
    
    print("🧪 TESTING ENHANCED REPEATER SYSTEM")
    print("=" * 60)
    
    # Create rules for a small test graph
    n = 100
    ascender_nodes = list(range(0, 10))  # 10% ascenders
    even_nodes = list(range(10, 25))     # 15% even nodes
    repeater_nodes_dict = {i: 3 for i in range(25, 35)}  # 10% repeaters with k=3
    
    rules = [
        AscenderRule(ascender_nodes),
        EvenRule(even_nodes),
        RepeaterRule(repeater_nodes_dict)
    ]
    
    print(f"Test graph: {n} nodes")
    print(f"Repeater nodes: {list(repeater_nodes_dict.keys())} (k=3)")
    
    # Generate graph with enhanced repeater system
    G = generate_random_graph(
        n=n,
        rules=rules,
        num_walks=50,
        min_walk_length=8,
        max_walk_length=15,
        verbose=True,
        min_edge_density=0.5
    )
    
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE K-CYCLES PER REPEATER")
    print("=" * 60)
    
    # Test multiple k-cycles per repeater
    repeater_nodes = list(repeater_nodes_dict.keys())
    test_repeater = repeater_nodes[0]
    
    cycles = G.get_all_repeater_cycles(test_repeater)
    print(f"\nRepeater {test_repeater} has {len(cycles) if cycles else 0} k-cycles:")
    
    if cycles:
        for i, cycle in enumerate(cycles):
            print(f"  Cycle {i+1}: {cycle}")
            # Verify cycle validity
            if len(cycle) >= 3 and cycle[0] == test_repeater and cycle[-1] == test_repeater:
                print(f"    ✅ Valid k-cycle (length: {len(cycle)-1})")
            else:
                print(f"    ❌ Invalid k-cycle")
    
    print("\n" + "=" * 60) 
    print("TESTING EXPONENTIAL EDGE WEIGHTS")
    print("=" * 60)
    
    # Test exponential edge weights
    sample_nodes = list(range(5))  # Test first 5 nodes
    exponential_scores = []
    
    for node in sample_nodes:
        neighbors, probs = G.get_edge_probabilities(node)
        if len(neighbors) > 1:
            # Test if probabilities follow exponential-like distribution
            sorted_probs = sorted(probs, reverse=True)
            
            # Calculate ratio of largest to smallest probability
            prob_ratio = max(probs) / min(probs) if min(probs) > 0 else float('inf')
            
            # Calculate coefficient of variation (std/mean) - higher for exponential
            prob_mean = np.mean(probs)
            prob_std = np.std(probs)
            cv = prob_std / prob_mean if prob_mean > 0 else 0
            
            exponential_scores.append(cv)
            
            print(f"\nNode {node} edge weights:")
            print(f"  Neighbors: {len(neighbors)}")
            print(f"  Probabilities: {[f'{p:.3f}' for p in sorted_probs[:8]]}{'...' if len(sorted_probs) > 8 else ''}")
            print(f"  Max/Min ratio: {prob_ratio:.2f}")
            print(f"  Coefficient of variation: {cv:.3f}")
    
    if exponential_scores:
        avg_cv = np.mean(exponential_scores)
        print(f"\nAverage coefficient of variation: {avg_cv:.3f}")
        print(f"Distribution assessment: {'✅ Exponential-like' if avg_cv > 0.5 else '⚠️ Some variation' if avg_cv > 0.2 else '❌ Too uniform'}")
    
    print("\n" + "=" * 60)
    print("TESTING WALK BEHAVIOR")
    print("=" * 60)
    
    # Test walk behavior: starting from repeater vs random start
    num_test_walks = 20
    
    # Walks starting from repeater
    repeater_kcycle_follows = 0
    for _ in range(num_test_walks):
        walk = generate_valid_walk(G, test_repeater, 8, 15, rules, verbose=False)
        if walk:
            # Check if walk follows a k-cycle
            k_cycle = G.get_repeater_cycle(test_repeater)  # Get random k-cycle
            if k_cycle and len(k_cycle) > 2 and len(walk) >= len(k_cycle)-1:
                expected_segment = k_cycle[1:-1]  # Remove start/end repeater
                actual_segment = walk[1:1+len(expected_segment)]
                if actual_segment == expected_segment:
                    repeater_kcycle_follows += 1
    
    repeater_follow_rate = repeater_kcycle_follows / num_test_walks * 100
    
    # Walks starting from random nodes
    random_repeater_encounters = 0
    for _ in range(num_test_walks):
        random_start = np.random.choice([n for n in range(G.n) if n not in repeater_nodes])
        walk = generate_valid_walk(G, random_start, 8, 15, rules, verbose=False)
        if walk and test_repeater in walk:
            random_repeater_encounters += 1
    
    random_encounter_rate = random_repeater_encounters / num_test_walks * 100
    
    print(f"\nWalk behavior analysis:")
    print(f"  Repeater-started k-cycle following: {repeater_kcycle_follows}/{num_test_walks} ({repeater_follow_rate:.1f}%)")
    print(f"  Random-started repeater encounters: {random_repeater_encounters}/{num_test_walks} ({random_encounter_rate:.1f}%)")
    
    print("\n" + "=" * 60)
    print("SYSTEM STATUS")
    print("=" * 60)
    
    # Overall assessment
    issues = []
    
    if not cycles or len(cycles) < 2:
        issues.append("❌ Repeater doesn't have multiple k-cycles")
    
    if not exponential_scores or avg_cv < 0.2:
        issues.append("❌ Edge weights are not exponentially distributed")
    
    if repeater_follow_rate < 50:
        issues.append("⚠️ Low k-cycle following rate from repeater starts")
        
    if random_encounter_rate > 30:
        issues.append("⚠️ High random repeater encounters (weights may not be uniform enough)")
    
    if not issues:
        print("🎉 ENHANCED REPEATER SYSTEM WORKING CORRECTLY!")
        print("✅ Multiple k-cycles per repeater implemented")
        print("✅ Exponential edge weight distribution implemented")  
        print("✅ Expected walk behavior observed")
    else:
        print("⚠️ SYSTEM ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = test_enhanced_repeaters()
    sys.exit(0 if success else 1)