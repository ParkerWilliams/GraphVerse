#!/usr/bin/env python3
"""
Test that repeater cycles complete correctly without infinite loops.
"""

import sys
from pathlib import Path
import numpy as np

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule
from graphverse.graph.walk import generate_valid_walk, has_incomplete_repeaters
from graphverse.graph.walk_enhanced import generate_valid_walk_enhanced


def test_repeater_completion():
    """Test that repeater cycles must complete."""
    print("🧪 TESTING REPEATER CYCLE COMPLETION")
    print("=" * 70)
    
    # Small test graph
    n = 15
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    # Single repeater rule
    repeaters = {5: 2}  # k=2 repeater at node 5
    rules = [RepeaterRule(repeaters)]
    
    print(f"Test setup: {n} nodes, repeater 5 with k=2")
    print(f"Expected: Repeater 5 must complete cycle with exactly 2 nodes between")
    print()
    
    # Test with original walk generation
    print("Testing with original walk.py:")
    for i in range(3):
        walk = generate_valid_walk(
            graph=graph,
            start_vertex=0,
            min_length=8,
            max_length=10,
            rules=rules,
            verbose=False
        )
        
        if walk:
            analyze_repeater_walk(walk, repeaters)
    
    print("\n" + "-" * 50)
    print("\nTesting with enhanced walk_enhanced.py:")
    for i in range(3):
        walk = generate_valid_walk_enhanced(
            graph=graph,
            start_vertex=0,
            min_length=8,
            max_length=10,
            rules=rules,
            verbose=False
        )
        
        if walk:
            analyze_repeater_walk(walk, repeaters)


def analyze_repeater_walk(walk, repeaters):
    """Analyze a walk for repeater compliance."""
    print(f"Walk: {walk} (len={len(walk)})")
    
    # Check each repeater
    for repeater_node, k_value in repeaters.items():
        if repeater_node in walk:
            positions = [i for i, x in enumerate(walk) if x == repeater_node]
            
            if len(positions) == 0:
                continue
            elif len(positions) == 1:
                print(f"  ❌ Repeater {repeater_node} appears once at pos {positions[0]} - INCOMPLETE!")
            else:
                print(f"  Repeater {repeater_node} at positions: {positions}")
                
                # Check each cycle
                for i in range(len(positions) - 1):
                    pos1, pos2 = positions[i], positions[i + 1]
                    nodes_between = pos2 - pos1 - 1
                    sequence = walk[pos1:pos2+1]
                    
                    if nodes_between == k_value:
                        print(f"    ✅ Cycle {i+1}: {sequence} ({nodes_between} nodes between)")
                    else:
                        print(f"    ❌ Cycle {i+1}: {sequence} ({nodes_between} nodes, need {k_value})")
                
                # Check for immediate re-visits (potential infinite loop)
                if len(positions) > 2:
                    # Check if repeater appears too frequently
                    for i in range(len(positions) - 2):
                        span = positions[i+2] - positions[i]
                        if span < (2 * k_value + 2):  # Two complete cycles minimum span
                            print(f"    ⚠️  WARNING: Repeater visiting too frequently (span={span})")


def test_no_infinite_loops():
    """Test that walks don't get stuck in infinite repeater loops."""
    print("\n" + "=" * 70)
    print("🔍 TESTING NO INFINITE LOOPS")
    print("=" * 70)
    
    n = 10
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    # Multiple repeaters with different k values
    repeaters = {2: 1, 4: 2, 6: 3}
    rules = [RepeaterRule(repeaters)]
    
    print(f"Test: {n} nodes, repeaters at 2(k=1), 4(k=2), 6(k=3)")
    print("Generating 5 walks to check for infinite loop patterns...")
    
    for i in range(5):
        walk = generate_valid_walk(
            graph=graph,
            start_vertex=0,
            min_length=10,
            max_length=15,
            rules=rules,
            verbose=False
        )
        
        if walk:
            # Count repeater visits
            repeater_counts = {}
            for r in repeaters:
                count = walk.count(r)
                if count > 0:
                    repeater_counts[r] = count
            
            # Check if any repeater appears too many times
            excessive = False
            for r, count in repeater_counts.items():
                expected_max = len(walk) // (repeaters[r] + 2)  # Rough upper bound
                if count > expected_max:
                    excessive = True
                    print(f"  ⚠️  Walk {i+1}: Repeater {r} appears {count} times (excessive)")
                    break
            
            if not excessive and repeater_counts:
                print(f"  ✅ Walk {i+1}: Normal repeater usage {repeater_counts}")
            elif not repeater_counts:
                print(f"  ℹ️  Walk {i+1}: No repeaters encountered")


def test_rule_validation():
    """Test that the RepeaterRule correctly validates walks."""
    print("\n" + "=" * 70)
    print("🧾 TESTING REPEATER RULE VALIDATION")
    print("=" * 70)
    
    repeaters = {5: 2}
    rule = RepeaterRule(repeaters)
    
    test_cases = [
        {
            'walk': [1, 2, 5, 3, 4, 5, 6],
            'expected': True,
            'description': 'Complete cycle with k=2'
        },
        {
            'walk': [1, 2, 5, 3, 4],
            'expected': True,  # Single visit is valid according to current rule!
            'description': 'Single visit (current rule says valid)'
        },
        {
            'walk': [1, 2, 5, 3, 5, 6],
            'expected': False,
            'description': 'Incomplete cycle (only 1 node between)'
        },
        {
            'walk': [1, 5, 2, 3, 4, 5, 6, 7, 8, 5],
            'expected': True,
            'description': 'Two complete cycles'
        }
    ]
    
    for case in test_cases:
        walk = case['walk']
        expected = case['expected']
        desc = case['description']
        
        is_valid = rule.is_satisfied_by(walk, None)
        status = "✅" if is_valid == expected else "❌"
        
        print(f"{status} {desc}")
        print(f"   Walk: {walk}")
        print(f"   Valid: {is_valid}, Expected: {expected}")
        
        if 5 in walk:
            positions = [i for i, x in enumerate(walk) if x == 5]
            print(f"   Repeater positions: {positions}")


def main():
    test_repeater_completion()
    test_no_infinite_loops()
    test_rule_validation()
    
    print("\n" + "=" * 70)
    print("🎯 KEY FINDINGS")
    print("=" * 70)
    print("\n⚠️  IMPORTANT ISSUE:")
    print("The RepeaterRule.is_satisfied_by() currently returns True for single visits!")
    print("This is because it only checks consecutive pairs (line 193 in rules.py)")
    print("\nThe rule should be updated to require cycle completion for ANY repeater visit.")


if __name__ == "__main__":
    main()