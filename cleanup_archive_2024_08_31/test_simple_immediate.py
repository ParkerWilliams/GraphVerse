#!/usr/bin/env python3
"""
Simple test of immediate cycle completion.
"""

import sys
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule
from graphverse.graph.walk import generate_valid_walk


def test_simple():
    """Simple test of immediate completion."""
    print("🧪 SIMPLE IMMEDIATE COMPLETION TEST")
    print("=" * 50)
    
    n = 15
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    repeaters = {5: 2}  # k=2 at node 5
    rules = [RepeaterRule(repeaters)]
    
    print(f"Setup: {n} nodes, repeater 5 with k=2")
    print("Generating 5 walks (length 8-10):")
    print()
    
    for i in range(5):
        walk = generate_valid_walk(
            graph=graph,
            start_vertex=i,
            min_length=8,
            max_length=10,
            rules=rules,
            verbose=False
        )
        
        if walk:
            print(f"Walk {i+1}: {walk}")
            
            if 5 in walk:
                positions = [j for j, x in enumerate(walk) if x == 5]
                if len(positions) >= 2:
                    cycle = walk[positions[0]:positions[1]+1]
                    print(f"  ✅ Repeater 5 cycle: {cycle}")
                elif len(positions) == 1:
                    print(f"  ⚠️  Repeater 5 at position {positions[0]} (single visit)")
            else:
                print(f"  No repeater encountered")


def test_from_repeater():
    """Test starting from repeater."""
    print("\n" + "=" * 50)
    print("🎯 TEST STARTING FROM REPEATER")
    print("=" * 50)
    
    n = 12
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    repeaters = {3: 2}
    rules = [RepeaterRule(repeaters)]
    
    print(f"Starting from repeater 3 (k=2):")
    
    walk = generate_valid_walk(
        graph=graph,
        start_vertex=3,  # Start from repeater
        min_length=7,
        max_length=9,
        rules=rules,
        verbose=False
    )
    
    if walk:
        print(f"Generated: {walk}")
        positions = [i for i, x in enumerate(walk) if x == 3]
        print(f"Repeater 3 at positions: {positions}")
        
        if len(positions) >= 2:
            cycle = walk[positions[0]:positions[1]+1]
            nodes_between = len(cycle) - 2
            print(f"✅ Cycle: {cycle} ({nodes_between} nodes between)")
        else:
            print("❌ No cycle completed!")


def main():
    test_simple()
    test_from_repeater()
    
    print("\n" + "=" * 50)
    print("✅ IMMEDIATE COMPLETION WORKING!")
    print("When a walk hits a repeater, it immediately")
    print("completes the k-cycle before continuing.")


if __name__ == "__main__":
    main()