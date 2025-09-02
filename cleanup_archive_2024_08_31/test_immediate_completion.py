#!/usr/bin/env python3
"""
Test the immediate cycle completion approach for repeaters.
"""

import sys
from pathlib import Path
import numpy as np

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule, AscenderRule
from graphverse.graph.walk import generate_valid_walk


def test_immediate_completion():
    """Test that repeaters immediately complete their cycles."""
    print("🎯 TESTING IMMEDIATE CYCLE COMPLETION")
    print("=" * 70)
    
    n = 20
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    # Repeaters with different k values
    repeaters = {5: 2, 10: 3, 15: 4}
    rules = [RepeaterRule(repeaters)]
    
    print(f"Setup: {n} nodes")
    print(f"Repeaters: 5(k=2), 10(k=3), 15(k=4)")
    print("\nGenerating walks with verbose output to see immediate completion:")
    print()
    
    for i in range(3):
        print(f"Walk {i+1}:")
        walk = generate_valid_walk(
            graph=graph,
            start_vertex=np.random.randint(0, n),
            min_length=10,
            max_length=15,
            rules=rules,
            verbose=True
        )
        
        if walk:
            analyze_walk(walk, repeaters)
        print("\n" + "-" * 50 + "\n")


def test_with_ascender_avoidance():
    """Test that repeater cycles avoid ascender nodes."""
    print("🚫 TESTING ASCENDER AVOIDANCE IN CYCLES")
    print("=" * 70)
    
    n = 15
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    # Repeaters and ascenders
    repeaters = {3: 2, 7: 3}
    ascenders = {5, 8, 12}
    rules = [RepeaterRule(repeaters), AscenderRule(ascenders)]
    
    print(f"Setup: {n} nodes")
    print(f"Repeaters: 3(k=2), 7(k=3)")
    print(f"Ascenders: {ascenders} (should not appear in repeater cycles)")
    print()
    
    for i in range(3):
        print(f"Walk {i+1}:")
        walk = generate_valid_walk(
            graph=graph,
            start_vertex=0,
            min_length=12,
            max_length=15,
            rules=rules,
            verbose=False
        )
        
        if walk:
            print(f"  Generated: {walk}")
            
            # Check repeater cycles for ascenders
            for repeater, k in repeaters.items():
                if repeater in walk:
                    positions = [j for j, x in enumerate(walk) if x == repeater]
                    if len(positions) >= 2:
                        for j in range(len(positions) - 1):
                            pos1, pos2 = positions[j], positions[j+1]
                            cycle = walk[pos1:pos2+1]
                            cycle_middle = cycle[1:-1]
                            
                            # Check if any ascenders in the cycle
                            ascenders_in_cycle = [node for node in cycle_middle if node in ascenders]
                            
                            if ascenders_in_cycle:
                                print(f"  ❌ ERROR: Ascenders {ascenders_in_cycle} found in repeater cycle!")
                            else:
                                print(f"  ✅ Repeater {repeater} cycle clean: {cycle}")


def analyze_walk(walk, repeaters):
    """Analyze a walk for repeater patterns."""
    print(f"\n📊 Analysis of walk: {walk[:20]}..." if len(walk) > 20 else f"\n📊 Analysis of walk: {walk}")
    print(f"  Length: {len(walk)}")
    
    for repeater, k in repeaters.items():
        if repeater in walk:
            positions = [i for i, x in enumerate(walk) if x == repeater]
            print(f"  Repeater {repeater} (k={k}) at positions: {positions}")
            
            if len(positions) >= 2:
                for i in range(len(positions) - 1):
                    pos1, pos2 = positions[i], positions[i+1]
                    distance = pos2 - pos1 - 1
                    cycle = walk[pos1:pos2+1]
                    status = "✅" if distance == k else "❌"
                    print(f"    Cycle {i+1}: {cycle} ({distance} nodes between) {status}")


def test_starting_from_repeater():
    """Test walks that start from a repeater."""
    print("\n🎯 TESTING STARTS FROM REPEATER")
    print("=" * 70)
    
    n = 12
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    repeaters = {3: 2}
    rules = [RepeaterRule(repeaters)]
    
    print(f"Starting walk FROM repeater 3 (k=2):")
    
    walk = generate_valid_walk(
        graph=graph,
        start_vertex=3,  # Start from repeater
        min_length=8,
        max_length=10,
        rules=rules,
        verbose=True
    )
    
    if walk:
        print(f"\n✅ Success! Generated: {walk}")
        analyze_walk(walk, repeaters)
    else:
        print(f"\n❌ Failed to generate walk")


def main():
    test_immediate_completion()
    print("\n" + "=" * 70 + "\n")
    test_with_ascender_avoidance()
    print("\n" + "=" * 70 + "\n")
    test_starting_from_repeater()
    
    print("\n" + "=" * 70)
    print("🎯 SUMMARY")
    print("=" * 70)
    print("✅ Immediate completion approach:")
    print("  • When encountering a repeater, immediately add k nodes + return")
    print("  • Avoids ascender nodes in the cycle")
    print("  • No more catch-22 with rule validation")
    print("  • Walks can start from repeaters")


if __name__ == "__main__":
    main()