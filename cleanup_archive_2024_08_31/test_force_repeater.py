#!/usr/bin/env python3
"""
Test walk generation starting FROM a repeater to ensure cycles complete.
"""

import sys
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule
from graphverse.graph.walk import generate_valid_walk


def test_starting_from_repeater():
    """Test walks that start from a repeater node."""
    print("🎯 TESTING WALKS STARTING FROM REPEATER")
    print("=" * 70)
    
    n = 10
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    repeaters = {3: 2}  # k=2 at node 3
    rules = [RepeaterRule(repeaters)]
    
    print(f"Setup: {n} nodes, repeater at node 3 with k=2")
    print("Starting walks FROM the repeater node 3")
    print()
    
    for i in range(5):
        print(f"Walk {i+1}:")
        walk = generate_valid_walk(
            graph=graph,
            start_vertex=3,  # START from repeater
            min_length=6,
            max_length=8,
            rules=rules,
            verbose=False
        )
        
        if walk:
            print(f"  Generated: {walk} (len={len(walk)})")
            
            # Must have repeater 3
            positions = [j for j, x in enumerate(walk) if x == 3]
            print(f"  Repeater 3 at positions: {positions}")
            
            if len(positions) == 1:
                print(f"  ❌ ERROR: Only one visit! Walk should have extended!")
            elif len(positions) >= 2:
                for j in range(len(positions) - 1):
                    dist = positions[j+1] - positions[j] - 1
                    status = "✅" if dist == 2 else "❌"
                    print(f"    Cycle {j+1}: {dist} nodes between {status}")
                
                if len(walk) > 8:
                    print(f"  📏 Extended to {len(walk)} to complete cycle")
        else:
            print(f"  ❌ Failed to generate")
    
    print("\n" + "=" * 70)
    print("✅ When starting from a repeater:")
    print("  • Walk MUST complete the cycle")
    print("  • Will extend beyond max_length if needed")
    print("  • Single visits should never occur")


def main():
    test_starting_from_repeater()


if __name__ == "__main__":
    main()