#!/usr/bin/env python3
"""
Test with verbose output to see what's happening.
"""

import sys
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule
from graphverse.graph.walk import generate_valid_walk


def test_verbose():
    """Test with verbose to see the issue."""
    print("🔍 VERBOSE TEST")
    print("=" * 70)
    
    n = 10
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    repeaters = {3: 2}
    rules = [RepeaterRule(repeaters)]
    
    print("Attempting to generate walk starting from repeater 3...")
    print()
    
    walk = generate_valid_walk(
        graph=graph,
        start_vertex=3,
        min_length=6,
        max_length=8,
        rules=rules,
        max_attempts=3,  # Low attempts to see what happens
        verbose=True
    )
    
    if walk:
        print(f"\n✅ Success: {walk}")
    else:
        print(f"\n❌ Failed to generate walk")
    
    print("\n" + "=" * 70)
    print("INSIGHT: The issue is that RepeaterRule.is_satisfied_by()")
    print("returns False for incomplete walks during generation!")
    print("\nDuring generation, [3, 1] is invalid because it's a single visit.")
    print("But we need to BUILD the walk step by step...")


def main():
    test_verbose()


if __name__ == "__main__":
    main()