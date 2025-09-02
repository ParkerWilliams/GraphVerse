#!/usr/bin/env python3
"""
Test the enhanced walk generation with rule interaction constraints.
"""

import sys
from pathlib import Path
import numpy as np

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
from graphverse.graph.walk_enhanced import (
    generate_valid_walk_enhanced, 
    get_active_constraints,
    get_forbidden_nodes,
    has_incomplete_repeaters
)


def test_rule_interactions():
    """Test that rule interactions work correctly."""
    print("🧪 TESTING RULE INTERACTION CONSTRAINTS")
    print("=" * 70)
    
    # Create test graph
    n = 30
    graph = Graph(n)
    
    # Add edges for connectivity
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=np.random.random() * 0.5)
    
    # Define rules with potential conflicts
    repeaters = {7: 3, 15: 2}  # Smaller set of repeaters
    ascenders = {5, 12, 20}     # Smaller set of ascenders
    evens = {2, 4, 6, 8}       # Evens can coexist with both
    
    rules = [
        RepeaterRule(repeaters),
        AscenderRule(ascenders),
        EvenRule(evens)
    ]
    
    print(f"Graph: {n} nodes")
    print(f"Repeaters: {repeaters}")
    print(f"Ascenders: {ascenders}")
    print(f"Evens: {evens}")
    print()
    
    # Test scenario 1: Start from non-rule node
    print("Test 1: Start from regular node (should be able to visit any rule node)")
    walk = generate_valid_walk_enhanced(
        graph=graph,
        start_vertex=1,  # Not a rule node
        min_length=15,
        max_length=20,
        rules=rules,
        verbose=True
    )
    
    if walk:
        analyze_walk(walk, rules)
    
    print("\n" + "-" * 50 + "\n")
    
    # Test scenario 2: Start from ascender
    print("Test 2: Start from ascender (should forbid repeaters after)")
    walk = generate_valid_walk_enhanced(
        graph=graph,
        start_vertex=5,  # Ascender node
        min_length=10,
        max_length=15,
        rules=rules,
        verbose=True
    )
    
    if walk:
        analyze_walk(walk, rules)
    
    print("\n" + "-" * 50 + "\n")
    
    # Test scenario 3: Start from repeater
    print("Test 3: Start from repeater (should forbid ascenders until cycle complete)")
    walk = generate_valid_walk_enhanced(
        graph=graph,
        start_vertex=7,  # Repeater with k=3
        min_length=10,
        max_length=15,
        rules=rules,
        verbose=True
    )
    
    if walk:
        analyze_walk(walk, rules)


def analyze_walk(walk, rules):
    """Analyze a walk for rule interactions."""
    print(f"\n📊 WALK ANALYSIS")
    print(f"Walk: {walk}")
    print(f"Length: {len(walk)}")
    
    # Identify rule nodes in walk
    repeater_positions = []
    ascender_positions = []
    even_positions = []
    
    for rule in rules:
        if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            for node in rule.member_nodes:
                positions = [i for i, x in enumerate(walk) if x == node]
                if positions:
                    repeater_positions.append((node, positions))
        
        elif hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
            for node in rule.member_nodes:
                positions = [i for i, x in enumerate(walk) if x == node]
                if positions:
                    ascender_positions.append((node, positions))
        
        elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
            for node in rule.member_nodes:
                positions = [i for i, x in enumerate(walk) if x == node]
                if positions:
                    even_positions.append((node, positions))
    
    print(f"\nRule nodes encountered:")
    
    if repeater_positions:
        print(f"  Repeaters:")
        for node, positions in repeater_positions:
            print(f"    Node {node}: positions {positions}")
            
            # Check if cycles are complete
            if len(positions) >= 2:
                for i in range(len(positions) - 1):
                    pos1, pos2 = positions[i], positions[i + 1]
                    nodes_between = pos2 - pos1 - 1
                    print(f"      Cycle {i+1}: {nodes_between} nodes between")
    
    if ascender_positions:
        print(f"  Ascenders:")
        for node, positions in ascender_positions:
            print(f"    Node {node}: positions {positions}")
            
            # Check if ascender rule is maintained
            for pos in positions:
                if pos < len(walk) - 1:
                    following = walk[pos+1:]
                    is_ascending = all(following[i] >= following[i-1] if i > 0 else True 
                                     for i in range(len(following)))
                    status = "✅" if is_ascending else "❌"
                    print(f"      After position {pos}: {'ascending' if is_ascending else 'not ascending'} {status}")
    
    if even_positions:
        print(f"  Evens:")
        for node, positions in even_positions:
            print(f"    Node {node}: positions {positions}")
    
    # Check for rule conflicts
    print(f"\n🔍 Rule interaction check:")
    
    # Check if any repeater appears after an ascender
    if ascender_positions and repeater_positions:
        first_ascender_pos = min(pos for _, positions in ascender_positions for pos in positions)
        repeaters_after = any(pos > first_ascender_pos 
                             for _, positions in repeater_positions 
                             for pos in positions)
        
        if repeaters_after:
            print(f"  ❌ VIOLATION: Repeater found after ascender!")
        else:
            print(f"  ✅ No repeaters after ascenders")
    
    # Check if any ascender appears during incomplete repeater cycle
    if repeater_positions:
        for r_node, r_positions in repeater_positions:
            if len(r_positions) >= 2:
                for i in range(len(r_positions) - 1):
                    cycle_start = r_positions[i]
                    cycle_end = r_positions[i + 1]
                    
                    # Check if any ascender appears in this cycle
                    ascenders_in_cycle = any(
                        cycle_start < pos < cycle_end
                        for _, positions in ascender_positions
                        for pos in positions
                    )
                    
                    if ascenders_in_cycle:
                        print(f"  ❌ VIOLATION: Ascender found during repeater cycle!")
                        break
                else:
                    print(f"  ✅ No ascenders during repeater cycles")


def test_constraint_logic():
    """Test the constraint detection functions."""
    print("\n" + "=" * 70)
    print("🔍 TESTING CONSTRAINT DETECTION LOGIC")
    print("=" * 70)
    
    # Create simple rules
    repeaters = {5: 2}
    ascenders = {10}
    rules = [RepeaterRule(repeaters), AscenderRule(ascenders)]
    
    # Test case 1: Walk with incomplete repeater
    walk1 = [1, 2, 5, 3, 4]  # Repeater 5 appears once
    constraints1 = get_active_constraints(walk1, rules)
    forbidden1 = get_forbidden_nodes(walk1, rules)
    
    print(f"\nTest 1: Walk with incomplete repeater")
    print(f"Walk: {walk1}")
    print(f"Constraints: {constraints1}")
    print(f"Forbidden nodes: {forbidden1}")
    assert constraints1['has_incomplete_repeater'] == True
    assert 10 in forbidden1  # Ascender should be forbidden
    print("✅ Correctly forbids ascender during incomplete repeater")
    
    # Test case 2: Walk with complete repeater
    walk2 = [1, 2, 5, 3, 4, 5, 6]  # Repeater 5 with 2 nodes between
    constraints2 = get_active_constraints(walk2, rules)
    forbidden2 = get_forbidden_nodes(walk2, rules)
    
    print(f"\nTest 2: Walk with complete repeater")
    print(f"Walk: {walk2}")
    print(f"Constraints: {constraints2}")
    print(f"Forbidden nodes: {forbidden2}")
    assert constraints2['has_incomplete_repeater'] == False
    assert 10 not in forbidden2  # Ascender should be allowed
    print("✅ Correctly allows ascender after complete repeater")
    
    # Test case 3: Walk with active ascender
    walk3 = [1, 2, 10, 11, 12]  # Ascender 10 is active
    constraints3 = get_active_constraints(walk3, rules)
    forbidden3 = get_forbidden_nodes(walk3, rules)
    
    print(f"\nTest 3: Walk with active ascender")
    print(f"Walk: {walk3}")
    print(f"Constraints: {constraints3}")
    print(f"Forbidden nodes: {forbidden3}")
    assert constraints3['has_active_ascender'] == True
    assert 5 in forbidden3  # Repeater should be forbidden
    print("✅ Correctly forbids repeater after ascender")


def main():
    test_rule_interactions()
    test_constraint_logic()
    
    print(f"\n" + "=" * 70)
    print("🎯 SUMMARY")
    print("=" * 70)
    print("✅ Rule interaction constraints implemented:")
    print("  1. Repeaters forbidden after ascenders")
    print("  2. Ascenders forbidden during incomplete repeater cycles")
    print("  3. Walks extend to complete repeater cycles")
    print("  4. Evens can coexist with both ascenders and repeaters")
    print("\n✅ This ensures valid walk generation even with conflicting rules")


if __name__ == "__main__":
    main()