#!/usr/bin/env python3
"""
Test what the CORRECT repeater logic should be vs current implementation.
k=2 repeater should mean exactly 2 nodes between visits: repeater -> node1 -> node2 -> repeater
"""

import numpy as np
import sys
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.rules import RepeaterRule


def test_correct_vs_current_logic():
    """Test the difference between correct and current repeater logic."""
    print("🔍 TESTING CORRECT vs CURRENT REPEATER LOGIC")
    print("=" * 70)
    
    # Define test repeaters
    repeaters = {5: 2, 10: 3, 15: 4}
    
    # Create rule with current implementation
    current_rule = RepeaterRule(repeaters)
    
    print("📋 TEST CASES:")
    
    for repeater_node, k_value in repeaters.items():
        print(f"\n🔄 REPEATER {repeater_node} (k={k_value}):")
        print(f"   CORRECT SPEC: Should require exactly {k_value} nodes between visits")
        
        # CORRECT walk: k nodes between repeater visits
        correct_intermediate = list(range(80, 80 + k_value))  # k nodes
        correct_walk = [9, 7, 11, repeater_node] + correct_intermediate + [repeater_node, 3, 19]
        
        print(f"   ✅ CORRECT walk: {correct_walk}")
        print(f"      Pattern: {repeater_node} → {correct_intermediate} → {repeater_node}")
        print(f"      Nodes between: {len(correct_intermediate)} (should be {k_value})")
        
        # Test with current implementation
        is_valid_current = current_rule.is_satisfied_by(correct_walk, None)
        print(f"      Current rule says: {'✅ VALID' if is_valid_current else '❌ INVALID'}")
        
        # WRONG walk that current implementation accepts: k-1 nodes between
        wrong_intermediate = list(range(80, 80 + k_value - 1))  # k-1 nodes  
        wrong_walk = [9, 7, 11, repeater_node] + wrong_intermediate + [repeater_node, 3, 19]
        
        print(f"   ❌ WRONG walk (current accepts): {wrong_walk}")
        print(f"      Pattern: {repeater_node} → {wrong_intermediate} → {repeater_node}")
        print(f"      Nodes between: {len(wrong_intermediate)} (only {k_value-1}, not {k_value})")
        
        # Test with current implementation  
        is_valid_wrong = current_rule.is_satisfied_by(wrong_walk, None)
        print(f"      Current rule says: {'✅ VALID' if is_valid_wrong else '❌ INVALID'}")
        
        print(f"\n   🎯 CONCLUSION:")
        if is_valid_current and not is_valid_wrong:
            print(f"      ✅ Current implementation works correctly")
        elif not is_valid_current and is_valid_wrong:
            print(f"      ❌ Current implementation has off-by-one error")
            print(f"         Accepts {k_value-1} nodes but rejects {k_value} nodes")
        elif is_valid_current and is_valid_wrong:
            print(f"      ⚠️ Current implementation accepts both (too permissive)")
        else:
            print(f"      ❌ Current implementation rejects both (too restrictive)")


def analyze_position_vs_node_counting():
    """Analyze the difference between position distance and node counting."""
    print(f"\n" + "=" * 70)
    print("📊 POSITION DISTANCE vs NODE COUNTING ANALYSIS")
    print("=" * 70)
    
    # Example walk with k=2 repeater at node 5
    walk = [9, 7, 11, 5, 12, 81, 5, 3, 19]  # Correct: 2 nodes between
    positions_of_5 = [i for i, x in enumerate(walk) if x == 5]
    
    print(f"Example walk: {walk}")
    print(f"Repeater 5 appears at positions: {positions_of_5}")
    
    pos1, pos2 = positions_of_5[0], positions_of_5[1]
    position_distance = pos2 - pos1
    nodes_between = pos2 - pos1 - 1
    sequence = walk[pos1:pos2+1]
    
    print(f"\nSequence: {sequence}")
    print(f"Position distance: {position_distance} (pos {pos2} - pos {pos1})")
    print(f"Nodes between: {nodes_between} (distance - 1)")
    print(f"Intermediate nodes: {sequence[1:-1]}")
    
    print(f"\n📋 RULE INTERPRETATIONS:")
    print(f"Current implementation: checks if position_distance == k")
    print(f"   For k=2: position_distance should be 2")
    print(f"   This gives {position_distance-1} = {nodes_between} nodes between")
    print(f"   Result: ❌ WRONG (only 1 node between, not 2)")
    
    print(f"\nCorrect interpretation: checks if nodes_between == k") 
    print(f"   For k=2: nodes_between should be 2")
    print(f"   This requires position_distance = k + 1 = 3")
    print(f"   Result: ✅ CORRECT (exactly 2 nodes between)")
    
    # Show the bug in current implementation
    print(f"\n🐛 THE BUG:")
    print(f"Current code: if indices[i + 1] - indices[i] != k:")
    print(f"Should be:    if indices[i + 1] - indices[i] != k + 1:")
    print(f"Or:           if (indices[i + 1] - indices[i] - 1) != k:")


def show_real_examples():
    """Show real examples of what walks should look like."""
    print(f"\n" + "=" * 70)
    print("🎯 REAL WORLD EXAMPLES")
    print("=" * 70)
    
    examples = [
        (5, 2, [91, 12, 15, 16, 5, 49, 87, 5, 78, 62]),  # k=2: exactly 2 nodes between
        (10, 3, [44, 23, 10, 55, 66, 77, 10, 88, 99]),   # k=3: exactly 3 nodes between  
        (15, 4, [1, 2, 15, 30, 40, 50, 60, 15, 70])      # k=4: exactly 4 nodes between
    ]
    
    current_rule = RepeaterRule({5: 2, 10: 3, 15: 4})
    
    for repeater, k, walk in examples:
        print(f"\n🔄 Example: {repeater}-repeater with k={k}")
        print(f"Walk: {walk}")
        
        # Find repeater positions
        positions = [i for i, x in enumerate(walk) if x == repeater]
        if len(positions) >= 2:
            pos1, pos2 = positions[0], positions[1]
            sequence = walk[pos1:pos2+1]
            nodes_between = len(sequence) - 2
            
            print(f"Sequence: {sequence}")
            print(f"Nodes between: {nodes_between} (required: {k})")
            print(f"✅ CORRECT: Exactly {k} nodes between repeater visits")
            
            # Test current rule
            is_valid = current_rule.is_satisfied_by(walk, None)
            print(f"Current implementation says: {'✅ VALID' if is_valid else '❌ INVALID'}")
            
            if not is_valid:
                print(f"❌ Current rule rejects this correct walk!")


def main():
    test_correct_vs_current_logic()
    analyze_position_vs_node_counting()
    show_real_examples()


if __name__ == "__main__":
    main()