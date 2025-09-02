#!/usr/bin/env python3
"""
Quick test to create and validate a walk with a k=7 repeater.
"""

import sys
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.rules import RepeaterRule


def test_k7_repeater():
    """Test a walk with a k=7 repeater."""
    print("🧪 TESTING K=7 REPEATER WALK")
    print("=" * 50)
    
    # Define a k=7 repeater at node 15
    repeaters = {15: 7}
    rule = RepeaterRule(repeaters)
    
    # Create a walk of length 20 with k=7 repeater
    # Pattern: prefix + repeater + 7_nodes + repeater + suffix
    prefix = [91, 42, 63, 88]  # 4 nodes before
    repeater_node = 15
    seven_nodes = [12, 34, 56, 78, 90, 11, 33]  # exactly 7 nodes between
    suffix = [99, 77, 55, 44, 22, 88, 66, 11]  # 8 nodes after
    
    walk = prefix + [repeater_node] + seven_nodes + [repeater_node] + suffix
    
    print(f"Generated walk (length {len(walk)}):")
    print(f"  {walk}")
    print(f"\nWalk structure:")
    print(f"  Prefix: {prefix}")
    print(f"  Repeater: {repeater_node}")
    print(f"  Seven nodes: {seven_nodes}")
    print(f"  Repeater: {repeater_node}")
    print(f"  Suffix: {suffix}")
    
    # Find repeater positions
    positions = [i for i, x in enumerate(walk) if x == repeater_node]
    print(f"\nRepeater positions: {positions}")
    
    if len(positions) >= 2:
        pos1, pos2 = positions[0], positions[1]
        sequence = walk[pos1:pos2+1]
        nodes_between = len(sequence) - 2
        position_distance = pos2 - pos1
        
        print(f"\nAnalysis:")
        print(f"  Position 1: {pos1}")
        print(f"  Position 2: {pos2}")
        print(f"  Position distance: {position_distance}")
        print(f"  Sequence: {sequence}")
        print(f"  Nodes between: {nodes_between}")
        print(f"  Expected nodes between: 7")
        
        # Test with the corrected rule
        is_valid = rule.is_satisfied_by(walk, None)
        
        print(f"\n🎯 RESULT:")
        print(f"  Corrected RepeaterRule validation: {'✅ VALID' if is_valid else '❌ INVALID'}")
        
        if is_valid:
            print(f"  ✅ SUCCESS: Walk correctly follows k=7 repeater rule!")
        else:
            print(f"  ❌ FAILED: Walk violates k=7 repeater rule!")
            
        return is_valid
    else:
        print("❌ ERROR: Not enough repeater visits found")
        return False


def test_invalid_k7_walk():
    """Test an invalid walk with k=7 repeater (should have only 6 nodes between)."""
    print(f"\n" + "=" * 50)
    print("🧪 TESTING INVALID K=7 REPEATER WALK")
    print("=" * 50)
    
    repeaters = {15: 7}
    rule = RepeaterRule(repeaters)
    
    # Create an INVALID walk (only 6 nodes between instead of 7)
    prefix = [91, 42, 63, 88]
    repeater_node = 15
    six_nodes = [12, 34, 56, 78, 90, 11]  # only 6 nodes (should be invalid)
    suffix = [99, 77, 55, 44, 22, 88, 66, 11, 33]
    
    invalid_walk = prefix + [repeater_node] + six_nodes + [repeater_node] + suffix
    
    print(f"Invalid walk (length {len(invalid_walk)}):")
    print(f"  {invalid_walk}")
    
    positions = [i for i, x in enumerate(invalid_walk) if x == repeater_node]
    pos1, pos2 = positions[0], positions[1]
    sequence = invalid_walk[pos1:pos2+1]
    nodes_between = len(sequence) - 2
    
    print(f"\nAnalysis:")
    print(f"  Sequence: {sequence}")
    print(f"  Nodes between: {nodes_between} (expected: 7)")
    
    is_valid = rule.is_satisfied_by(invalid_walk, None)
    
    print(f"\n🎯 RESULT:")
    print(f"  Corrected RepeaterRule validation: {'✅ VALID' if is_valid else '❌ INVALID'}")
    
    if not is_valid:
        print(f"  ✅ SUCCESS: Rule correctly rejects invalid walk!")
    else:
        print(f"  ❌ FAILED: Rule incorrectly accepts invalid walk!")
    
    return not is_valid


def main():
    valid_result = test_k7_repeater()
    invalid_result = test_invalid_k7_walk()
    
    print(f"\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Valid k=7 walk test: {'✅ PASSED' if valid_result else '❌ FAILED'}")
    print(f"Invalid k=7 walk test: {'✅ PASSED' if invalid_result else '❌ FAILED'}")
    
    if valid_result and invalid_result:
        print("🎉 ALL TESTS PASSED - RepeaterRule working correctly!")
    else:
        print("❌ SOME TESTS FAILED - RepeaterRule needs more work!")


if __name__ == "__main__":
    main()