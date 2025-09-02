#!/usr/bin/env python3
"""
Test the repeater node extension issue and demonstrate the fix.
"""

import sys
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule
from graphverse.graph.walk import check_rule_compliance


def test_repeater_extension_problem():
    """Test the current problem where walks don't extend for incomplete repeater cycles."""
    print("🐛 TESTING REPEATER EXTENSION PROBLEM")
    print("=" * 70)
    
    # Create test scenario: k=3 repeater at node 7
    repeaters = {7: 3}
    rule = RepeaterRule(repeaters)
    
    # Example from user: walk ends with repeater near the end
    problem_walk = [77, 1, 5, 51, 2, 6, 5, 9, 7, 41]  # Incomplete - stops at 41
    correct_walk = [77, 1, 5, 51, 2, 6, 5, 9, 7, 41, 22, 6, 7]  # Complete cycle
    
    print(f"Problem walk (incomplete): {problem_walk}")
    print(f"Length: {len(problem_walk)}")
    
    # Check if problem walk is valid
    is_valid_problem = rule.is_satisfied_by(problem_walk, None)
    print(f"Is problem walk valid? {'✅ YES' if is_valid_problem else '❌ NO'}")
    
    # Find repeater position
    repeater_positions = [i for i, x in enumerate(problem_walk) if x == 7]
    print(f"Repeater 7 appears at positions: {repeater_positions}")
    
    if len(repeater_positions) == 1:
        print(f"⚠️  ISSUE: Repeater appears only once, cannot form k=3 cycle")
        print(f"Walk should extend to complete the cycle")
    
    print(f"\nCorrect walk (complete): {correct_walk}")
    print(f"Length: {len(correct_walk)}")
    
    # Check if correct walk is valid
    is_valid_correct = rule.is_satisfied_by(correct_walk, None)
    print(f"Is correct walk valid? {'✅ YES' if is_valid_correct else '❌ NO'}")
    
    # Analyze the correct walk
    repeater_positions_correct = [i for i, x in enumerate(correct_walk) if x == 7]
    print(f"Repeater 7 appears at positions: {repeater_positions_correct}")
    
    if len(repeater_positions_correct) >= 2:
        pos1, pos2 = repeater_positions_correct[0], repeater_positions_correct[1]
        nodes_between = pos2 - pos1 - 1
        sequence = correct_walk[pos1:pos2+1]
        print(f"Cycle: positions {pos1} → {pos2}")
        print(f"Sequence: {sequence}")
        print(f"Nodes between: {nodes_between} (required: 3)")
        
        if nodes_between == 3:
            print("✅ Correct cycle completed")
        else:
            print("❌ Cycle still incorrect")


def needs_repeater_extension(walk, repeaters):
    """
    Check if a walk needs extension to complete incomplete repeater cycles.
    
    Args:
        walk: The current walk
        repeaters: Dict of {repeater_node: k_value}
        
    Returns:
        List of (repeater_node, k_value) that need completion
    """
    incomplete_repeaters = []
    
    for repeater_node, k_value in repeaters.items():
        if repeater_node in walk:
            positions = [i for i, x in enumerate(walk) if x == repeater_node]
            
            # Check if last occurrence is incomplete
            if len(positions) >= 1:
                last_pos = positions[-1]
                
                # If repeater appears only once, it's incomplete
                if len(positions) == 1:
                    incomplete_repeaters.append((repeater_node, k_value))
                
                # If last occurrence is too close to end to complete a cycle
                elif len(positions) >= 2:
                    second_last_pos = positions[-2]
                    nodes_between_last_cycle = last_pos - second_last_pos - 1
                    
                    # If the last cycle is incomplete
                    if nodes_between_last_cycle != k_value:
                        incomplete_repeaters.append((repeater_node, k_value))
    
    return incomplete_repeaters


def test_extension_detection():
    """Test the logic for detecting when walks need extension."""
    print(f"\n" + "=" * 70)
    print("🔍 TESTING EXTENSION DETECTION LOGIC")
    print("=" * 70)
    
    repeaters = {7: 3, 12: 2}
    
    test_cases = [
        {
            'walk': [1, 2, 7, 4, 5],  # Incomplete: 7 appears once
            'description': 'Single repeater visit (needs extension)'
        },
        {
            'walk': [1, 2, 7, 4, 5, 6, 7, 8, 9],  # Complete: 7 has 3 nodes between
            'description': 'Complete repeater cycle (no extension needed)'
        },
        {
            'walk': [1, 2, 7, 4, 5, 7, 8, 9],  # Incomplete: only 2 nodes between 7s
            'description': 'Incomplete repeater cycle (needs extension)'
        },
        {
            'walk': [1, 12, 3, 12, 5, 7, 8],  # 12 complete (2 nodes), 7 incomplete
            'description': 'Mixed: one complete, one incomplete'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        walk = case['walk']
        desc = case['description']
        
        print(f"\nTest {i}: {desc}")
        print(f"Walk: {walk}")
        
        incomplete = needs_repeater_extension(walk, repeaters)
        
        if incomplete:
            print(f"❌ Needs extension for: {incomplete}")
        else:
            print(f"✅ No extension needed")
        
        # Verify with rule compliance
        rule = RepeaterRule(repeaters)
        is_valid = rule.is_satisfied_by(walk, None)
        print(f"Current rule validation: {'✅ VALID' if is_valid else '❌ INVALID'}")


def propose_fix():
    """Propose the fix for the walk generation algorithm."""
    print(f"\n" + "=" * 70)
    print("🔧 PROPOSED FIX FOR WALK GENERATION")
    print("=" * 70)
    
    print("""
The current generate_valid_walk() algorithm should be modified to:

1. CURRENT ISSUE:
   - Stops when len(walk) >= target_length
   - Doesn't check for incomplete repeater cycles
   - Returns walks that violate repeater rules

2. PROPOSED FIX:
   - After reaching target_length, check for incomplete repeater cycles
   - Extend walk to complete any incomplete cycles
   - Only stop when all repeater rules are satisfied

3. ALGORITHM MODIFICATION:

   while len(walk) < target_length OR has_incomplete_repeaters(walk, rules):
       # Generate next step
       # If past target_length, only allow moves that complete repeater cycles
       
4. IMPLEMENTATION STEPS:
   a) Add has_incomplete_repeaters() function
   b) Modify main loop condition in generate_valid_walk()
   c) Add logic to prioritize repeater completion when past target_length
   d) Ensure walk doesn't become infinitely long

5. BENEFITS:
   - All generated walks will be rule-compliant
   - No more incomplete repeater cycles
   - Training data will be higher quality
   - Models will learn correct patterns
""")


def main():
    test_repeater_extension_problem()
    test_extension_detection()
    propose_fix()
    
    print(f"\n" + "=" * 70)
    print("🎯 SUMMARY")
    print("=" * 70)
    print("❌ CONFIRMED: Current walk generation has repeater extension bug")
    print("❌ PROBLEM: Walks can end with incomplete repeater cycles") 
    print("❌ IMPACT: Training data contains rule-violating walks")
    print("🔧 SOLUTION: Modify generate_valid_walk() to extend for repeater completion")
    print("✅ BENEFIT: Higher quality training data with correct rule compliance")


if __name__ == "__main__":
    main()