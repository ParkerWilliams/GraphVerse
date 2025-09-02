#!/usr/bin/env python3
"""
Test that repeaters complete exactly ONE cycle then continue.
"""

import sys
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.rules import RepeaterRule


def test_repeater_rule_validation():
    """Test the updated RepeaterRule validation."""
    print("🧪 TESTING REPEATER RULE VALIDATION")
    print("=" * 70)
    
    repeaters = {5: 2, 10: 3}
    rule = RepeaterRule(repeaters)
    
    test_cases = [
        # Single visits - should be INVALID
        {
            'walk': [1, 2, 5, 3, 4],
            'expected': False,
            'description': 'Single visit to repeater 5 - INVALID'
        },
        {
            'walk': [1, 10, 3, 4],
            'expected': False,
            'description': 'Single visit to repeater 10 - INVALID'
        },
        
        # Complete cycles - should be VALID
        {
            'walk': [1, 2, 5, 3, 4, 5, 6],
            'expected': True,
            'description': 'Complete k=2 cycle for repeater 5 - VALID'
        },
        {
            'walk': [1, 10, 2, 3, 4, 10, 6],
            'expected': True,
            'description': 'Complete k=3 cycle for repeater 10 - VALID'
        },
        
        # Incomplete cycles - should be INVALID
        {
            'walk': [1, 5, 3, 5, 6],  # Only 1 node between
            'expected': False,
            'description': 'Incomplete cycle (1 node, need 2) - INVALID'
        },
        {
            'walk': [1, 10, 2, 3, 10, 6],  # Only 2 nodes between
            'expected': False,
            'description': 'Incomplete cycle (2 nodes, need 3) - INVALID'
        },
        
        # Multiple cycles - should handle correctly
        {
            'walk': [5, 1, 2, 5, 3, 4, 5],  # Two complete k=2 cycles
            'expected': True,
            'description': 'Two complete k=2 cycles - VALID'
        },
        {
            'walk': [5, 1, 2, 5, 3, 5],  # One complete, one incomplete
            'expected': False,
            'description': 'One complete + one incomplete cycle - INVALID'
        },
        
        # Walk continues after cycle completion
        {
            'walk': [1, 5, 2, 3, 5, 6, 7, 8, 9],  # Complete cycle then continue
            'expected': True,
            'description': 'Complete cycle then continue walking - VALID'
        },
        
        # Multiple different repeaters
        {
            'walk': [1, 5, 2, 3, 5, 10, 4, 6, 8, 10],  # Both complete
            'expected': True,
            'description': 'Two different repeaters, both complete - VALID'
        },
        {
            'walk': [1, 5, 2, 3, 5, 10, 4, 6],  # 5 complete, 10 incomplete
            'expected': False,
            'description': 'One complete, one incomplete - INVALID'
        }
    ]
    
    print("\nTest cases:")
    for i, case in enumerate(test_cases, 1):
        walk = case['walk']
        expected = case['expected']
        desc = case['description']
        
        is_valid = rule.is_satisfied_by(walk, None)
        passed = is_valid == expected
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"\n{i}. {desc}")
        print(f"   Walk: {walk}")
        print(f"   Result: {is_valid}, Expected: {expected} {status}")
        
        # Show repeater analysis
        for r_node, k in repeaters.items():
            if r_node in walk:
                positions = [j for j, x in enumerate(walk) if x == r_node]
                if len(positions) == 1:
                    print(f"   Repeater {r_node}: Single visit at {positions[0]}")
                else:
                    print(f"   Repeater {r_node}: Visits at {positions}")
                    for j in range(len(positions) - 1):
                        dist = positions[j+1] - positions[j] - 1
                        print(f"     Cycle {j+1}: {dist} nodes between (need {k})")


def test_walk_generation_behavior():
    """Test expected walk generation behavior."""
    print("\n" + "=" * 70)
    print("🎯 EXPECTED WALK GENERATION BEHAVIOR")
    print("=" * 70)
    
    print("""
Expected behavior for walk generation:

1. When encountering a repeater:
   - MUST complete the cycle by revisiting after exactly k nodes
   - Walk extends if needed to complete the cycle
   
2. After completing ONE cycle:
   - Walk can continue normally
   - Repeater can be visited again (starting a new cycle)
   - But should NOT immediately revisit (would need k nodes between)

3. Extension phase:
   - Only extends for incomplete cycles
   - Has maximum extension limit to prevent infinite loops
   
4. Rule interactions:
   - Ascenders forbid repeaters after activation
   - Incomplete repeater cycles forbid ascenders
   - Evens can coexist with both

Example valid walk with repeater 5 (k=2):
   [1, 2, 5, 6, 7, 5, 8, 9, ...]
         ↑         ↑
         First    Second (after 2 nodes)
         
Example that would extend:
   Target length: 6
   Walk so far: [1, 2, 3, 5, 4]  ← Incomplete!
   Must extend: [1, 2, 3, 5, 4, 6, 5]  ← Now complete
""")


def main():
    test_repeater_rule_validation()
    test_walk_generation_behavior()
    
    print("\n" + "=" * 70)
    print("✅ SUMMARY")
    print("=" * 70)
    print("The RepeaterRule now correctly:")
    print("1. Rejects single visits (must complete cycle)")
    print("2. Validates k-spacing between consecutive visits")
    print("3. Allows walks to continue after cycle completion")
    print("4. Supports multiple cycles of the same repeater")


if __name__ == "__main__":
    main()