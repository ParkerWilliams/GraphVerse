"""
Enhanced walk generation with rule interaction constraints.

Key constraints:
1. After encountering an ascender, forbid visiting repeaters
2. During an incomplete repeater cycle, forbid visiting ascenders
3. After completing a repeater cycle, ascenders are allowed again
"""

import random
from .rules import Rule
from typing import List, Optional, Set


def check_rule_compliance(walk, graph, rules, verbose=False):
    """Check if walk complies with all rules."""
    if verbose:
        print(f"Checking rule compliance for walk: {walk}")
    for rule in rules:
        if verbose:
            print(f"Checking rule: {rule.__class__.__name__}")
        result = rule.apply(walk, graph)
        if not result:
            if verbose:
                print(f"Rule {rule.__class__.__name__} violated for walk: {walk}")
            return False
    if verbose:
        print(f"Rule compliance check completed for walk: {walk}")
    return True


def get_active_constraints(walk, rules):
    """
    Determine which rule constraints are currently active.
    
    Returns:
        dict with:
        - 'has_active_ascender': True if walk has hit an ascender and is ascending
        - 'has_incomplete_repeater': True if walk has incomplete repeater cycle
        - 'active_repeaters': Set of repeater nodes with incomplete cycles
    """
    constraints = {
        'has_active_ascender': False,
        'has_incomplete_repeater': False,
        'active_repeaters': set()
    }
    
    # Check for active ascender constraint
    for rule in rules:
        if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
            for i, node in enumerate(walk):
                if node in rule.member_nodes:
                    # Ascender is active from this point onward
                    constraints['has_active_ascender'] = True
                    break
    
    # Check for incomplete repeater cycles
    for rule in rules:
        if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            for repeater_node, k_value in rule.members_nodes_dict.items():
                if repeater_node in walk:
                    positions = [i for i, x in enumerate(walk) if x == repeater_node]
                    
                    # Single visit = incomplete
                    if len(positions) == 1:
                        constraints['has_incomplete_repeater'] = True
                        constraints['active_repeaters'].add(repeater_node)
                    
                    # Check if last cycle is complete
                    elif len(positions) >= 2:
                        last_pos = positions[-1]
                        second_last_pos = positions[-2]
                        nodes_between = last_pos - second_last_pos - 1
                        
                        if nodes_between != k_value:
                            constraints['has_incomplete_repeater'] = True
                            constraints['active_repeaters'].add(repeater_node)
    
    return constraints


def get_forbidden_nodes(walk, rules):
    """
    Get set of nodes that should be forbidden based on rule interactions.
    
    Returns:
        Set of node IDs that should not be visited next
    """
    forbidden = set()
    constraints = get_active_constraints(walk, rules)
    
    # If we have an active ascender, forbid all repeater nodes
    if constraints['has_active_ascender']:
        for rule in rules:
            if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                forbidden.update(rule.member_nodes)
    
    # If we have incomplete repeater cycles, forbid all ascender nodes
    if constraints['has_incomplete_repeater']:
        for rule in rules:
            if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
                forbidden.update(rule.member_nodes)
    
    return forbidden


def has_incomplete_repeaters(walk, rules):
    """
    Check if walk has incomplete repeater cycles that need extension.
    
    A repeater MUST complete its cycle - single visits are incomplete!
    However, we only check the LAST occurrence to avoid infinite loops.
    
    Args:
        walk: Current walk
        rules: List of rule objects
    
    Returns:
        True if walk has incomplete repeater cycles, False otherwise
    """
    for rule in rules:
        if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            for repeater_node, k_value in rule.members_nodes_dict.items():
                if repeater_node in walk:
                    positions = [i for i, x in enumerate(walk) if x == repeater_node]
                    
                    # Single visit = incomplete cycle, MUST complete
                    if len(positions) == 1:
                        return True
                    
                    # For multiple visits, only check if the LAST cycle is complete
                    # This prevents infinite loops - we complete one cycle and move on
                    if len(positions) >= 2:
                        # Check only the last pair
                        last_pos = positions[-1]
                        second_last_pos = positions[-2]
                        nodes_between = last_pos - second_last_pos - 1
                        
                        # If last cycle doesn't have exactly k nodes between, it's incomplete
                        if nodes_between != k_value:
                            return True
                        # If last cycle is complete, we're done with this repeater
    
    return False


def generate_valid_walk_enhanced(graph, start_vertex, min_length, max_length, rules, 
                                 max_attempts=10, max_extension_attempts=20, verbose=False):
    """
    Generate a valid walk with rule interaction constraints.
    
    Key improvements:
    1. Extends walks to complete repeater cycles
    2. Forbids repeaters after ascenders
    3. Forbids ascenders during incomplete repeater cycles
    """
    if verbose:
        print(f"Generating walk from {start_vertex} (length {min_length}-{max_length})")
        print(f"Rules: {[rule.__class__.__name__ for rule in rules]}")
    
    target_length = random.randint(min_length, max_length)
    walk = [start_vertex]
    attempts = 0
    extension_attempts = 0
    
    # Main generation loop
    # Extend for ANY incomplete repeaters (single or multi-visit)
    while len(walk) < target_length or (has_incomplete_repeaters(walk, rules) and 
                                        extension_attempts < max_extension_attempts):
        
        in_extension_phase = len(walk) >= target_length
        
        if verbose:
            phase = "Extension" if in_extension_phase else "Generation"
            print(f"[{phase}] Walk length: {len(walk)}, Target: {target_length}")
            if in_extension_phase:
                print(f"  Completing incomplete repeater cycles...")
        
        # Get forbidden nodes based on rule interactions
        forbidden_nodes = get_forbidden_nodes(walk, rules)
        
        # Find valid neighbors (excluding forbidden ones)
        valid_neighbors = [
            neighbor for neighbor in range(graph.n)
            if neighbor not in forbidden_nodes and
               check_rule_compliance(walk + [neighbor], graph, rules, verbose=False)
        ]
        
        if verbose and forbidden_nodes:
            constraints = get_active_constraints(walk, rules)
            if constraints['has_active_ascender']:
                print(f"  Active ascender - forbidding repeater nodes")
            if constraints['has_incomplete_repeater']:
                print(f"  Incomplete repeater - forbidding ascender nodes")
        
        if not valid_neighbors:
            if in_extension_phase:
                extension_attempts += 1
                if verbose:
                    print(f"  No valid extension. Attempts: {extension_attempts}/{max_extension_attempts}")
                
                if extension_attempts >= max_extension_attempts:
                    if verbose:
                        print("  Max extension attempts reached. Ending walk.")
                    break
                else:
                    # Backtrack during extension
                    if len(walk) > 1:
                        walk.pop()
            else:
                attempts += 1
                if verbose:
                    print(f"  No valid neighbors. Attempts: {attempts}/{max_attempts}")
                
                if attempts >= max_attempts:
                    if verbose:
                        print("  Max attempts reached. Resetting walk.")
                    walk = [start_vertex]
                    attempts = 0
                    target_length = random.randint(min_length, max_length)
                else:
                    # Backtrack during generation
                    if len(walk) > 1:
                        walk.pop()
        else:
            # Choose next vertex
            next_vertex = random.choice(valid_neighbors)
            
            if verbose:
                action = "Extending" if in_extension_phase else "Adding"
                print(f"  {action} vertex {next_vertex}")
                
                # Show if this completes a repeater cycle
                for rule in rules:
                    if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                        if next_vertex in rule.member_nodes:
                            if next_vertex in walk:
                                positions = [i for i, x in enumerate(walk) if x == next_vertex]
                                if positions:
                                    k_value = rule.members_nodes_dict[next_vertex]
                                    nodes_since = len(walk) - positions[-1] - 1
                                    print(f"    Completing repeater {next_vertex} cycle (k={k_value}, nodes since: {nodes_since})")
            
            # Add edge if needed
            if not graph.has_edge(walk[-1], next_vertex):
                graph.add_edge(walk[-1], next_vertex)
            
            walk.append(next_vertex)
            
            # Reset extension attempts on successful addition
            if in_extension_phase:
                extension_attempts = 0
    
    # Final validation
    if len(walk) >= min_length:
        if verbose:
            print(f"✅ Valid walk generated: {walk}")
            print(f"   Length: {len(walk)} (target was {target_length})")
            
            # Show rule interactions that occurred
            constraints = get_active_constraints(walk, rules)
            if constraints['has_active_ascender']:
                print("   Contains active ascender constraint")
            if not has_incomplete_repeaters(walk, rules):
                print("   All repeater cycles complete")
        
        return walk
    else:
        if verbose:
            print(f"❌ Failed to generate valid walk")
        return None


def generate_multiple_walks_enhanced(graph, num_walks, min_length, max_length, rules, 
                                    verbose=False, parallel=False):
    """Generate multiple walks with enhanced rule interaction handling."""
    from tqdm import tqdm
    
    walks = []
    failed_attempts = 0
    
    if verbose:
        print(f"\nGenerating {num_walks} walks (length {min_length}-{max_length})")
        print(f"Rules: {[r.__class__.__name__ for r in rules]}")
        progress_bar = tqdm(total=num_walks, desc="Generating walks", unit="walk")
    
    while len(walks) < num_walks:
        start_vertex = random.choice(list(range(graph.n)))
        
        walk = generate_valid_walk_enhanced(
            graph=graph,
            start_vertex=start_vertex,
            min_length=min_length,
            max_length=max_length,
            rules=rules,
            verbose=False
        )
        
        if walk:
            walks.append(walk)
            if verbose:
                progress_bar.update(1)
                success_rate = len(walks) / (len(walks) + failed_attempts) * 100
                progress_bar.set_postfix({
                    "failed": failed_attempts,
                    "success_rate": f"{success_rate:.1f}%"
                })
        else:
            failed_attempts += 1
    
    if verbose:
        progress_bar.close()
        print(f"✅ Generated {len(walks)} walks")
        if failed_attempts > 0:
            success_rate = len(walks) / (len(walks) + failed_attempts) * 100
            print(f"   Failed attempts: {failed_attempts} ({100-success_rate:.1f}% failure rate)")
    
    return walks