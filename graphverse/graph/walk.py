import random
import numpy as np
from .rules import Rule
from typing import List, Optional

def check_rule_compliance(walk, graph, rules, verbose=False):
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

def has_incomplete_repeaters(walk, rules):
    """
    Check if walk has incomplete repeater cycles that need extension.
    
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
                    
                    # If repeater appears only once, cycle is incomplete
                    if len(positions) == 1:
                        return True
                    
                    # Check if the last cycle is complete
                    if len(positions) >= 2:
                        last_pos = positions[-1]
                        second_last_pos = positions[-2]
                        nodes_between = last_pos - second_last_pos - 1
                        
                        # If last cycle doesn't have exactly k nodes between, it's incomplete
                        if nodes_between != k_value:
                            return True
    
    return False


def generate_valid_walk(graph, start_vertex, min_length, max_length, rules, max_attempts=10, verbose=False):
    if verbose:
        print(f"Generating valid walk starting from vertex {start_vertex} with rules: {[rule.__class__.__name__ for rule in rules]}")
    target_length = random.randint(min_length, max_length)
    walk = [start_vertex]
    attempts = 0
    extension_attempts = 0
    max_extension_attempts = 20  # Prevent infinite loops
    
    # Check if we're starting from a repeater - if so, immediately complete its cycle
    for rule in rules:
        if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            if start_vertex in rule.members_nodes_dict:
                k_value = rule.members_nodes_dict[start_vertex]
                if verbose:
                    print(f"Starting from repeater {start_vertex} (k={k_value}), completing cycle immediately")
                
                # Get ascender nodes to avoid
                ascender_nodes = set()
                for r in rules:
                    if hasattr(r, 'is_ascender_rule') and r.is_ascender_rule:
                        ascender_nodes.update(r.member_nodes)
                
                # Try to build initial cycle using existing edges
                cycle_complete = False
                cycle_attempts = 0
                max_cycle_attempts = 10
                
                while not cycle_complete and cycle_attempts < max_cycle_attempts:
                    temp_walk = [start_vertex]
                    cycle_attempts += 1
                    
                    # Try to find k connected nodes that form a path back to start
                    for i in range(k_value):
                        current = temp_walk[-1]
                        # Get valid neighbors (existing edges only)
                        neighbors = [n for n in graph.get_neighbors(current) 
                                   if n not in ascender_nodes and n not in temp_walk]
                        
                        if not neighbors:
                            break  # Can't continue this path
                        
                        temp_walk.append(random.choice(neighbors))
                    
                    # Check if we can complete the cycle
                    if len(temp_walk) == k_value + 1 and graph.has_edge(temp_walk[-1], start_vertex):
                        walk = temp_walk + [start_vertex]
                        cycle_complete = True
                    
                    if verbose:
                        print(f"Initial cycle completed: {walk}")

    # Main generation loop - with immediate completion, we don't need to check for incomplete repeaters
    while len(walk) < target_length:
        if verbose:
            print(f"Current walk: {walk}, Target length: {target_length}")
        
        # Only consider actual neighbors in the graph
        valid_neighbors = [
            neighbor for neighbor in graph.get_neighbors(walk[-1])
            if check_rule_compliance(walk + [neighbor], graph, rules, verbose)
        ]

        if not valid_neighbors:
            attempts += 1
            if verbose:
                print(f"No valid neighbors found. Attempts: {attempts}/{max_attempts}")

            if attempts >= max_attempts:
                if verbose:
                    print("Maximum attempts reached. Resetting walk.")
                walk = [start_vertex]
                attempts = 0
            else:
                if verbose:
                    print("Backtracking...")
                if len(walk) > 1:  # Don't pop the starting vertex
                    walk.pop()
                else:
                    # Can't backtrack further, reset
                    if verbose:
                        print("Can't backtrack further, resetting")
                    walk = [start_vertex]
                    attempts = 0
        else:
            # Default behavior: random selection from valid neighbors
            next_vertex = random.choice(valid_neighbors)
            if verbose:
                print(f"Adding vertex {next_vertex} to the walk.")

            walk.append(next_vertex)
            
            # Check if we just added a repeater node - if so, immediately complete its cycle
            for rule in rules:
                if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                    if next_vertex in rule.members_nodes_dict:
                        # This is a repeater! Get its k value
                        k_value = rule.members_nodes_dict[next_vertex]
                        
                        # Check if this is the first visit to this repeater
                        repeater_count = walk.count(next_vertex)
                        
                        if repeater_count == 1:  # First visit - immediately complete the cycle
                            if verbose:
                                print(f"  Encountered repeater {next_vertex} (k={k_value}), completing cycle immediately")
                            
                            # Find k valid intermediate nodes (no ascenders during the cycle)
                            # Get ascender nodes to avoid
                            ascender_nodes = set()
                            for r in rules:
                                if hasattr(r, 'is_ascender_rule') and r.is_ascender_rule:
                                    ascender_nodes.update(r.member_nodes)
                            
                            # Try to build a k-cycle using existing edges
                            cycle_complete = False
                            temp_position = len(walk) - 1  # Position of repeater in walk
                            
                            # Try to find k connected nodes that form a path back to repeater
                            for _ in range(5):  # Try a few times
                                temp_walk = walk[:temp_position + 1]  # Copy walk up to repeater
                                success = True
                                
                                for i in range(k_value):
                                    current = temp_walk[-1]
                                    # Get valid neighbors (existing edges only)
                                    neighbors = [n for n in graph.get_neighbors(current) 
                                               if n not in ascender_nodes and n != next_vertex and n not in temp_walk[temp_position:]]
                                    
                                    if not neighbors:
                                        success = False
                                        break
                                    
                                    temp_walk.append(random.choice(neighbors))
                                
                                # Check if we can complete the cycle
                                if success and graph.has_edge(temp_walk[-1], next_vertex):
                                    walk = temp_walk + [next_vertex]
                                    cycle_complete = True
                                    break
                            
                            if cycle_complete and verbose:
                                print(f"    Completed cycle: {walk[-(k_value+2):]}")
                            else:
                                # Not enough nodes to complete cycle - this shouldn't happen in practice
                                if verbose:
                                    print(f"    WARNING: Not enough nodes to complete k={k_value} cycle")
            

    if len(walk) >= min_length:
        if verbose:
            print(f"Valid walk generated: {walk}")
        return walk
    else:
        if verbose:
            print("Failed to generate a valid walk.")
        return None

def generate_multiple_walks(graph, num_walks, min_length, max_length, rules, verbose=False, 
                           parallel=None, n_workers=None, device=None):
    """
    Generate multiple random walks on a graph.
    
    Args:
        graph: Graph object to walk on
        num_walks: Number of walks to generate  
        min_length: Minimum walk length
        max_length: Maximum walk length
        rules: List of rules to follow during generation
        verbose: Whether to show progress information
        parallel: Enable parallel processing (None=auto, True=force, False=sequential)
        n_workers: Number of parallel workers (None=auto-detect)
        device: GPU device for acceleration (None=auto-detect, "cpu"=force CPU)
        
    Returns:
        List of generated walks
    """
    # Auto-decide whether to use parallel processing
    if parallel is None:
        # Use parallel for larger workloads and when multiple cores available
        import multiprocessing
        parallel = (num_walks >= 1000 and multiprocessing.cpu_count() >= 4)
    
    if parallel:
        try:
            from .parallel_walk import generate_multiple_walks_parallel
            return generate_multiple_walks_parallel(
                graph, num_walks, min_length, max_length, rules, 
                verbose=verbose, n_workers=n_workers, device=device
            )
        except ImportError:
            if verbose:
                print("  ⚠ Parallel processing not available, using sequential")
            parallel = False
    
    # Sequential implementation (original code)
    from tqdm import tqdm
    
    walks = []
    attempts = 0
    max_attempts = 10
    total_attempts = 0
    failed_attempts = 0
    
    if verbose:
        print(f"\n  Generating {num_walks} walks (length {min_length}-{max_length})...")
        if not parallel:
            print("  Using sequential processing")
        progress_bar = tqdm(total=num_walks, desc="Generating walks", unit="walk")
    
    while len(walks) < num_walks:
        start_vertex = random.choice(list(range(graph.n)))
        walk = generate_valid_walk(graph, start_vertex, min_length, max_length, rules, max_attempts, verbose=False)
        
        if walk:
            walks.append(walk)
            attempts = 0
            if verbose:
                progress_bar.update(1)
                progress_bar.set_postfix({"failed": failed_attempts, "success_rate": f"{len(walks)/(len(walks)+failed_attempts):.1%}"})
        else:
            attempts += 1
            total_attempts += 1
            failed_attempts += 1
            
            if attempts >= max_attempts:
                attempts = 0
    
    if verbose:
        progress_bar.close()
        print(f"  ✓ {len(walks)} walks generated successfully")
        if failed_attempts > 0:
            print(f"  ⚠ {failed_attempts} failed attempts ({failed_attempts/(len(walks)+failed_attempts):.1%} failure rate)")
    
    return walks


def generate_multiple_walks_sequential(graph, num_walks, min_length, max_length, rules, verbose=False):
    """
    Original sequential implementation of generate_multiple_walks.
    
    This function preserves the exact original behavior for compatibility and testing.
    """
    from tqdm import tqdm
    
    walks = []
    attempts = 0
    max_attempts = 10
    total_attempts = 0
    failed_attempts = 0
    
    if verbose:
        print(f"\n  Generating {num_walks} walks (length {min_length}-{max_length})...")
        progress_bar = tqdm(total=num_walks, desc="Generating walks", unit="walk")
    
    while len(walks) < num_walks:
        start_vertex = random.choice(list(range(graph.n)))
        walk = generate_valid_walk(graph, start_vertex, min_length, max_length, rules, max_attempts, verbose=False)
        
        if walk:
            walks.append(walk)
            attempts = 0
            if verbose:
                progress_bar.update(1)
                progress_bar.set_postfix({"failed": failed_attempts, "success_rate": f"{len(walks)/(len(walks)+failed_attempts):.1%}"})
        else:
            attempts += 1
            total_attempts += 1
            failed_attempts += 1
            
            if attempts >= max_attempts:
                attempts = 0
    
    if verbose:
        progress_bar.close()
        print(f"  ✓ {len(walks)} walks generated successfully")
        if failed_attempts > 0:
            print(f"  ⚠ {failed_attempts} failed attempts ({failed_attempts/(len(walks)+failed_attempts):.1%} failure rate)")
    
    return walks
