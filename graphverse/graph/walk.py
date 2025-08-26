import random
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

def generate_valid_walk(graph, start_vertex, min_length, max_length, rules, max_attempts=10, verbose=False):
    if verbose:
        print(f"Generating valid walk starting from vertex {start_vertex} with rules: {[rule.__class__.__name__ for rule in rules]}")
    target_length = random.randint(min_length, max_length)
    walk = [start_vertex]
    attempts = 0

    while len(walk) < target_length:
        if verbose:
            print(f"Current walk: {walk}, Target length: {target_length}")
        valid_neighbors = [
            neighbor for neighbor in range(graph.n)
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
                walk.pop()
        else:
            current_node = walk[-1]
            
            # Check if current node is a repeater and we should follow its k-cycle
            if (hasattr(graph, 'is_repeater_node') and graph.is_repeater_node(current_node)):
                
                # Get the k-cycle for this repeater
                k_cycle = graph.get_repeater_cycle(current_node)
                if k_cycle and len(k_cycle) > 2:  # Valid k-cycle exists
                    
                    # Get cycle nodes (excluding both start and end repeater since current repeater is already in walk)
                    cycle_nodes = k_cycle[1:-1]  # Skip first and last node (both are the current repeater)
                    
                    # Check if we can fit the full cycle
                    if len(walk) + len(cycle_nodes) <= target_length:
                        # Follow the k-cycle - no validation needed since k-cycles are pre-built to be compliant
                        for cycle_node in cycle_nodes:
                            walk.append(cycle_node)
                        if verbose:
                            print(f"Followed k-cycle for repeater {current_node}: {cycle_nodes}")
                        continue
            
            # Default behavior: random selection from valid neighbors
            next_vertex = random.choice(valid_neighbors)
            if verbose:
                print(f"Adding vertex {next_vertex} to the walk.")

            if not graph.has_edge(walk[-1], next_vertex):
                if verbose:
                    print(f"Adding edge {walk[-1]} -> {next_vertex}")
                graph.add_edge(walk[-1], next_vertex)

            walk.append(next_vertex)

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
