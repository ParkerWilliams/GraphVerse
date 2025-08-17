import random
from .rules import Rule

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

def generate_multiple_walks(graph, num_walks, min_length, max_length, rules, verbose=False):
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
