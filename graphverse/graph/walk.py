import random
from tqdm import tqdm

from .rules import Rule


def check_rule_compliance(graph, walk, rules, verbose=False):
    if verbose:
        print(f"Checking rule compliance for walk: {walk}")
    for rule in rules:
        if verbose:
            print(f"Checking rule: {rule.__class__.__name__}")
        result = rule.apply(graph, walk)
        if not result:
            if verbose:
                print(f"Rule {rule.__class__.__name__} violated for walk: {walk}")
            return False
    if verbose:
        print(f"Rule compliance check completed for walk: {walk}")
    return True


def generate_valid_walk(
    graph, start_vertex, min_length, max_length, rules, max_attempts=10, verbose=False
):
    if verbose:
        print(
            f"Generating valid walk starting from vertex {start_vertex} with rules: {[rule.__class__.__name__ for rule in rules]}"
        )
    target_length = random.randint(min_length, max_length)
    walk = [start_vertex]
    attempts = 0

    while len(walk) < target_length:
        if verbose:
            print(f"Current walk: {walk}, Target length: {target_length}")
        valid_neighbors = [
            neighbor
            for neighbor in graph.nodes()
            if check_rule_compliance(graph, walk + [neighbor], rules, verbose)
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


def generate_multiple_walks(
    graph, num_walks, min_length, max_length, rules, verbose=False
):
    if verbose:
        print(f"Generating {num_walks} walks...")
    walks = []
    attempts = 0
    max_attempts = 10
    total_attempts = 0

    # Initialize progress bar
    pbar = tqdm(total=num_walks, desc="Generating walks")
    current_walks = 0

    while len(walks) < num_walks:
        if verbose:
            print(
                f"Attempts: {attempts}/{max_attempts}, Walks generated: {len(walks)}/{num_walks}, Total attempts: {total_attempts}"
            )
        start_vertex = random.choice(list(graph.nodes()))
        walk = generate_valid_walk(
            graph, start_vertex, min_length, max_length, rules, max_attempts, verbose
        )

        if walk:
            walks.append(walk)
            attempts = 0
            # Update progress bar
            new_walks = len(walks) - current_walks
            pbar.update(new_walks)
            current_walks = len(walks)
        else:
            attempts += 1
            total_attempts += 1

            if attempts >= max_attempts:
                if verbose:
                    print("Maximum attempts reached. Resetting attempts counter.")
                attempts = 0

    pbar.close()
    if verbose:
        print(f"{len(walks)} walks generated successfully.")
    return walks
