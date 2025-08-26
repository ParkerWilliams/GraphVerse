import random
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .walk import generate_multiple_walks
from .rules import RepeaterRule, AscenderRule, EvenRule
from .base import Graph


def save_walks_to_files(walks, output_dir, max_file_size=50*1024*1024, verbose=False):
    """
    Save walks to multiple files, ensuring each file is under max_file_size (in bytes).
    
    Args:
        walks: List of walks to save
        output_dir: Directory to save the files
        max_file_size: Maximum file size in bytes (default 50MB)
        verbose: Whether to print progress
    """
    # Convert numpy integers to Python integers
    walks = [[int(node) for node in walk] for walk in walks]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    walks_per_file = len(walks) // 10  # Start with rough estimate of 10 files
    current_file = 0
    start_idx = 0
    
    while start_idx < len(walks):
        # Try to write a chunk of walks
        test_walks = walks[start_idx:start_idx + walks_per_file]
        test_json = json.dumps(test_walks)
        
        # If file would be too large, reduce walks_per_file
        if len(test_json.encode('utf-8')) > max_file_size:
            walks_per_file = walks_per_file // 2
            continue
            
        # Save the walks
        output_file = Path(output_dir) / f"walks_{current_file}.json"
        with open(output_file, 'w') as f:
            json.dump(test_walks, f)
            
        if verbose:
            print(f"Saved {len(test_walks)} walks to {output_file}")
            
        start_idx += walks_per_file
        current_file += 1


def generate_random_graph(
    n, rules, num_walks=1000, min_walk_length=5, max_walk_length=20, 
    verbose=False, save_walks=False, output_dir="walks",
    min_edge_density=0.4,  # Minimum edge density for connectivity
    exponential_scale=1.2  # Scale parameter for edge weight distribution
):
    if verbose:
        print("\n  Starting graph generation...")
        print(f"  Target: {n} nodes with minimum edge density of {min_edge_density}")

    # Create graph with adjacency matrix
    G = Graph(n)
    
    def print_density_stats(stage):
        if verbose:
            density = calculate_edge_density(G, verbose=False)
            edges = G.edge_count  # Use graph's edge counter
            max_edges = (n * (n - 1)) // 2  # Max undirected edges
            print(f"\n{stage}:")
            print(f"  Edge density: {density:.3f}")
            print(f"  Edges: {edges}/{max_edges}")

    # Initial state
    print_density_stats("Initial graph state")

    # Find the correct rule instances
    repeater_rule = next(rule for rule in rules if isinstance(rule, RepeaterRule))
    ascender_rule = next(rule for rule in rules if isinstance(rule, AscenderRule))
    even_rule = next(rule for rule in rules if isinstance(rule, EvenRule))
    
    # Assign rules to nodes
    if verbose:
        print(f"\n  Assigning rules to nodes:")
        print(f"    Ascender nodes: {len(ascender_rule.member_nodes)}")
        print(f"    Even rule nodes: {len(even_rule.member_nodes)}")
        print(f"    Repeater nodes: {len(repeater_rule.member_nodes)}")
    
    iterator = tqdm(range(n), desc="Assigning rules to nodes") if verbose else range(n)
    for node in iterator:
        G.node_attributes[node] = {"rule": "none"}
        
        if node in ascender_rule.member_nodes:
            G.node_attributes[node]["rule"] = "ascender"
        elif node in even_rule.member_nodes:
            G.node_attributes[node]["rule"] = "even"
        elif node in repeater_rule.member_nodes:
            G.node_attributes[node]["rule"] = "repeater"
            G.node_attributes[node]["repetitions"] = repeater_rule.members_nodes_dict[node]

    # First ensure each rule type has valid paths
    if verbose:
        print("\n  Setting up rule-specific paths...")
    
    # 1. Handle repeater nodes to ensure valid loop paths exist
    repeater_iterator = tqdm(range(n), desc="Setting up repeater paths") if verbose else range(n)
    for node in repeater_iterator:
        if G.node_attributes[node]["rule"] == "repeater":
            repetitions = G.node_attributes[node]["repetitions"]
            
            # Find valid intermediate nodes (not even-rule or ascender nodes)
            valid_nodes = [
                v for v in range(n)
                if v != node
                and G.node_attributes[v]["rule"] not in ["even", "ascender"]
                and v not in repeater_rule.member_nodes  # Avoid other repeaters
            ]
            
            if len(valid_nodes) < repetitions:
                raise ValueError(f"Not enough valid nodes for repeater {node} with {repetitions} repetitions")
            
            # Create multiple distinct k-cycles for this repeater (2-4 cycles)
            num_cycles = random.randint(2, 4)
            
            for cycle_idx in range(num_cycles):
                # Create a path: repeater -> intermediate1 -> intermediate2 -> ... -> back to repeater
                # This ensures the repeater can loop back to itself in exactly k steps
                path_nodes = random.sample(valid_nodes, k=repetitions)
                
                # Build the complete k-cycle path
                k_cycle = [node] + path_nodes + [node]  # Full cycle including return
                
                # Create the loop: node -> path_nodes[0] -> path_nodes[1] -> ... -> node
                current = node
                for i, path_node in enumerate(path_nodes):
                    G.add_edge(current, path_node)
                    current = path_node
                
                # Close the loop: last path node back to repeater
                G.add_edge(current, node)
                
                # Store this k-cycle path for efficient walk generation
                G.add_repeater_cycle(node, k_cycle)
                
                if verbose and cycle_idx == 0:
                    print(f"  Created {num_cycles} k-cycles for repeater {node} (k={repetitions})")
                
                # Remove used nodes to ensure cycles are distinct
                for used_node in path_nodes:
                    if used_node in valid_nodes:
                        valid_nodes.remove(used_node)
    
    print_density_stats("After setting up repeater paths")

    # 2. Handle ascender nodes to ensure valid ascending paths
    ascender_iterator = tqdm(range(n), desc="Setting up ascender paths") if verbose else range(n)
    for node in ascender_iterator:
        if G.node_attributes[node]["rule"] == "ascender":
            # Ensure at least one valid ascending path of length 3-5
            path_length = random.randint(3, 5)
            
            # Find valid nodes that are higher than current node
            higher_nodes = [
                v for v in range(n)
                if v > node
                and G.node_attributes[v]["rule"] != "repeater"  # Avoid repeaters as they have fixed paths
            ]
            
            if len(higher_nodes) < path_length - 1:  # -1 because we start with current node
                raise ValueError(f"Not enough higher nodes for ascender {node}")
            
            # Sort by value to ensure ascending order
            higher_nodes.sort()
            
            # Select path_length-1 nodes that will form ascending path
            path_nodes = []
            last_node = node
            for _ in range(path_length - 1):
                valid_nodes = [v for v in higher_nodes if v > last_node]
                if not valid_nodes:
                    break
                next_node = random.choice(valid_nodes)
                path_nodes.append(next_node)
                last_node = next_node
                higher_nodes.remove(next_node)
            
            # Add edges to create the ascending path
            current = node
            for next_node in path_nodes:
                G.add_edge(current, next_node)
                current = next_node

    print_density_stats("After setting up ascender paths")

    # 3. Handle even nodes to ensure valid even-only paths
    even_iterator = tqdm(range(n), desc="Setting up even paths") if verbose else range(n)
    for node in even_iterator:
        if G.node_attributes[node]["rule"] == "even":
            # Ensure at least one valid path through even numbers
            path_length = random.randint(3, 5)
            
            # Find valid even nodes
            even_nodes = [
                v for v in range(n)
                if v % 2 == 0
                and v != node
                and G.node_attributes[v]["rule"] != "repeater"  # Avoid repeaters
            ]
            
            if len(even_nodes) < path_length - 1:
                raise ValueError(f"Not enough even nodes for even node {node}")
            
            # Select path_length-1 even nodes
            path_nodes = random.sample(even_nodes, k=path_length - 1)
            
            # Add edges to create the even path
            current = node
            for next_node in path_nodes:
                G.add_edge(current, next_node)
                current = next_node

    print_density_stats("After setting up even paths")

    # Now add additional edges for variety while respecting rules
    if verbose:
        print("\n  Adding additional edges for graph connectivity...")
    edge_iterator = tqdm(range(n), desc="Adding additional edges") if verbose else range(n)
    for node in edge_iterator:
        rule = G.node_attributes[node]["rule"]
        
        if rule == "ascender":
            candidates = [v for v in range(n) if v > node]
            if candidates:
                num_edges = random.randint(1, min(len(candidates), 5))  # Limit max edges
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)
        elif rule == "even":
            candidates = [v for v in range(n) if v % 2 == 0]
            if candidates:
                num_edges = random.randint(1, min(len(candidates), 5))  # Limit max edges
                for v in random.sample(candidates, num_edges):
                    G.add_edge(node, v)

    print_density_stats("After adding additional rule-based edges")

    # Ensure minimum edge density is met
    current_density = calculate_edge_density(G, verbose=False)
    if current_density < min_edge_density:
        if verbose:
            print(f"\nCurrent edge density {current_density:.3f} below minimum {min_edge_density}")
            print("Adding additional edges to meet minimum density...")
        
        # Calculate how many edges we need to add
        # For undirected graphs represented with bidirectional edges:
        # - We only count positive edges (one direction)
        # - Max possible edges = n*(n-1)/2 for undirected graph
        num_nodes = G.n
        max_possible_edges = (num_nodes * (num_nodes - 1)) // 2  # Undirected graph
        target_edges = int(min_edge_density * max_possible_edges) + (num_nodes // 10)  # Add adaptive buffer
        current_edges = G.edge_count  # Use graph's edge counter
        edges_to_add = target_edges - current_edges
        
        if verbose:
            print(f"Target edges: {target_edges} ({min_edge_density:.1%} of {max_possible_edges})")
            print(f"Current edges: {current_edges}")
            print(f"Edges to add: {edges_to_add}")
        
        # Add edges between non-rule nodes first
        non_rule_nodes = [n for n in range(n) if G.node_attributes[n]['rule'] == 'none']
        edges_added = 0
        
        # Try to add edges while respecting rules
        attempts = 0
        max_attempts = edges_to_add * 2  # Allow some failed attempts
        
        with tqdm(total=edges_to_add, desc="Adding density edges") as pbar:
            # Phase 1: Respect rule constraints as much as possible
            phase1_target = min(edges_to_add, int(edges_to_add * 0.2))  # Try 20% with rules first
            phase1_attempts = phase1_target * 3
            
            while edges_added < phase1_target and attempts < phase1_attempts:
                # Prefer non-rule nodes but occasionally use rule nodes
                if random.random() < 0.8 and non_rule_nodes:
                    source = random.choice(non_rule_nodes)
                else:
                    source = random.randint(0, n-1)
                
                # For rule nodes, respect their constraints
                rule = G.node_attributes[source]['rule']
                if rule == 'ascender':
                    candidates = [v for v in range(n) if v > source]
                elif rule == 'even':
                    candidates = [v for v in range(n) if v % 2 == 0]
                elif rule == 'repeater':
                    # Skip repeater nodes as their paths are already set
                    attempts += 1
                    continue
                else:
                    candidates = list(range(n))
                    candidates.remove(source)
                
                # Remove existing neighbors
                existing_neighbors = set(np.where(G.adjacency[source] > 0)[0])
                candidates = [v for v in candidates if v not in existing_neighbors]
                
                if candidates:
                    target = random.choice(candidates)
                    G.add_edge(source, target)
                    edges_added += 1
                    pbar.update(1)
                
                attempts += 1
            
            # Phase 2: Add random edges to guarantee density target
            if verbose and edges_added < edges_to_add:
                remaining = edges_to_add - edges_added
                print(f"\nPhase 2: Adding {remaining} random edges to reach density target...")
            
            # Pre-compute all possible undirected edges to avoid infinite loops
            possible_edges = []
            for i in range(n):
                for j in range(i+1, n):  # Only consider i < j to avoid duplicates
                    if G.adjacency[i, j] == 0:  # No edge exists yet
                        possible_edges.append((i, j))
            
            # Randomly shuffle the possible edges
            random.shuffle(possible_edges)
            
            # Add edges until we reach the target
            edge_idx = 0
            while edges_added < edges_to_add and edge_idx < len(possible_edges):
                source, target = possible_edges[edge_idx]
                G.add_edge(source, target)
                edges_added += 1
                pbar.update(1)
                edge_idx += 1
        
        if verbose:
            final_density = calculate_edge_density(G, verbose=False)
            print(f"Final edge density: {final_density:.3f}")
            if final_density < min_edge_density:
                print("Warning: Could not reach target density while respecting rules")

    print_density_stats("After density enforcement")

    # Ensure each node has at least one outgoing edge
    for node in range(n):
        if np.sum(G.adjacency[node] > 0) == 0:  # No outgoing edges
            # Randomly select a target node that is not the current node
            target = random.choice([v for v in range(n) if v != node])
            G.add_edge(node, target)

    print_density_stats("After ensuring each node has an outgoing edge")

    # Final connectivity check and repair if needed
    if verbose:
        print("\n  Verifying final connectivity...")
    
    if not G.is_connected():
        if verbose:
            print("  Graph not connected, adding connectivity edges...")
        
        # Find disconnected components and connect them
        visited = set()
        components = []
        
        for start_node in range(n):
            if start_node not in visited:
                # BFS to find this component
                component = set()
                queue = [start_node]
                component.add(start_node)
                visited.add(start_node)
                
                while queue:
                    current = queue.pop(0)
                    neighbors = G.get_neighbors(current)
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
        
        # Connect components by adding edges between them
        for i in range(len(components) - 1):
            node1 = random.choice(list(components[i]))
            node2 = random.choice(list(components[i + 1]))
            G.add_edge(node1, node2)
            if verbose:
                print(f"    Connected component {i} to {i+1} via edge {node1}-{node2}")
    
    if verbose and G.is_connected():
        print("  ✓ Final connectivity verified")

    print_density_stats("After connectivity verification")

    # Assign exponential distribution weights for outbound transitions
    # Each vertex gets exponentially distributed transition probabilities
    # Use the passed exponential_scale parameter for experiment tracking
    
    for node in tqdm(range(n), desc="Assigning exponential edge probabilities"):
        neighbors = G.get_neighbors(node)
        if len(neighbors) > 0:
            # Generate weights from exponential distribution
            weights = np.random.exponential(scale=exponential_scale, size=len(neighbors))
            
            # Normalize to create valid probability distribution
            total_weight = np.sum(weights)
            probabilities = weights / total_weight
            
            for neighbor, prob in zip(neighbors, probabilities):
                # Update both directions with same positive weight (undirected graph)
                G.adjacency[node, neighbor] = prob
                G.adjacency[neighbor, node] = prob

    # Calculate and report edge density statistics
    if verbose:
        print("\nFinal Edge Density Statistics:")
        # Overall edge density
        overall_density = calculate_edge_density(G, verbose=False)
        print(f"Overall edge density: {overall_density:.3f}")
        
        # Edge density by rule type
        rule_types = ['ascender', 'even', 'repeater', 'none']
        for rule_type in rule_types:
            # Get nodes of this type
            type_nodes = [n for n in range(G.n) if G.node_attributes[n]['rule'] == rule_type]
            if not type_nodes:
                continue
                
            # Count edges from these nodes
            num_edges = sum(np.sum(G.adjacency[n] > 0) for n in type_nodes)
            max_possible = len(type_nodes) * (G.n - 1)  # Each node could connect to all others
            density = num_edges / max_possible if max_possible > 0 else 0
            
            # Count edges between nodes of this type
            internal_edges = sum(
                np.sum(G.adjacency[n1, n2] > 0)
                for i, n1 in enumerate(type_nodes)
                for n2 in type_nodes[i+1:]
            )
            max_internal = (len(type_nodes) * (len(type_nodes) - 1)) // 2
            internal_density = internal_edges / max_internal if max_internal > 0 else 0
            
            print(f"\n{rule_type.capitalize()} nodes:")
            print(f"  Count: {len(type_nodes)}")
            print(f"  Outgoing edge density: {density:.3f}")
            print(f"  Internal edge density: {internal_density:.3f}")

    # Generate and save walks if requested
    if save_walks:
        if verbose:
            print("\nGenerating walks...")
        walks = generate_multiple_walks(
            G, num_walks, min_walk_length, max_walk_length, 
            rules, verbose=verbose
        )
        if verbose:
            print("\nSaving walks...")
        save_walks_to_files(walks, output_dir, verbose=verbose)

    return G


def calculate_edge_density(graph, verbose=False):
    if verbose:
        print("Calculating edge density...")
    num_nodes = graph.n
    # Use graph's edge counter
    num_edges = graph.edge_count
    max_possible_edges = (num_nodes * (num_nodes - 1)) // 2  # Undirected graph
    edge_density = num_edges / max_possible_edges
    if verbose:
        print(f"Edge density: {edge_density}")
    return edge_density
