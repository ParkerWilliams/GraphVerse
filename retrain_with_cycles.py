#!/usr/bin/env python3
"""
Improved retraining script that pre-builds repeater cycles in the graph.
This ensures the model can learn repeater patterns while maintaining edge-following behavior.
"""

import os
import sys
import pickle
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import AscenderRule, EvenRule, RepeaterRule
from graphverse.graph.walk import generate_multiple_walks, generate_valid_walk
from graphverse.data.preparation import prepare_training_data
from graphverse.llm.model_enhanced import EnhancedWalkTransformer
from graphverse.llm.training_enhanced import train_model_enhanced
from graphverse.llm.evaluation import evaluate_model


def create_graph_with_repeater_cycles(n=1000, target_density=0.4, repeater_config=None, seed=42):
    """
    Create a dense graph with pre-built repeater cycles.
    This ensures repeater nodes have valid k-cycles in the graph structure.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Creating dense graph with {n} nodes and pre-built repeater cycles...")
    graph = Graph(n)
    
    # Calculate number of edges needed for target density
    total_possible_edges = n * (n - 1) // 2
    target_edges = int(total_possible_edges * target_density)
    
    print(f"  Target density: {target_density}")
    print(f"  Target edges: {target_edges:,}")
    
    edges_added = 0
    
    # Step 1: Create basic connectivity with spanning tree
    print("  Creating spanning tree for basic connectivity...")
    nodes = list(range(n))
    random.shuffle(nodes)
    
    for i in range(1, n):
        prev_node = random.choice(nodes[:i])
        graph.add_edge(nodes[i], prev_node)
        edges_added += 1
    
    print(f"  Added {edges_added} edges for basic connectivity")
    
    # Step 2: Pre-build repeater cycles if config provided
    if repeater_config:
        repeater_nodes = repeater_config['nodes']
        k_values = repeater_config['k_values']
        
        print(f"\n  Building repeater cycles for {len(repeater_nodes)} nodes...")
        cycles_built = 0
        
        for node, k in zip(repeater_nodes, [k_values[i % len(k_values)] for i in range(len(repeater_nodes))]):
            # Try to build a k-cycle for this repeater node
            cycle_built = False
            attempts = 0
            max_attempts = 20
            
            while not cycle_built and attempts < max_attempts:
                attempts += 1
                
                # Start from the repeater node and try to build a path
                cycle = [node]
                current = node
                
                # Build a path of length k
                for step in range(k):
                    # Get all possible next nodes (not in current path, not repeater nodes)
                    candidates = [n for n in range(graph.n) 
                                if n not in cycle and n not in repeater_nodes]
                    
                    if not candidates:
                        break  # Can't continue
                    
                    # Pick a random candidate
                    next_node = random.choice(candidates)
                    
                    # Add edge if it doesn't exist
                    if not graph.has_edge(current, next_node):
                        graph.add_edge(current, next_node)
                        edges_added += 1
                    
                    cycle.append(next_node)
                    current = next_node
                
                # Complete the cycle back to the repeater node
                if len(cycle) == k + 1:
                    if not graph.has_edge(current, node):
                        graph.add_edge(current, node)
                        edges_added += 1
                    cycle_built = True
                    cycles_built += 1
                    
                    # Store the cycle in the graph
                    if not hasattr(graph, 'repeater_cycles'):
                        graph.repeater_cycles = {}
                    graph.repeater_cycles[node] = k
                    
                    if cycles_built % 20 == 0:
                        print(f"    Built {cycles_built}/{len(repeater_nodes)} cycles...")
        
        print(f"  ✅ Successfully built {cycles_built}/{len(repeater_nodes)} repeater cycles")
    
    # Step 3: Add random edges to reach target density
    print(f"\n  Adding random edges to reach target density...")
    attempts = 0
    max_attempts = target_edges * 10
    
    while edges_added < target_edges and attempts < max_attempts:
        v1 = random.randint(0, n-1)
        v2 = random.randint(0, n-1)
        attempts += 1
        
        if v1 != v2 and not graph.has_edge(v1, v2):
            graph.add_edge(v1, v2)
            edges_added += 1
            
            if edges_added % 10000 == 0:
                print(f"    Added {edges_added:,}/{target_edges:,} edges...")
    
    # Verify final density
    actual_edges = np.sum(graph.adjacency > 0) // 2
    actual_density = actual_edges / total_possible_edges
    
    print(f"\n  Graph created successfully!")
    print(f"  Final edges: {actual_edges:,}")
    print(f"  Final density: {actual_density:.4f}")
    
    # Check connectivity
    if graph.is_connected():
        print("  ✅ Graph is connected")
    else:
        print("  ⚠️  Warning: Graph is not fully connected")
    
    return graph


def setup_rules_with_cycles(graph, config):
    """
    Set up rules and ensure repeater nodes have their cycles.
    """
    n = graph.n
    
    # Extract rule configuration
    num_ascenders = config.get('num_ascenders', 50)
    num_evens = config.get('num_evens', 100)
    num_repeaters = config.get('num_repeaters', 100)
    k_values = config.get('repeater_k_values', [8, 14, 18, 24])
    
    print(f"\nSetting up rules...")
    
    # Select ascender nodes
    start_idx = n // 4
    end_idx = 3 * n // 4
    ascender_candidates = list(range(start_idx, end_idx))
    random.shuffle(ascender_candidates)
    ascenders = ascender_candidates[:num_ascenders]
    
    # Select even nodes
    even_candidates = [i for i in range(n) if i % 2 == 0 and i not in ascenders]
    random.shuffle(even_candidates)
    evens = even_candidates[:num_evens]
    
    # Select repeater nodes
    repeater_candidates = [i for i in range(n) if i not in ascenders and i not in evens]
    random.shuffle(repeater_candidates)
    repeater_nodes = repeater_candidates[:num_repeaters]
    
    # Assign k-values to repeaters
    repeaters = {}
    for i, node in enumerate(repeater_nodes):
        k_idx = i % len(k_values)
        repeaters[node] = k_values[k_idx]
    
    # Update graph's repeater_cycles if not already set
    if not hasattr(graph, 'repeater_cycles'):
        graph.repeater_cycles = {}
    graph.repeater_cycles.update(repeaters)
    
    # Create rule objects
    rules = []
    if ascenders:
        rules.append(AscenderRule(ascenders))
    if evens:
        rules.append(EvenRule(evens))
    if repeaters:
        rules.append(RepeaterRule(repeaters))
    
    print(f"  Ascenders: {len(ascenders)} nodes")
    print(f"  Evens: {len(evens)} nodes")
    print(f"  Repeaters: {len(repeaters)} nodes")
    print(f"  K-values distribution: {k_values}")
    
    # Store rules in graph node attributes
    graph.node_attributes = {}
    for node in ascenders:
        graph.node_attributes[node] = {'rule': 'ascender'}
    for node in evens:
        graph.node_attributes[node] = {'rule': 'even'}
    for node, k in repeaters.items():
        graph.node_attributes[node] = {'rule': 'repeater', 'k': k}
    
    return rules, {
        'ascenders': ascenders,
        'evens': evens,
        'repeaters': repeaters
    }


def generate_hybrid_training_data(graph, rules, config, rule_nodes):
    """
    Generate hybrid training data:
    - Most walks follow strict edge constraints
    - Some walks demonstrate repeater cycles
    - Balanced to teach both behaviors
    """
    num_walks = config['num_walks']
    min_length = config['min_walk_length']
    max_length = config['max_walk_length']
    
    print(f"\nGenerating hybrid training data...")
    print(f"  Total walks: {num_walks:,}")
    print(f"  Walk lengths: {min_length}-{max_length}")
    
    # Split walks: 70% regular, 30% repeater-focused
    num_regular = int(num_walks * 0.7)
    num_repeater = num_walks - num_regular
    
    print(f"  Regular walks: {num_regular:,}")
    print(f"  Repeater-focused walks: {num_repeater:,}")
    
    all_walks = []
    
    # Generate regular walks (strict edge-following)
    print("\n  Generating regular walks...")
    for i in range(num_regular):
        start_node = random.randint(0, graph.n - 1)
        walk = generate_valid_walk(
            graph, start_node, min_length, max_length, rules,
            max_attempts=10, verbose=False
        )
        if walk:
            all_walks.append(walk)
        
        if (i + 1) % 10000 == 0:
            print(f"    Generated {i + 1:,}/{num_regular:,} regular walks...")
    
    print(f"  ✅ Generated {len(all_walks)} regular walks")
    
    # Generate repeater-focused walks
    print("\n  Generating repeater-focused walks...")
    repeater_nodes = list(rule_nodes['repeaters'].keys())
    repeater_walks_count = 0
    
    for i in range(num_repeater):
        # Start from or near a repeater node
        if random.random() < 0.5 and repeater_nodes:
            # Start directly from a repeater
            start_node = random.choice(repeater_nodes)
        else:
            # Start from a neighbor of a repeater
            repeater = random.choice(repeater_nodes) if repeater_nodes else 0
            neighbors = graph.get_neighbors(repeater)
            start_node = random.choice(neighbors) if neighbors else repeater
        
        walk = generate_valid_walk(
            graph, start_node, min_length, max_length, rules,
            max_attempts=10, verbose=False
        )
        
        if walk:
            all_walks.append(walk)
            # Check if walk contains a repeater pattern
            if any(node in repeater_nodes for node in walk):
                repeater_walks_count += 1
        
        if (i + 1) % 5000 == 0:
            print(f"    Generated {i + 1:,}/{num_repeater:,} repeater-focused walks...")
    
    print(f"  ✅ Generated {repeater_walks_count} walks with repeater patterns")
    
    # Add some guaranteed repeater demonstrations
    print("\n  Adding synthetic repeater demonstrations...")
    num_demos = min(1000, len(repeater_nodes) * 10)
    
    for _ in range(num_demos):
        repeater = random.choice(repeater_nodes)
        k = rule_nodes['repeaters'][repeater]
        
        # Build a walk that demonstrates the repeater cycle
        demo_walk = [repeater]
        current = repeater
        
        # Try to follow the pre-built cycle
        for step in range(k):
            neighbors = graph.get_neighbors(current)
            valid_next = [n for n in neighbors if n not in demo_walk]
            if valid_next:
                next_node = random.choice(valid_next)
                demo_walk.append(next_node)
                current = next_node
            else:
                break
        
        # Complete the cycle if possible
        if len(demo_walk) == k + 1 and graph.has_edge(current, repeater):
            demo_walk.append(repeater)
            # Extend the walk a bit more
            for _ in range(random.randint(5, 10)):
                neighbors = graph.get_neighbors(current)
                if neighbors:
                    current = random.choice(neighbors)
                    demo_walk.append(current)
                else:
                    break
            
            all_walks.append(demo_walk)
    
    print(f"  ✅ Total walks generated: {len(all_walks)}")
    
    # Shuffle walks
    random.shuffle(all_walks)
    
    # Create vocabulary and prepare training data
    print("\n  Preparing training sequences...")
    from graphverse.data.preparation import WalkVocabulary
    
    vocab = WalkVocabulary(all_walks)
    
    # Convert walks to training sequences
    sequences = []
    context_window = config['context_window_size']
    
    for walk in all_walks:
        # Add START and END tokens
        walk_tokens = ['<START>'] + [str(v) for v in walk] + ['<END>']
        walk_indices = [vocab.token2idx[token] for token in walk_tokens]
        
        # Create training sequences with context windows
        for i in range(1, len(walk_indices)):
            start_idx = max(0, i - context_window)
            context = walk_indices[start_idx:i]
            target = walk_indices[i]
            
            # Pad context if needed
            if len(context) < context_window:
                padding = [vocab.token2idx['<PAD>']] * (context_window - len(context))
                context = padding + context
            
            sequences.append(context + [target])
    
    # Convert to tensor
    training_data = torch.tensor(sequences, dtype=torch.long)
    
    print(f"  Training tensor shape: {training_data.shape}")
    print(f"  Vocabulary size: {len(vocab.token2idx)}")
    
    # Create metadata
    from graphverse.analysis.metadata import TrainingCorpusMetadata
    metadata = TrainingCorpusMetadata(all_walks, graph)
    
    return training_data, vocab, metadata


def main():
    """Run improved retraining with pre-built repeater cycles."""
    
    print("="*70)
    print("IMPROVED MODEL RETRAINING WITH PRE-BUILT REPEATER CYCLES")
    print("="*70)
    
    # Configuration
    config = {
        'num_nodes': 1000,
        'graph_density': 0.4,
        'num_ascenders': 50,
        'num_evens': 100,
        'num_repeaters': 100,
        'repeater_k_values': [8, 14, 18, 24],
        'num_walks': 100000,
        'min_walk_length': 32,
        'max_walk_length': 32,
        'context_window_size': 16,
        'batch_size': 64,
        'epochs': 15,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    }
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/improved_run_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    data_dir = exp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"\nExperiment directory: {exp_dir}")
    
    # Step 1: Create graph with pre-built repeater cycles
    repeater_config = {
        'nodes': list(range(100, 200)),  # Will be updated in setup_rules
        'k_values': config['repeater_k_values']
    }
    
    graph = create_graph_with_repeater_cycles(
        n=config['num_nodes'],
        target_density=config['graph_density'],
        repeater_config=None,  # Will add cycles after rule setup
        seed=42
    )
    
    # Step 2: Set up rules
    rules, rule_nodes = setup_rules_with_cycles(graph, config)
    
    # Step 3: Now rebuild the graph with proper repeater cycles
    print("\nRebuilding graph with repeater cycles for selected nodes...")
    graph = create_graph_with_repeater_cycles(
        n=config['num_nodes'],
        target_density=config['graph_density'],
        repeater_config={
            'nodes': list(rule_nodes['repeaters'].keys()),
            'k_values': config['repeater_k_values']
        },
        seed=42
    )
    
    # Re-apply rules to the new graph
    rules, rule_nodes = setup_rules_with_cycles(graph, config)
    
    # Save graph
    with open(data_dir / "graph.pkl", "wb") as f:
        pickle.dump(graph, f)
    print(f"✅ Graph saved to {data_dir}/graph.pkl")
    
    # Step 4: Generate hybrid training data
    training_data, vocab, metadata = generate_hybrid_training_data(
        graph, rules, config, rule_nodes
    )
    
    # Save vocabulary
    with open(data_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f"✅ Vocabulary saved to {data_dir}/vocab.pkl")
    
    # Step 5: Train model
    print(f"\nTraining model on {config['device']}...")
    
    model = train_model_enhanced(
        training_data=training_data,
        vocab=vocab,
        hidden_size=384,
        num_layers=4,
        num_heads=6,
        dropout=0.1,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        device=config['device'],
        verbose=True
    )
    
    # Save model
    model_path = exp_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Step 6: Quick evaluation
    print("\n" + "="*70)
    print("QUICK EVALUATION")
    print("="*70)
    
    # Test repeater success rate
    print("\nTesting repeater patterns...")
    repeater_nodes = list(rule_nodes['repeaters'].keys())
    successful_repeaters = 0
    total_tests = min(100, len(repeater_nodes))
    
    for _ in range(total_tests):
        repeater = random.choice(repeater_nodes)
        walk = generate_valid_walk(
            graph, repeater, 40, 50, rules,
            max_attempts=5, verbose=False
        )
        if walk and walk.count(repeater) >= 2:
            successful_repeaters += 1
    
    print(f"Repeater success rate: {successful_repeaters}/{total_tests} = {100*successful_repeaters/total_tests:.1f}%")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model and data saved in: {exp_dir}")
    print("\nNext steps:")
    print("1. Run full evaluation to verify performance")
    print("2. Test with analyze_repeater_kl_focused.py")
    print("3. Compare with previous models")
    
    return str(exp_dir)


if __name__ == "__main__":
    exp_dir = main()
    print(f"\nExperiment directory: {exp_dir}")