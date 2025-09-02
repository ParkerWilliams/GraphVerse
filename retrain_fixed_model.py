#!/usr/bin/env python3
"""
Retrain the model with fixed walk generation (no edge adding).
This script:
1. Creates a pre-built dense graph
2. Generates training walks using only existing edges
3. Trains a new model
4. Evaluates to verify it works correctly
"""

import os
import sys
import pickle
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import AscenderRule, EvenRule, RepeaterRule
from graphverse.graph.walk import generate_multiple_walks
from graphverse.data.preparation import prepare_training_data
from graphverse.llm.model_enhanced import EnhancedWalkTransformer
from graphverse.llm.training_enhanced import train_model_enhanced
from graphverse.llm.evaluation import evaluate_model


def create_dense_graph(n=1000, target_density=0.4, seed=42):
    """
    Create a pre-built dense graph with sufficient connectivity.
    This ensures walks can be generated without adding edges.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Creating dense graph with {n} nodes...")
    graph = Graph(n)
    
    # Calculate number of edges needed for target density
    total_possible_edges = n * (n - 1) // 2  # For undirected graph
    target_edges = int(total_possible_edges * target_density)
    
    print(f"  Target density: {target_density}")
    print(f"  Target edges: {target_edges:,}")
    
    # Add edges randomly but ensure good connectivity
    edges_added = 0
    
    # First, ensure basic connectivity - create a random spanning tree
    print("  Creating spanning tree for basic connectivity...")
    nodes = list(range(n))
    random.shuffle(nodes)
    
    # Connect nodes in a random tree structure
    for i in range(1, n):
        # Connect node i to a random previous node
        prev_node = random.choice(nodes[:i])
        graph.add_edge(nodes[i], prev_node)
        edges_added += 1
    
    print(f"  Added {edges_added} edges for basic connectivity")
    
    # Add remaining edges randomly to reach target density
    print(f"  Adding random edges to reach target density...")
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
    actual_edges = np.sum(graph.adjacency > 0) // 2  # Divide by 2 for undirected
    actual_density = actual_edges / total_possible_edges
    
    print(f"  Graph created successfully!")
    print(f"  Final edges: {actual_edges:,}")
    print(f"  Final density: {actual_density:.4f}")
    
    # Check connectivity
    if graph.is_connected():
        print("  ✅ Graph is connected")
    else:
        print("  ⚠️  Warning: Graph is not fully connected")
    
    return graph


def setup_rules(graph, config):
    """
    Set up rules for the experiment based on configuration.
    """
    n = graph.n
    
    # Extract rule configuration
    num_ascenders = config.get('num_ascenders', 50)
    num_evens = config.get('num_evens', 100)
    num_repeaters = config.get('num_repeaters', 100)
    
    print(f"\nSetting up rules...")
    
    # Select ascender nodes (middle portion for better connectivity)
    start_idx = n // 4
    end_idx = 3 * n // 4
    ascender_candidates = list(range(start_idx, end_idx))
    random.shuffle(ascender_candidates)
    ascenders = ascender_candidates[:num_ascenders]
    
    # Select even nodes
    even_candidates = [i for i in range(n) if i % 2 == 0 and i not in ascenders]
    random.shuffle(even_candidates)
    evens = even_candidates[:num_evens]
    
    # Select repeater nodes with k-values
    repeater_candidates = [i for i in range(n) if i not in ascenders and i not in evens]
    random.shuffle(repeater_candidates)
    repeater_nodes = repeater_candidates[:num_repeaters]
    
    # Assign k-values based on configuration
    k_values = config.get('repeater_k_values', [8, 14, 18, 24])
    repeaters = {}
    for i, node in enumerate(repeater_nodes):
        # Distribute k-values evenly across repeater nodes
        k_idx = i % len(k_values)
        repeaters[node] = k_values[k_idx]
    
    # Store in graph for later reference
    graph.repeater_cycles = repeaters
    
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
    
    # Store rules in graph node attributes for metadata
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


def generate_training_data(graph, rules, config):
    """
    Generate training data using the fixed walk generation (no edge adding).
    """
    num_walks = config['num_walks']
    min_length = config['min_walk_length']
    max_length = config['max_walk_length']
    context_window = config['context_window_size']
    
    print(f"\nGenerating training data...")
    print(f"  Number of walks: {num_walks:,}")
    print(f"  Walk lengths: {min_length}-{max_length}")
    print(f"  Context window: {context_window}")
    
    # Test that walks can be generated
    print("\n  Testing walk generation...")
    test_walks = []
    test_attempts = 10
    for _ in range(test_attempts):
        start_node = random.randint(0, graph.n - 1)
        from graphverse.graph.walk import generate_valid_walk
        walk = generate_valid_walk(
            graph, start_node, min_length, max_length, rules,
            max_attempts=10, verbose=False
        )
        if walk:
            test_walks.append(walk)
    
    if len(test_walks) < test_attempts // 2:
        print(f"  ⚠️  Warning: Only {len(test_walks)}/{test_attempts} test walks succeeded")
        print("  Graph may need higher density for reliable walk generation")
    else:
        print(f"  ✅ Test walks successful: {len(test_walks)}/{test_attempts}")
    
    # Generate full training data
    print(f"\n  Generating {num_walks:,} training walks...")
    
    # Use prepare_training_data which handles vocabulary creation and data formatting
    training_data, vocab, corpus_metadata = prepare_training_data(
        graph, num_walks, min_length, max_length, rules, 
        verbose=True
    )
    
    # The prepare_training_data returns padded sequences based on context + prediction windows
    # We need to ensure it matches our expected context window
    
    print(f"\n  Training data generated!")
    print(f"  Vocabulary size: {len(vocab.token2idx)}")
    print(f"  Training tensor shape: {training_data.shape}")
    
    return training_data, vocab, corpus_metadata


def train_new_model(training_data, vocab, config, device='cpu'):
    """
    Train a new model on the fixed training data.
    """
    print(f"\nTraining model...")
    print(f"  Device: {device}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    
    # Model configuration
    hidden_size = 384  # Match the original model
    num_layers = 4
    num_heads = 6
    
    print(f"  Model config: hidden={hidden_size}, layers={num_layers}, heads={num_heads}")
    
    # Train the model using train_model_enhanced which creates the model internally
    trained_model = train_model_enhanced(
        training_data=training_data,
        vocab=vocab,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
        batch_size=config['batch_size'],
        num_epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        context_window_size=config['context_window_size'],
        verbose=True
    )
    
    print("  ✅ Model training complete!")
    
    return trained_model


def evaluate_fixed_model(model, graph, rules, vocab, num_test_walks=100):
    """
    Evaluate the retrained model to verify it works correctly.
    """
    print(f"\nEvaluating retrained model...")
    print(f"  Test walks: {num_test_walks}")
    
    device = next(model.parameters()).device
    
    # Run evaluation
    results, error_summary, _, _, _, _, _ = evaluate_model(
        model=model,
        graph=graph,
        vocab=vocab,
        num_walks=num_test_walks,
        min_start_length=32,
        max_start_length=32,
        rules=rules,
        verbose=True,
        track_token_details=False,
        trajectory_sampling_config=None,
        config=None,
        fast_mode=True
    )
    
    print(f"\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"  Repeater error rate: {error_summary['repeater_error_rate']:.2%}")
    print(f"  Ascender error rate: {error_summary['ascender_error_rate']:.2%}")
    print(f"  Even error rate: {error_summary['even_error_rate']:.2%}")
    print(f"  Broken graph rate: {error_summary.get('broken_graph_rate', 0):.2%}")
    print(f"  Avg steps per walk: {error_summary['avg_steps_per_walk']:.1f}")
    
    # Check if the model is properly following graph edges
    if error_summary.get('broken_graph_rate', 0) < 0.1:  # Less than 10% broken
        print("\n  ✅ SUCCESS: Model is following graph edges correctly!")
    else:
        print("\n  ⚠️  Warning: Model still has high broken graph rate")
    
    return error_summary


def save_experiment(graph, rules, vocab, model, config, output_dir):
    """
    Save all experiment artifacts.
    """
    print(f"\nSaving experiment to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    
    # Save graph
    with open(os.path.join(output_dir, "data", "graph.pkl"), "wb") as f:
        pickle.dump(graph, f)
    
    # Save vocabulary
    with open(os.path.join(output_dir, "data", "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    
    # Save configuration
    import json
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("  ✅ Experiment saved successfully!")
    
    return output_dir


def main():
    """
    Main retraining pipeline.
    """
    print("="*60)
    print("RETRAINING MODEL WITH FIXED WALK GENERATION")
    print("="*60)
    
    # Configuration matching medium scale experiment
    config = {
        'n': 1000,
        'num_walks': 100000,
        'context_window_size': 16,
        'min_walk_length': 32,
        'max_walk_length': 32,
        'num_ascenders': 50,
        'num_evens': 100,
        'num_repeaters': 100,
        'repeater_k_values': [8, 14, 18, 24],
        'epochs': 15,
        'batch_size': 64,
        'learning_rate': 0.001,
        'edge_density': 0.4,
        'seed': 42
    }
    
    # Set random seeds
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Create pre-built dense graph
    graph = create_dense_graph(
        n=config['n'],
        target_density=config['edge_density'],
        seed=config['seed']
    )
    
    # Step 2: Set up rules
    rules, rule_nodes = setup_rules(graph, config)
    
    # Step 3: Generate training data (without adding edges!)
    training_data, vocab, corpus_metadata = generate_training_data(graph, rules, config)
    
    # Step 4: Train new model
    model = train_new_model(training_data, vocab, config, device)
    
    # Step 5: Evaluate to verify it works
    error_summary = evaluate_fixed_model(model, graph, rules, vocab, num_test_walks=100)
    
    # Step 6: Save everything
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiments/fixed_run_{timestamp}"
    save_experiment(graph, rules, vocab, model, config, output_dir)
    
    print(f"\n" + "="*60)
    print("RETRAINING COMPLETE!")
    print("="*60)
    print(f"  Output directory: {output_dir}")
    print(f"  Final broken graph rate: {error_summary.get('broken_graph_rate', 0):.2%}")
    print(f"  Final avg steps per walk: {error_summary['avg_steps_per_walk']:.1f}")
    
    if error_summary.get('broken_graph_rate', 0) < 0.1:
        print("\n🎉 SUCCESS: Model has been successfully retrained!")
        print("   The model now correctly follows graph edges without adding new ones.")
    else:
        print("\n⚠️  The model may need more training or a denser graph.")


if __name__ == "__main__":
    main()