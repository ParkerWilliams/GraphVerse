#!/usr/bin/env python3
"""
Small-scale retraining test to verify the fix works.
Uses smaller parameters for quick testing.
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

from retrain_fixed_model import (
    create_dense_graph, 
    setup_rules, 
    train_new_model,
    evaluate_fixed_model,
    save_experiment
)
from graphverse.data.preparation import prepare_training_data


def main():
    """
    Small-scale retraining for testing.
    """
    print("="*60)
    print("SMALL-SCALE RETRAINING TEST")
    print("="*60)
    
    # Small configuration for quick testing
    config = {
        'n': 200,  # Smaller graph
        'num_walks': 5000,  # Much fewer walks
        'context_window_size': 16,
        'min_walk_length': 32,
        'max_walk_length': 32,
        'num_ascenders': 10,
        'num_evens': 20,
        'num_repeaters': 20,
        'repeater_k_values': [8, 14, 18, 24],
        'epochs': 3,  # Fewer epochs
        'batch_size': 32,
        'learning_rate': 0.001,
        'edge_density': 0.5,  # Higher density for better connectivity
        'seed': 42
    }
    
    print("\nConfiguration:")
    print(f"  Graph size: {config['n']} nodes")
    print(f"  Training walks: {config['num_walks']:,}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Edge density: {config['edge_density']}")
    
    # Set random seeds
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Create pre-built dense graph
    print("\n" + "="*40)
    print("STEP 1: Creating Graph")
    print("="*40)
    graph = create_dense_graph(
        n=config['n'],
        target_density=config['edge_density'],
        seed=config['seed']
    )
    
    # Store initial edge count to verify no edges are added
    initial_edges = np.sum(graph.adjacency > 0) // 2
    
    # Step 2: Set up rules
    print("\n" + "="*40)
    print("STEP 2: Setting Up Rules")
    print("="*40)
    rules, rule_nodes = setup_rules(graph, config)
    
    # Step 3: Generate training data
    print("\n" + "="*40)
    print("STEP 3: Generating Training Data")
    print("="*40)
    
    print(f"  Generating {config['num_walks']:,} walks...")
    training_data, vocab, corpus_metadata = prepare_training_data(
        graph, 
        config['num_walks'], 
        config['min_walk_length'], 
        config['max_walk_length'], 
        rules,
        verbose=True
    )
    
    # Verify no edges were added during training data generation
    final_edges = np.sum(graph.adjacency > 0) // 2
    edges_added = final_edges - initial_edges
    
    print(f"\n  Training data shape: {training_data.shape}")
    print(f"  Vocabulary size: {len(vocab.token2idx)}")
    print(f"  Edges added during generation: {edges_added}")
    
    if edges_added > 0:
        print("  ❌ ERROR: Edges were added during walk generation!")
        return
    else:
        print("  ✅ SUCCESS: No edges added during walk generation")
    
    # Step 4: Train model
    print("\n" + "="*40)
    print("STEP 4: Training Model")
    print("="*40)
    
    model = train_new_model(training_data, vocab, config, device)
    
    # Step 5: Evaluate
    print("\n" + "="*40)
    print("STEP 5: Evaluating Model")
    print("="*40)
    
    error_summary = evaluate_fixed_model(model, graph, rules, vocab, num_test_walks=50)
    
    # Step 6: Save if successful
    if error_summary.get('broken_graph_rate', 1.0) < 0.2:  # Less than 20% broken
        print("\n" + "="*40)
        print("STEP 6: Saving Experiment")
        print("="*40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"experiments/small_test_{timestamp}"
        save_experiment(graph, rules, vocab, model, config, output_dir)
        
        print(f"\n🎉 SUCCESS: Small-scale test passed!")
        print(f"   Output: {output_dir}")
        print(f"   Broken graph rate: {error_summary.get('broken_graph_rate', 0):.2%}")
    else:
        print(f"\n⚠️  Test showed high broken graph rate: {error_summary.get('broken_graph_rate', 1.0):.2%}")
        print("   This may be normal for a small test - the full training should work better")


if __name__ == "__main__":
    main()