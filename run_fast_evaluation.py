#!/usr/bin/env python3
"""
Fast evaluation script for testing the fixed fast_mode.
"""

import os
import sys
import pickle
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import AscenderRule, EvenRule, RepeaterRule
from graphverse.llm.model_enhanced import EnhancedWalkTransformer
from graphverse.llm.evaluation import evaluate_model


def main():
    """Run fast evaluation on the most recent model."""
    
    # Find most recent experiment
    experiment_dir = "experiments/run_20250831_121356"
    
    print(f"Loading from: {experiment_dir}")
    
    # Load graph
    print("Loading graph...")
    with open(os.path.join(experiment_dir, "data", "graph.pkl"), "rb") as f:
        graph = pickle.load(f)
    print(f"  Graph: {graph.n} nodes")
    
    # Load vocabulary
    print("Loading vocabulary...")
    with open(os.path.join(experiment_dir, "data", "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    print(f"  Vocab size: {len(vocab.token2idx)}")
    
    # Load metadata to get rules
    print("Loading metadata...")
    with open(os.path.join(experiment_dir, "data", "experiment_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    
    # Extract rules from graph metadata
    rule_comp = metadata.graph.rule_composition['rule_types']
    
    ascenders = rule_comp['ascender']['nodes']
    evens = rule_comp['even']['nodes']
    repeater_nodes = rule_comp['repeater']['nodes']
    repeater_k_values = rule_comp['repeater']['k_values']
    
    # Create rule objects
    rules = []
    if ascenders:
        rules.append(AscenderRule(ascenders))
    if evens:
        rules.append(EvenRule(evens))
    if repeater_k_values:
        rules.append(RepeaterRule(repeater_k_values))
    
    print(f"  Rules: ascenders={len(ascenders)}, evens={len(evens)}, repeaters={len(repeater_k_values)}")
    
    # Load model
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint to get dimensions
    checkpoint = torch.load(
        os.path.join(experiment_dir, "model.pth"),
        map_location=device
    )
    
    # Infer dimensions
    vocab_size = checkpoint['embedding.weight'].shape[0]
    hidden_size = checkpoint['embedding.weight'].shape[1]
    
    # Count layers
    num_layers = max([int(k.split('.')[1]) for k in checkpoint.keys() 
                     if k.startswith('layers.') and '.' in k], default=-1) + 1
    
    # Determine num_heads
    if hidden_size == 384:
        num_heads = 6
    else:
        num_heads = 8
    
    # Get max_seq_len from pos_encoding
    if 'pos_encoding.pe' in checkpoint:
        max_seq_len = checkpoint['pos_encoding.pe'].shape[1]
    else:
        max_seq_len = 57  # Default from error
    
    print(f"  Model: vocab={vocab_size}, hidden={hidden_size}, layers={num_layers}, heads={num_heads}, max_seq={max_seq_len}")
    
    model = EnhancedWalkTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
        max_seq_len=max_seq_len
    )
    
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f"  Device: {device}")
    
    # Run evaluation with explicit fast_mode
    print("\n" + "="*60)
    print("RUNNING FAST EVALUATION")
    print("="*60)
    
    # Test with small number of walks first
    num_test_walks = 10
    
    print(f"\nTesting with {num_test_walks} walks...")
    
    results, error_summary, kl_series, token_data, _, _, _ = evaluate_model(
        model=model,
        graph=graph,
        vocab=vocab,
        num_walks=num_test_walks,
        min_start_length=32,
        max_start_length=32,
        rules=rules,
        verbose=True,
        track_token_details=True,  # Will be overridden by fast_mode
        trajectory_sampling_config=None,
        config=None,
        fast_mode=True  # EXPLICITLY ENABLE FAST MODE
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"  Repeater error rate: {error_summary['repeater_error_rate']:.2%}")
    print(f"  Ascender error rate: {error_summary['ascender_error_rate']:.2%}")
    print(f"  Even error rate: {error_summary['even_error_rate']:.2%}")
    print(f"  Avg steps per walk: {error_summary['avg_steps_per_walk']:.1f}")
    print("="*60)


if __name__ == "__main__":
    main()