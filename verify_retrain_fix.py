#!/usr/bin/env python3
"""
Verify that the retrained model correctly follows graph edges.
Compare with the original model to show the improvement.
"""

import os
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import AscenderRule, EvenRule, RepeaterRule
from graphverse.llm.model_enhanced import EnhancedWalkTransformer
from graphverse.llm.evaluation import evaluate_model

def load_and_evaluate_model(experiment_dir, num_test_walks=50):
    """Load a model and evaluate it."""
    
    print(f"\nLoading from: {experiment_dir}")
    
    # Check if directory exists
    if not os.path.exists(experiment_dir):
        print(f"  ❌ Directory not found: {experiment_dir}")
        return None
    
    # Load graph
    graph_path = os.path.join(experiment_dir, "data", "graph.pkl")
    if os.path.exists(graph_path):
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        print(f"  Graph: {graph.n} nodes")
    else:
        print(f"  ❌ Graph not found at {graph_path}")
        return None
    
    # Load vocabulary
    vocab_path = os.path.join(experiment_dir, "data", "vocab.pkl")
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        print(f"  Vocab size: {len(vocab.token2idx)}")
    else:
        print(f"  ❌ Vocabulary not found at {vocab_path}")
        return None
    
    # Extract rules from graph attributes
    ascenders = []
    evens = []
    repeaters = {}
    
    if hasattr(graph, 'node_attributes'):
        for node, attrs in graph.node_attributes.items():
            if attrs.get('rule') == 'ascender':
                ascenders.append(node)
            elif attrs.get('rule') == 'even':
                evens.append(node)
            elif attrs.get('rule') == 'repeater':
                repeaters[node] = attrs.get('k', 8)
    
    # Create rule objects
    rules = []
    if ascenders:
        rules.append(AscenderRule(ascenders))
    if evens:
        rules.append(EvenRule(evens))
    if repeaters:
        rules.append(RepeaterRule(repeaters))
    
    print(f"  Rules: ascenders={len(ascenders)}, evens={len(evens)}, repeaters={len(repeaters)}")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(experiment_dir, "model.pth")
    
    if not os.path.exists(model_path):
        print(f"  ❌ Model not found at {model_path}")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Infer model dimensions
    vocab_size = checkpoint['embedding.weight'].shape[0]
    hidden_size = checkpoint['embedding.weight'].shape[1]
    
    # Count layers
    num_layers = max([int(k.split('.')[1]) for k in checkpoint.keys() 
                     if k.startswith('layers.') and '.' in k], default=-1) + 1
    
    # Determine num_heads based on hidden size
    num_heads = 6 if hidden_size == 384 else 8
    
    # Get max_seq_len
    max_seq_len = checkpoint.get('pos_encoding.pe', torch.zeros(1, 57, 1)).shape[1]
    
    print(f"  Model: hidden={hidden_size}, layers={num_layers}, heads={num_heads}")
    
    # Create and load model
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
    
    # Run evaluation
    print(f"\nEvaluating with {num_test_walks} test walks...")
    
    results, error_summary, _, _, _, _, _ = evaluate_model(
        model=model,
        graph=graph,
        vocab=vocab,
        num_walks=num_test_walks,
        min_start_length=32,
        max_start_length=32,
        rules=rules,
        verbose=False,
        track_token_details=False,
        trajectory_sampling_config=None,
        config=None,
        fast_mode=True
    )
    
    return error_summary


def main():
    """Compare original and retrained models."""
    
    print("="*70)
    print("VERIFYING RETRAIN FIX")
    print("="*70)
    print()
    print("This script compares the original model (with edge-adding bug)")
    print("against the retrained model (with fixed walk generation).")
    print()
    
    # Find the most recent fixed model
    experiments_dir = Path("experiments")
    fixed_dirs = sorted([d for d in experiments_dir.glob("fixed_run_*") if d.is_dir()])
    
    if not fixed_dirs:
        print("❌ No retrained models found.")
        print("   Please run 'python run_full_retrain.py' first.")
        return 1
    
    latest_fixed = fixed_dirs[-1]
    print(f"Latest retrained model: {latest_fixed.name}")
    
    # Also look for original model
    original_dirs = sorted([d for d in experiments_dir.glob("run_*") if d.is_dir() and not d.name.startswith("fixed_")])
    
    if original_dirs:
        latest_original = original_dirs[-1]
        print(f"Latest original model: {latest_original.name}")
    else:
        latest_original = None
        print("No original model found for comparison.")
    
    print("\n" + "-"*70)
    
    # Evaluate retrained model
    print("\n📊 RETRAINED MODEL (Fixed Walk Generation)")
    print("-"*40)
    fixed_results = load_and_evaluate_model(str(latest_fixed), num_test_walks=100)
    
    if fixed_results:
        print("\nResults:")
        print(f"  ✅ Broken graph rate: {fixed_results.get('broken_graph_rate', 0):.2%}")
        print(f"  Repeater error rate: {fixed_results['repeater_error_rate']:.2%}")
        print(f"  Ascender error rate: {fixed_results['ascender_error_rate']:.2%}")
        print(f"  Even error rate: {fixed_results['even_error_rate']:.2%}")
        print(f"  Avg steps per walk: {fixed_results['avg_steps_per_walk']:.1f}")
    
    # Evaluate original model if available
    if latest_original:
        print("\n" + "-"*70)
        print("\n📊 ORIGINAL MODEL (With Edge-Adding Bug)")
        print("-"*40)
        original_results = load_and_evaluate_model(str(latest_original), num_test_walks=100)
        
        if original_results:
            print("\nResults:")
            print(f"  ❌ Broken graph rate: {original_results.get('broken_graph_rate', 0):.2%}")
            print(f"  Repeater error rate: {original_results['repeater_error_rate']:.2%}")
            print(f"  Ascender error rate: {original_results['ascender_error_rate']:.2%}")
            print(f"  Even error rate: {original_results['even_error_rate']:.2%}")
            print(f"  Avg steps per walk: {original_results['avg_steps_per_walk']:.1f}")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    if fixed_results:
        broken_rate = fixed_results.get('broken_graph_rate', 0)
        if broken_rate < 0.1:  # Less than 10%
            print("\n✅ SUCCESS: Retrained model correctly follows graph edges!")
            print(f"   Broken graph rate: {broken_rate:.2%}")
            print("   The fix has resolved the edge-adding issue.")
        elif broken_rate < 0.3:  # Less than 30%
            print("\n⚠️  PARTIAL SUCCESS: Retrained model shows significant improvement")
            print(f"   Broken graph rate: {broken_rate:.2%}")
            print("   Model may benefit from additional training epochs.")
        else:
            print("\n❌ ISSUE: Retrained model still has high broken graph rate")
            print(f"   Broken graph rate: {broken_rate:.2%}")
            print("   Further investigation needed.")
        
        if latest_original and original_results:
            improvement = original_results.get('broken_graph_rate', 1.0) - broken_rate
            print(f"\n   Improvement over original: {improvement:.2%} reduction in broken edges")
    
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())