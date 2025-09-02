#!/usr/bin/env python3
"""Run experiment with fast evaluation."""

import os
import sys
import json
import time
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.llm.evaluation_fast import evaluate_model_fast

def load_model_and_vocab(model_path, vocab_path, device='cpu'):
    """Load model and vocabulary."""
    import torch
    from graphverse.llm.model import WalkTransformer
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Create model with old architecture
    # The checkpoint uses 'max_length' and the positional encoding is 34 (likely 32 + special tokens)
    max_len = checkpoint['model_state_dict']['pos_encoder.weight'].shape[0]
    model = WalkTransformer(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        dropout=model_config.get('dropout', 0.15),
        max_seq_len=max_len
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load vocabulary
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    return model, vocab, model_config

def main():
    """Run fast evaluation experiment."""
    print("🚀 Fast Evaluation Experiment")
    print("=" * 60)
    
    # Configuration
    context_window = 8  # Use available model
    num_walks = 10000  # Start with 10k for testing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    models_dir = "small_models"
    graph_path = "small_graph_100"
    model_path = os.path.join(models_dir, f"model_ctx_{context_window}.pt")
    vocab_path = os.path.join(models_dir, f"vocab_ctx_{context_window}.pkl")
    
    # Check files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using: python scripts/train_models.py")
        return
    
    # Load graph
    print(f"\n📊 Loading graph from {graph_path}...")
    graph = Graph.load_graph(graph_path)
    
    # Load rules
    with open(f"{graph_path}_rules.json", "r") as f:
        rule_info = json.load(f)
    
    from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
    rules = [
        AscenderRule(rule_info["ascender_nodes"]),
        EvenRule(rule_info["even_nodes"]),
        RepeaterRule(rule_info["repeater_nodes_dict"])
    ]
    
    print(f"✅ Graph: {graph.n} nodes")
    print(f"✅ Rules: {len(rule_info['ascender_nodes'])} ascenders, "
          f"{len(rule_info['even_nodes'])} evens, "
          f"{len(rule_info['repeater_nodes_dict'])} repeaters")
    
    # Load model
    print(f"\n🧠 Loading model for context window {context_window}...")
    model, vocab, model_config = load_model_and_vocab(model_path, vocab_path, device)
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✅ Vocabulary: {len(vocab.token2idx)} tokens")
    
    # Run fast evaluation
    print(f"\n⚡ Running fast evaluation with {num_walks:,} walks...")
    start_time = time.time()
    
    results = evaluate_model_fast(
        model=model,
        graph=graph,
        rules=rules,
        vocab=vocab,
        num_walks=num_walks,
        min_start_length=5,
        max_start_length=10,
        max_walk_length=context_window * 3,  # 3x context window
        verbose=True,
        device=device
    )
    
    elapsed = time.time() - start_time
    
    # Save results
    output_dir = f"small_results/fast_eval_ctx_{context_window}"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    print(f"⏱️ Total time: {elapsed:.1f}s ({elapsed/num_walks*1000:.1f}ms per walk)")
    print(f"🚀 Speed: {num_walks/elapsed:.1f} walks/second")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Context window: {context_window}")
    print(f"Total walks: {num_walks:,}")
    print(f"Total steps: {results['total_steps']:,}")
    print(f"Completed walks: {results['completed_walks']:,}")
    print(f"Avg steps/walk: {results['avg_steps_per_walk']:.1f}")
    print("\nError rates:")
    for error_type, rate in results['error_rates'].items():
        print(f"  {error_type}: {rate*100:.2f}%")
    
    # Estimate time for full 100k walks
    if num_walks < 100000:
        estimated_100k = (100000 / num_walks) * elapsed / 60
        print(f"\n⏱️ Estimated time for 100k walks: {estimated_100k:.1f} minutes")

if __name__ == "__main__":
    import torch
    main()