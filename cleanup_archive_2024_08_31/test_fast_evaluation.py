#!/usr/bin/env python3
"""Test fast evaluation module performance."""

import time
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
from graphverse.llm.model_enhanced import EnhancedWalkTransformer
from graphverse.llm.evaluation_fast import evaluate_model_fast

class SimpleVocab:
    """Simple vocabulary for testing."""
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
    
    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

def test_fast_evaluation():
    """Test that fast evaluation runs quickly."""
    print("Testing Fast Evaluation Module")
    print("=" * 60)
    
    # Create small test graph
    n = 100
    graph = Graph(n)
    
    # Add edges
    edge_count = 0
    for i in range(n):
        for j in range(i+1, min(i+10, n)):
            if np.random.random() < 0.3:
                graph.add_edge(i, j, weight=1.0)
                edge_count += 1
    
    print(f"Created graph: {n} nodes, {edge_count} edges")
    
    # Create rules with some sample nodes
    repeater_dict = {10: 3, 20: 4, 30: 5}  # node: k-cycle length
    ascender_nodes = [15, 25, 35, 45]
    even_nodes = [i for i in range(0, n, 2)][:20]  # First 20 even nodes
    
    rules = [
        RepeaterRule(repeater_dict),
        AscenderRule(ascender_nodes),
        EvenRule(even_nodes)
    ]
    
    # Create vocabulary
    vocab = SimpleVocab()
    tokens = [str(i) for i in range(n)] + ["<END>", "<PAD>", "<UNK>"]
    vocab.add_tokens(tokens)
    print(f"Vocabulary size: {len(vocab.token2idx)}")
    
    # Create model
    model = EnhancedWalkTransformer(
        vocab_size=len(vocab.token2idx),
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different numbers of walks
    test_configs = [
        (10, "Quick test"),
        (100, "Small batch"),
        (1000, "Full test")
    ]
    
    for num_walks, desc in test_configs:
        print(f"\n{desc}: {num_walks} walks")
        print("-" * 40)
        
        start_time = time.time()
        results = evaluate_model_fast(
            model=model,
            graph=graph,
            rules=rules,
            vocab=vocab,
            num_walks=num_walks,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        walks_per_sec = num_walks / elapsed
        time_per_walk = elapsed / num_walks * 1000  # in ms
        
        print(f"Time: {elapsed:.2f}s")
        print(f"Speed: {walks_per_sec:.1f} walks/sec")
        print(f"Time per walk: {time_per_walk:.1f}ms")
        print(f"Total steps: {results['total_steps']}")
        print(f"Completed walks: {results['completed_walks']}")
        print(f"Avg steps/walk: {results['avg_steps_per_walk']:.1f}")
        
        # Check if performance is acceptable
        if time_per_walk > 100:  # More than 100ms per walk is too slow
            print("⚠️ WARNING: Performance is still too slow!")
        else:
            print("✅ Performance is acceptable")
    
    # Compare with expected performance
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print(f"Target: < 100ms per walk")
    print(f"Achieved: {time_per_walk:.1f}ms per walk")
    
    if time_per_walk < 100:
        print("✅ Fast evaluation module is working efficiently!")
        estimated_100k = (100000 * time_per_walk / 1000 / 60)
        print(f"Estimated time for 100k walks: {estimated_100k:.1f} minutes")
    else:
        print("❌ Fast evaluation still needs optimization")

if __name__ == "__main__":
    test_fast_evaluation()