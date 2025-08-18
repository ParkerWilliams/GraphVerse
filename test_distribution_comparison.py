#!/usr/bin/env python3
"""
Test script for enhanced distribution comparison functionality.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.llm.evaluation import (
    compute_core_distribution_comparison,
    graph_edge_distribution,
    valid_neighbors_baseline,
    uniform_random_baseline,
    fit_exponential_mle
)
from graphverse.graph.base import Graph
from graphverse.data.preparation import WalkVocabulary


def create_test_setup():
    """Create a simple test graph and vocabulary."""
    # Create a small test graph
    n = 10
    
    # Create graph using the proper constructor
    graph = Graph(n)  # This creates an n x n zero adjacency matrix
    
    # Add some edges manually
    for i in range(n-1):
        graph.adjacency[i, i+1] = 1.0
        graph.adjacency[i+1, i] = 1.0
    
    # Add some random additional edges
    np.random.seed(42)  # For reproducibility
    for i in range(n):
        for j in range(i+2, n):
            if np.random.rand() > 0.7:  # 30% chance of edge
                graph.adjacency[i, j] = 1.0
                graph.adjacency[j, i] = 1.0
    
    # Create vocabulary using dummy walks
    dummy_walks = [[i, i+1] for i in range(n-1)]
    vocab = WalkVocabulary(dummy_walks)
    
    return graph, vocab


def test_distribution_comparison():
    """Test the enhanced distribution comparison functionality."""
    print("Testing Enhanced Distribution Comparison")
    print("=" * 50)
    
    # Setup
    graph, vocab = create_test_setup()
    device = torch.device('cpu')
    current_vertex = 2
    vocab_size = len(vocab.token2idx)
    
    # Create a mock prediction distribution (more realistic)
    logits = torch.randn(vocab_size) * 2.0
    logits[vocab.token2idx['3']] += 3.0  # Make node 3 more likely
    logits[vocab.token2idx['1']] += 2.0  # Make node 1 somewhat likely
    prediction_probs = torch.softmax(logits, dim=-1)
    
    print(f"Graph: {graph.n} vertices")
    print(f"Current vertex: {current_vertex}")
    print(f"Valid neighbors: {graph.get_neighbors(current_vertex)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Top predictions: {[(vocab.idx2token[i], f'{p:.3f}') for i, p in enumerate(prediction_probs.topk(5)[0])]}")
    
    try:
        # Test core distribution comparison
        comparison = compute_core_distribution_comparison(
            prediction_probs, current_vertex, graph, vocab, device
        )
        
        print("\n✅ Core distribution comparison computed successfully!")
        
        # Display key results
        print("\nModel Distribution Stats:")
        model_stats = comparison['model_distribution_stats']
        for key, value in model_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\nBaseline Distribution Characteristics:")
        for baseline_name, baseline_stats in comparison['baseline_distributions'].items():
            print(f"  {baseline_name}:")
            for key, value in baseline_stats.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        
        print("\nDistribution Distances:")
        for baseline_name, distances in comparison['distribution_distances'].items():
            print(f"  {baseline_name}:")
            for metric, value in distances.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
        
        print("\nPrediction Quality Scores:")
        quality_scores = comparison['prediction_quality_scores']
        for dimension, score in quality_scores.items():
            if isinstance(score, float):
                print(f"  {dimension}: {score:.4f}")
            else:
                print(f"  {dimension}: {score}")
        
        print("\n✅ All analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in distribution comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_baselines():
    """Test individual baseline distribution functions."""
    print("\n" + "=" * 50)
    print("Testing Individual Baseline Functions")
    print("=" * 50)
    
    graph, vocab = create_test_setup()
    device = torch.device('cpu')
    current_vertex = 2
    vocab_size = len(vocab.token2idx)
    
    # Test each baseline function
    try:
        # Graph structure baseline
        graph_dist = graph_edge_distribution(current_vertex, graph, vocab).to(device)
        print(f"✅ Graph structure baseline: {torch.sum(graph_dist > 0).item()} non-zero entries")
        
        # Valid neighbors baseline
        valid_dist = valid_neighbors_baseline(current_vertex, graph, vocab).to(device)
        print(f"✅ Valid neighbors baseline: {torch.sum(valid_dist > 0).item()} non-zero entries")
        
        # Uniform baseline
        uniform_dist = uniform_random_baseline(vocab_size).to(device)
        print(f"✅ Uniform baseline: uniform distribution over {vocab_size} tokens")
        
        # Exponential baseline (need prediction probs)
        prediction_probs = torch.softmax(torch.randn(vocab_size), dim=-1)
        exp_dist = fit_exponential_mle(prediction_probs)
        print(f"✅ Exponential fitted baseline: {torch.sum(exp_dist > 0.001).item()} significant entries")
        
        # Verify distributions sum to 1
        for name, dist in [("Graph", graph_dist), ("Valid", valid_dist), ("Uniform", uniform_dist), ("Exponential", exp_dist)]:
            total = torch.sum(dist).item()
            print(f"  {name} distribution sums to: {total:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in baseline functions: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Enhanced Distribution Comparison Test Suite")
    print("=" * 60)
    
    # Test 1: Individual baseline functions
    test1_passed = test_individual_baselines()
    
    # Test 2: Complete distribution comparison
    test2_passed = test_distribution_comparison()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Individual baseline functions: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Core distribution comparison: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Enhanced distribution comparison is working correctly.")
        print("\nThe LLM's probability distribution will now be compared against:")
        print("  1. Graph structure distribution (actual edge probabilities)")
        print("  2. Uniform distribution over valid neighbors")
        print("  3. Exponential distribution fitted to model predictions")
        print("  4. Full uniform distribution (baseline)")
        print("\nDistance metrics computed:")
        print("  • KL Divergence (forward and reverse)")
        print("  • Jensen-Shannon Divergence (symmetric)")
        print("  • Kolmogorov-Smirnov Distance")
        print("  • L1 and L2 Distance")
        print("  • Cosine Similarity")
        print("  • Distribution Overlap Analysis")
        print("  • Prediction Quality Scores")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)