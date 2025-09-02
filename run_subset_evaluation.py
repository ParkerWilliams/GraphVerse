#!/usr/bin/env python3
"""
Subset Evaluation with Balanced Rule Distribution

Evaluates model performance on a subset of walks with stratified sampling
to ensure even distribution of rule exposure.
"""

import os
import sys
import json
import time
import pickle
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
from graphverse.llm.evaluation_fast import evaluate_model_fast


def load_model_and_vocab(model_path, vocab_path, device='cpu'):
    """Load model and vocabulary."""
    import torch
    from graphverse.llm.model import WalkTransformer
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Create model with old architecture
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


def categorize_nodes(graph, rules):
    """Categorize nodes by their rule membership."""
    repeater_rule = None
    ascender_rule = None
    even_rule = None
    
    for rule in rules:
        if isinstance(rule, RepeaterRule):
            repeater_rule = rule
        elif isinstance(rule, AscenderRule):
            ascender_rule = rule
        elif isinstance(rule, EvenRule):
            even_rule = rule
    
    # Create node categories (convert string keys to int for repeater nodes)
    node_categories = {
        'repeater': [int(k) for k in repeater_rule.members_nodes_dict.keys()] if repeater_rule else [],
        'ascender': list(ascender_rule.member_nodes) if ascender_rule else [],
        'even': list(even_rule.member_nodes) if even_rule else [],
        'regular': []
    }
    
    # Find regular nodes (not in any rule)
    all_rule_nodes = set()
    for category in ['repeater', 'ascender', 'even']:
        all_rule_nodes.update(node_categories[category])
    
    node_categories['regular'] = [n for n in range(graph.n) if n not in all_rule_nodes]
    
    return node_categories


def generate_balanced_start_nodes(node_categories, num_walks_per_category):
    """Generate balanced starting nodes for walks."""
    start_nodes = []
    
    for category, nodes in node_categories.items():
        if not nodes:
            continue
            
        # Sample nodes with replacement if needed
        category_starts = []
        for _ in range(num_walks_per_category):
            category_starts.append(random.choice(nodes))
        
        start_nodes.extend([(node, category) for node in category_starts])
    
    # Shuffle to mix categories
    random.shuffle(start_nodes)
    
    return start_nodes


def evaluate_with_balanced_sampling(
    model,
    graph,
    rules,
    vocab,
    total_walks=10000,
    min_start_length=5,
    max_start_length=10,
    max_walk_length=100,
    device='cpu',
    verbose=True
):
    """Evaluate model with balanced rule exposure."""
    
    # Categorize nodes
    node_categories = categorize_nodes(graph, rules)
    
    if verbose:
        print("\n📊 Node Distribution:")
        for category, nodes in node_categories.items():
            print(f"  {category.capitalize()}: {len(nodes)} nodes")
    
    # Calculate walks per category (equal distribution)
    num_categories = sum(1 for nodes in node_categories.values() if nodes)
    walks_per_category = total_walks // num_categories
    
    if verbose:
        print(f"\n🎯 Balanced Sampling:")
        print(f"  Total walks: {total_walks}")
        print(f"  Walks per category: {walks_per_category}")
    
    # Generate balanced starting nodes
    start_nodes = generate_balanced_start_nodes(node_categories, walks_per_category)
    
    # Track metrics by category
    category_metrics = defaultdict(lambda: {
        'total_walks': 0,
        'total_steps': 0,
        'completed_walks': 0,
        'repeater_errors': 0,
        'ascender_errors': 0,
        'even_errors': 0,
        'broken_graph_errors': 0,
        'avg_confidence': [],
        'walk_lengths': []
    })
    
    # Overall metrics
    overall_metrics = {
        'repeater_errors': 0,
        'ascender_errors': 0,
        'even_errors': 0,
        'broken_graph_errors': 0,
        'total_steps': 0,
        'completed_walks': 0
    }
    
    # Get unknown token index
    unk_token = vocab.token2idx.get("<UNK>", vocab.token2idx.get("<PAD>", 0))
    
    if verbose:
        print("\n⚡ Starting Balanced Evaluation...")
        from tqdm import tqdm
        iterator = tqdm(start_nodes[:total_walks], desc="Evaluating")
    else:
        iterator = start_nodes[:total_walks]
    
    model.eval()
    import torch
    
    for start_node, category in iterator:
        # Generate random walk length
        walk_length = random.randint(min_start_length, max_start_length)
        
        # Generate initial walk
        current = start_node
        generated_walk = [current]
        
        for _ in range(walk_length - 1):
            neighbors = graph.get_neighbors(current)
            if len(neighbors) == 0:
                break
            current = random.choice(neighbors)
            generated_walk.append(current)
        
        # Convert to tensor
        input_ids = [vocab.token2idx.get(str(node), unk_token) 
                    for node in generated_walk]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        current_vertex = generated_walk[-1]
        step_count = 0
        walk_completed = False
        confidences = []
        
        # Track metrics for this walk
        category_metrics[category]['total_walks'] += 1
        
        # Generate continuation
        while current_vertex in range(graph.n) and step_count < max_walk_length:
            # Get model prediction
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits[0, -1], dim=-1)
                next_vertex_idx = torch.argmax(logits[0, -1]).item()
                confidence = probs[next_vertex_idx].item()
                confidences.append(confidence)
            
            # Convert to vertex
            next_vertex_str = vocab.idx2token.get(next_vertex_idx, "<PAD>")
            
            # Check for end token
            if next_vertex_str == "<END>":
                walk_completed = True
                category_metrics[category]['completed_walks'] += 1
                overall_metrics['completed_walks'] += 1
                break
            
            # Try to parse as integer
            try:
                next_vertex = int(next_vertex_str)
            except (ValueError, TypeError):
                break
            
            # Check if edge exists
            if not graph.has_edge(current_vertex, next_vertex):
                category_metrics[category]['broken_graph_errors'] += 1
                overall_metrics['broken_graph_errors'] += 1
                break
            
            # Quick rule violation checks
            for rule in rules:
                if isinstance(rule, RepeaterRule):
                    # Check repeater violation (convert to string for dict lookup)
                    if str(current_vertex) in rule.members_nodes_dict:
                        k = rule.members_nodes_dict[str(current_vertex)]
                        # Simple check: if we've seen this node k times, next should be the repeater
                        recent_walk = generated_walk[-(k+1):] if len(generated_walk) > k else []
                        if len(recent_walk) == k+1 and recent_walk[0] == current_vertex:
                            if next_vertex != current_vertex:
                                category_metrics[category]['repeater_errors'] += 1
                                overall_metrics['repeater_errors'] += 1
                
                elif isinstance(rule, AscenderRule):
                    if current_vertex in rule.member_nodes:
                        if next_vertex <= current_vertex:
                            category_metrics[category]['ascender_errors'] += 1
                            overall_metrics['ascender_errors'] += 1
                
                elif isinstance(rule, EvenRule):
                    if current_vertex in rule.member_nodes:
                        if next_vertex % 2 != 0:
                            category_metrics[category]['even_errors'] += 1
                            overall_metrics['even_errors'] += 1
            
            # Add to walk
            generated_walk.append(next_vertex)
            
            # Update input tensor
            new_token_idx = vocab.token2idx.get(str(next_vertex), unk_token)
            new_token = torch.tensor([[new_token_idx]], dtype=torch.long).to(device)
            input_tensor = torch.cat([input_tensor, new_token], dim=1)
            
            # Keep tensor size manageable
            if input_tensor.size(1) > 100:
                input_tensor = input_tensor[:, -100:]
            
            current_vertex = next_vertex
            step_count += 1
            category_metrics[category]['total_steps'] += 1
            overall_metrics['total_steps'] += 1
        
        # Record walk metrics
        category_metrics[category]['walk_lengths'].append(len(generated_walk))
        if confidences:
            category_metrics[category]['avg_confidence'].append(np.mean(confidences))
    
    # Calculate final statistics
    results = {
        'total_walks': total_walks,
        'overall': overall_metrics,
        'by_category': {}
    }
    
    for category, metrics in category_metrics.items():
        if metrics['total_walks'] == 0:
            continue
            
        results['by_category'][category] = {
            'total_walks': metrics['total_walks'],
            'total_steps': metrics['total_steps'],
            'completed_walks': metrics['completed_walks'],
            'avg_walk_length': np.mean(metrics['walk_lengths']) if metrics['walk_lengths'] else 0,
            'avg_confidence': np.mean(metrics['avg_confidence']) if metrics['avg_confidence'] else 0,
            'error_rates': {
                'repeater': metrics['repeater_errors'] / max(1, metrics['total_steps']),
                'ascender': metrics['ascender_errors'] / max(1, metrics['total_steps']),
                'even': metrics['even_errors'] / max(1, metrics['total_steps']),
                'broken_graph': metrics['broken_graph_errors'] / max(1, metrics['total_walks'])
            }
        }
    
    # Calculate overall error rates
    results['overall']['error_rates'] = {
        'repeater': overall_metrics['repeater_errors'] / max(1, overall_metrics['total_steps']),
        'ascender': overall_metrics['ascender_errors'] / max(1, overall_metrics['total_steps']),
        'even': overall_metrics['even_errors'] / max(1, overall_metrics['total_steps']),
        'broken_graph': overall_metrics['broken_graph_errors'] / max(1, total_walks)
    }
    
    return results


def print_evaluation_summary(results):
    """Print a formatted summary of evaluation results."""
    print("\n" + "="*60)
    print("SUBSET EVALUATION RESULTS")
    print("="*60)
    
    # Overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"  Total walks: {results['total_walks']:,}")
    print(f"  Total steps: {results['overall']['total_steps']:,}")
    print(f"  Completed walks: {results['overall']['completed_walks']:,}")
    print(f"  Avg steps/walk: {results['overall']['total_steps']/results['total_walks']:.1f}")
    
    # Overall error rates
    print(f"\n❌ Overall Error Rates:")
    for error_type, rate in results['overall']['error_rates'].items():
        print(f"  {error_type}: {rate*100:.2f}%")
    
    # Per-category results
    print(f"\n📈 Results by Starting Node Category:")
    for category, metrics in results['by_category'].items():
        print(f"\n  {category.upper()}:")
        print(f"    Walks: {metrics['total_walks']}")
        print(f"    Avg length: {metrics['avg_walk_length']:.1f}")
        print(f"    Avg confidence: {metrics['avg_confidence']*100:.1f}%")
        print(f"    Completed: {metrics['completed_walks']}")
        print(f"    Error rates:")
        for error_type, rate in metrics['error_rates'].items():
            if rate > 0:
                print(f"      {error_type}: {rate*100:.2f}%")
    
    print("="*60)


def main():
    """Run subset evaluation with balanced rule distribution."""
    print("🎯 Subset Evaluation with Balanced Rule Distribution")
    print("="*60)
    
    # Configuration
    context_window = 8  # Use available model
    num_walks = 10000
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
    print(f"✅ Device: {device}")
    
    # Run balanced evaluation
    print(f"\n⚡ Running subset evaluation with {num_walks:,} walks...")
    start_time = time.time()
    
    results = evaluate_with_balanced_sampling(
        model=model,
        graph=graph,
        rules=rules,
        vocab=vocab,
        total_walks=num_walks,
        min_start_length=5,
        max_start_length=10,
        max_walk_length=context_window * 3,
        device=device,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    # Save results
    output_dir = f"small_results/subset_eval_ctx_{context_window}"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "balanced_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    print(f"⏱️ Total time: {elapsed:.1f}s ({elapsed/num_walks*1000:.1f}ms per walk)")
    print(f"🚀 Speed: {num_walks/elapsed:.1f} walks/second")
    
    # Print summary
    print_evaluation_summary(results)
    
    # Performance estimate
    if num_walks < 100000:
        estimated_100k = (100000 / num_walks) * elapsed / 60
        print(f"\n⏱️ Estimated time for 100k walks: {estimated_100k:.1f} minutes")


if __name__ == "__main__":
    import torch
    main()