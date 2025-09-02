#!/usr/bin/env python3
"""
Analyze KL divergence patterns for repeater walks.

This script tracks KL divergence from the first encounter with a repeater node
through the entire k-cycle, comparing successful vs unsuccessful completions.
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule


def load_model_and_vocab(model_path, vocab_path, device='cpu'):
    """Load model and vocabulary."""
    from graphverse.llm.model_enhanced import EnhancedWalkTransformer
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if it's a raw state dict or has structure
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        model_config = checkpoint.get('model_config', {})
    else:
        # It's just the state dict
        state_dict = checkpoint
        # Infer config from state dict
        model_config = {}
    
    # Infer model dimensions from state dict
    if 'embedding.weight' in state_dict:
        vocab_size = state_dict['embedding.weight'].shape[0]
        hidden_size = state_dict['embedding.weight'].shape[1]
    else:
        raise ValueError("Cannot determine model dimensions from checkpoint")
    
    # Determine max_seq_len (enhanced model uses pos_encoding.pe)
    if 'pos_encoding.pe' in state_dict:
        max_len = state_dict['pos_encoding.pe'].shape[1]
    elif 'pos_encoder.weight' in state_dict:
        max_len = state_dict['pos_encoder.weight'].shape[0]  
    else:
        max_len = 1024  # Default
    
    # Count transformer layers (enhanced model uses 'layers' instead of 'transformer.layers')
    num_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('layers.') and '.' in k], default=-1) + 1
    if num_layers == 0:
        num_layers = 4  # Default
    
    # Infer num_heads from attention in_proj dimensions
    if 'layers.0.self_attn.in_proj_weight' in state_dict:
        in_proj_weight = state_dict['layers.0.self_attn.in_proj_weight']
        # in_proj contains q, k, v so its first dim should be 3 * hidden_size
        num_heads = hidden_size // (in_proj_weight.shape[1] // hidden_size)
    else:
        num_heads = 8  # Default
    
    print(f"Detected model config: vocab={vocab_size}, hidden={hidden_size}, layers={num_layers}, heads={num_heads}")
    
    model = EnhancedWalkTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
        max_seq_len=max_len
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    return model, vocab, model_config


def compute_kl_divergence(model_probs, graph_probs):
    """
    Compute KL divergence: KL(model || graph)
    
    Args:
        model_probs: Model's probability distribution over vertices
        graph_probs: Graph's edge probability distribution
    
    Returns:
        KL divergence value
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    model_probs = model_probs + eps
    graph_probs = graph_probs + eps
    
    # Normalize to ensure they sum to 1
    model_probs = model_probs / model_probs.sum()
    graph_probs = graph_probs / graph_probs.sum()
    
    # Compute KL divergence
    kl_div = torch.sum(model_probs * torch.log(model_probs / graph_probs))
    
    return kl_div.item()


def get_graph_edge_distribution(vertex, graph, vocab_size):
    """
    Get the graph's edge probability distribution from a vertex.
    
    Args:
        vertex: Current vertex
        graph: Graph object
        vocab_size: Size of vocabulary
    
    Returns:
        Probability distribution tensor
    """
    probs = torch.zeros(vocab_size)
    
    if vertex < graph.n:
        # Get neighbors and their edge weights
        neighbors = graph.get_neighbors(vertex)
        
        if len(neighbors) > 0:
            # Get edge weights (uniform if not weighted)
            edge_weights = []
            for neighbor in neighbors:
                # Check if graph has edge weights
                if hasattr(graph, 'adjacency'):
                    weight = graph.adjacency[vertex, neighbor]
                else:
                    weight = 1.0
                edge_weights.append(weight)
            
            # Normalize weights to probabilities
            edge_weights = np.array(edge_weights)
            edge_probs = edge_weights / edge_weights.sum()
            
            # Assign probabilities to vocabulary indices
            for neighbor, prob in zip(neighbors, edge_probs):
                # Assuming vocabulary maps vertex indices to string tokens
                neighbor_idx = neighbor  # Direct mapping for simplicity
                if neighbor_idx < vocab_size:
                    probs[neighbor_idx] = prob
    
    return probs


def analyze_repeater_walk(
    model, 
    graph, 
    vocab, 
    walk, 
    repeater_rule,
    device='cpu'
):
    """
    Analyze KL divergence for a single repeater walk.
    
    Args:
        model: Trained model
        graph: Graph object
        vocab: Vocabulary
        walk: Walk sequence (list of vertices)
        repeater_rule: RepeaterRule object
        device: Compute device
    
    Returns:
        Dictionary with analysis results
    """
    # Find first repeater encounter
    repeater_start_idx = None
    repeater_vertex = None
    k_value = None
    
    for idx, vertex in enumerate(walk):
        if str(vertex) in repeater_rule.members_nodes_dict:
            repeater_start_idx = idx
            repeater_vertex = vertex
            k_value = repeater_rule.members_nodes_dict[str(vertex)]
            break
    
    if repeater_start_idx is None:
        return None  # No repeater in walk
    
    # Track KL divergences from repeater start
    kl_divergences = []
    positions = []
    
    # Convert walk to input tensor up to repeater start
    input_ids = [vocab.token2idx.get(str(v), vocab.token2idx.get("<PAD>", 0)) 
                 for v in walk[:repeater_start_idx + 1]]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Track through the k-cycle
    current_position = 0
    max_steps = k_value + 2  # k steps + return to repeater + buffer
    
    for step in range(max_steps):
        if repeater_start_idx + step >= len(walk):
            break
        
        current_vertex = walk[repeater_start_idx + step]
        
        # Get model prediction
        with torch.no_grad():
            logits = model(input_tensor)
            model_probs = F.softmax(logits[0, -1], dim=-1)
        
        # Get graph distribution
        vocab_size = len(vocab.token2idx)
        graph_probs = get_graph_edge_distribution(current_vertex, graph, vocab_size).to(device)
        
        # Compute KL divergence
        kl_div = compute_kl_divergence(model_probs, graph_probs)
        kl_divergences.append(kl_div)
        positions.append(step)
        
        # Update input for next step
        if repeater_start_idx + step + 1 < len(walk):
            next_vertex = walk[repeater_start_idx + step + 1]
            next_token_idx = vocab.token2idx.get(str(next_vertex), vocab.token2idx.get("<PAD>", 0))
            new_token = torch.tensor([[next_token_idx]], dtype=torch.long).to(device)
            input_tensor = torch.cat([input_tensor, new_token], dim=1)
    
    # Check if walk successfully returned to repeater
    expected_return_idx = repeater_start_idx + k_value + 1
    successful = False
    if expected_return_idx < len(walk):
        successful = (walk[expected_return_idx] == repeater_vertex)
    
    return {
        'repeater_vertex': repeater_vertex,
        'k_value': k_value,
        'start_idx': repeater_start_idx,
        'positions': positions,
        'kl_divergences': kl_divergences,
        'successful': successful,
        'walk_length': len(walk)
    }


def analyze_even_walk(
    model, 
    graph, 
    vocab, 
    walk, 
    even_rule,
    device='cpu'
):
    """
    Analyze KL divergence for a walk relative to even nodes.
    
    Args:
        model: Trained model
        graph: Graph object
        vocab: Vocabulary
        walk: Walk sequence (list of vertices)
        even_rule: EvenRule object
        device: Compute device
    
    Returns:
        Dictionary with analysis results
    """
    # Find all even node encounters
    even_encounters = []
    for idx, vertex in enumerate(walk):
        if vertex in even_rule.member_nodes:
            even_encounters.append(idx)
    
    if not even_encounters:
        return None  # No even nodes in walk
    
    # Track KL divergences relative to even nodes
    kl_data = []
    
    # For each position in walk, calculate distance to nearest even node
    for idx in range(len(walk)):
        # Find distance to nearest even node (negative if before, positive if after)
        distances_to_even = [idx - even_idx for even_idx in even_encounters]
        # Get the smallest absolute distance
        min_dist_idx = min(range(len(distances_to_even)), 
                          key=lambda i: abs(distances_to_even[i]))
        distance_from_even = distances_to_even[min_dist_idx]
        
        # Only track positions near even nodes (within ±10 steps)
        if abs(distance_from_even) <= 10:
            current_vertex = walk[idx]
            
            # Convert walk up to this point to input tensor
            input_ids = [vocab.token2idx.get(str(v), vocab.token2idx.get("<PAD>", 0)) 
                        for v in walk[:idx + 1]]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            # Get model prediction
            with torch.no_grad():
                logits = model(input_tensor)
                model_probs = F.softmax(logits[0, -1], dim=-1)
            
            # Get graph distribution
            vocab_size = len(vocab.token2idx)
            graph_probs = get_graph_edge_distribution(current_vertex, graph, vocab_size).to(device)
            
            # Compute KL divergence
            kl_div = compute_kl_divergence(model_probs, graph_probs)
            
            kl_data.append({
                'position': idx,
                'distance_from_even': distance_from_even,
                'kl_divergence': kl_div,
                'current_vertex': current_vertex,
                'is_even_node': current_vertex in even_rule.member_nodes
            })
    
    # Check for even rule violations
    violation_idx = None
    for idx in range(len(walk) - 1):
        current = walk[idx]
        next_vertex = walk[idx + 1]
        
        # Check if current is even node and next violates rule
        if current in even_rule.member_nodes:
            if next_vertex % 2 != 0:  # Next should be even but isn't
                violation_idx = idx + 1
                break
    
    return {
        'even_encounters': even_encounters,
        'kl_data': kl_data,
        'violation_idx': violation_idx,
        'successful': violation_idx is None,
        'walk_length': len(walk)
    }


def collect_repeater_walks(
    model,
    graph,
    vocab,
    rules,
    num_walks=1000,
    min_length=20,
    max_length=50,
    device='cpu'
):
    """
    Generate walks and collect those with repeater patterns.
    
    Returns:
        List of analyzed repeater walks
    """
    repeater_rule = None
    for rule in rules:
        if isinstance(rule, RepeaterRule):
            repeater_rule = rule
            break
    
    if not repeater_rule:
        print("No repeater rule found!")
        return []
    
    repeater_analyses = []
    successful_count = 0
    failed_count = 0
    
    print(f"Generating {num_walks} walks to find repeater patterns...")
    
    for _ in tqdm(range(num_walks)):
        # Generate a random walk
        import random
        start_vertex = random.randint(0, graph.n - 1)
        walk_length = random.randint(min_length, max_length)
        
        walk = [start_vertex]
        current = start_vertex
        
        for _ in range(walk_length - 1):
            neighbors = graph.get_neighbors(current)
            if len(neighbors) == 0:
                break
            current = random.choice(neighbors)
            walk.append(current)
        
        # Analyze if contains repeater
        analysis = analyze_repeater_walk(
            model, graph, vocab, walk, repeater_rule, device
        )
        
        if analysis:
            repeater_analyses.append(analysis)
            if analysis['successful']:
                successful_count += 1
            else:
                failed_count += 1
            
            # Stop if we have enough samples
            if successful_count >= 50 and failed_count >= 50:
                break
    
    print(f"Found {len(repeater_analyses)} repeater walks")
    print(f"  Successful: {successful_count}")
    print(f"  Failed: {failed_count}")
    
    return repeater_analyses


def collect_even_walks(
    model,
    graph,
    vocab,
    rules,
    num_walks=1000,
    min_length=20,
    max_length=50,
    device='cpu'
):
    """
    Generate walks and collect those with even rule patterns.
    
    Returns:
        List of analyzed even walks
    """
    even_rule = None
    for rule in rules:
        if isinstance(rule, EvenRule):
            even_rule = rule
            break
    
    if not even_rule:
        print("No even rule found!")
        return []
    
    even_analyses = []
    successful_count = 0
    failed_count = 0
    
    print(f"Generating {num_walks} walks to find even patterns...")
    
    for _ in tqdm(range(num_walks)):
        # Generate a random walk
        import random
        start_vertex = random.randint(0, graph.n - 1)
        walk_length = random.randint(min_length, max_length)
        
        walk = [start_vertex]
        current = start_vertex
        
        for _ in range(walk_length - 1):
            neighbors = graph.get_neighbors(current)
            if len(neighbors) == 0:
                break
            current = random.choice(neighbors)
            walk.append(current)
        
        # Analyze if contains even nodes
        analysis = analyze_even_walk(
            model, graph, vocab, walk, even_rule, device
        )
        
        if analysis:
            even_analyses.append(analysis)
            if analysis['successful']:
                successful_count += 1
            else:
                failed_count += 1
            
            # Stop if we have enough samples
            if successful_count >= 50 and failed_count >= 50:
                break
    
    print(f"Found {len(even_analyses)} even walks")
    print(f"  Successful: {successful_count}")
    print(f"  Failed: {failed_count}")
    
    return even_analyses


def plot_even_kl_divergence(even_analyses, output_path="even_kl_analysis.png"):
    """
    Plot KL divergence patterns relative to distance from even nodes.
    
    Args:
        even_analyses: List of analyzed even walks
        output_path: Path to save the plot
    """
    if not even_analyses:
        print("No even walk data to plot")
        return
    
    # Separate successful and failed walks
    successful_walks = [a for a in even_analyses if a['successful']]
    failed_walks = [a for a in even_analyses if not a['successful']]
    
    if not successful_walks or not failed_walks:
        print("Not enough data for comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. KL divergence by distance from even node
    ax1 = axes[0]
    
    # Collect data by distance
    distances = range(-10, 11)  # -10 to +10 steps from even node
    successful_kl_by_dist = defaultdict(list)
    failed_kl_by_dist = defaultdict(list)
    
    for analysis in successful_walks:
        for data_point in analysis['kl_data']:
            dist = data_point['distance_from_even']
            kl = data_point['kl_divergence']
            successful_kl_by_dist[dist].append(kl)
    
    for analysis in failed_walks:
        for data_point in analysis['kl_data']:
            dist = data_point['distance_from_even']
            kl = data_point['kl_divergence']
            failed_kl_by_dist[dist].append(kl)
    
    # Calculate averages
    successful_avg = []
    failed_avg = []
    x_positions = []
    
    for dist in distances:
        if dist in successful_kl_by_dist and len(successful_kl_by_dist[dist]) > 0:
            successful_avg.append(np.mean(successful_kl_by_dist[dist]))
        else:
            successful_avg.append(np.nan)
        
        if dist in failed_kl_by_dist and len(failed_kl_by_dist[dist]) > 0:
            failed_avg.append(np.mean(failed_kl_by_dist[dist]))
        else:
            failed_avg.append(np.nan)
        
        x_positions.append(dist)
    
    # Plot lines
    ax1.plot(x_positions, successful_avg, 'g-', label='Successful', linewidth=2, alpha=0.8)
    ax1.plot(x_positions, failed_avg, 'r-', label='Failed', linewidth=2, alpha=0.8)
    
    # Mark the even node position
    ax1.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Even Node')
    ax1.axvline(x=1, color='purple', linestyle=':', alpha=0.5, label='Decision Point')
    
    # Mark failure points for failed walks
    failure_distances = []
    for analysis in failed_walks:
        if analysis['violation_idx'] is not None:
            # Find distance from even node at failure
            for data_point in analysis['kl_data']:
                if data_point['position'] == analysis['violation_idx']:
                    failure_distances.append(data_point['distance_from_even'])
                    break
    
    if failure_distances:
        unique_failures, counts = np.unique(failure_distances, return_counts=True)
        # Scale marker size by frequency
        max_count = max(counts)
        for dist, count in zip(unique_failures, counts):
            # Find KL at this distance for failed walks
            kl_at_failure = np.mean(failed_kl_by_dist[dist]) if dist in failed_kl_by_dist else 0
            ax1.scatter(dist, kl_at_failure, color='red', s=100*(count/max_count), 
                       alpha=0.6, marker='x', zorder=5)
    
    ax1.set_xlabel('Distance from Even Node (steps)')
    ax1.set_ylabel('KL Divergence')
    ax1.set_title('KL Divergence Relative to Even Rule Nodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-10, 10)
    
    # 2. Distribution of failure points
    ax2 = axes[1]
    
    if failure_distances:
        ax2.hist(failure_distances, bins=21, range=(-10.5, 10.5), 
                color='red', alpha=0.7, edgecolor='darkred')
        ax2.axvline(x=1, color='purple', linestyle=':', alpha=0.5, 
                   label='Expected Failure Point')
        ax2.set_xlabel('Distance from Even Node at Failure')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Even Rule Violation Points')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No failures detected', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Distribution of Even Rule Violation Points')
    
    plt.suptitle('Even Rule KL Divergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved even rule plot to {output_path}")
    
    return output_path


def plot_kl_divergence_comparison(repeater_analyses, output_path="repeater_kl_analysis.png"):
    """
    Plot KL divergence patterns for successful vs unsuccessful repeater walks.
    
    Args:
        repeater_analyses: List of analyzed repeater walks
        output_path: Path to save the plot
    """
    # Separate successful and failed walks
    successful_walks = [a for a in repeater_analyses if a['successful']]
    failed_walks = [a for a in repeater_analyses if not a['successful']]
    
    if not successful_walks or not failed_walks:
        print("Not enough data for comparison")
        return
    
    # Find maximum k value for x-axis range
    max_k = max(a['k_value'] for a in repeater_analyses)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Average KL divergence trajectory
    ax1 = axes[0, 0]
    
    # Compute average trajectories
    max_steps = max_k + 2
    successful_avg = np.zeros(max_steps)
    successful_count = np.zeros(max_steps)
    failed_avg = np.zeros(max_steps)
    failed_count = np.zeros(max_steps)
    
    for analysis in successful_walks:
        for pos, kl in zip(analysis['positions'], analysis['kl_divergences']):
            if pos < max_steps:
                successful_avg[pos] += kl
                successful_count[pos] += 1
    
    for analysis in failed_walks:
        for pos, kl in zip(analysis['positions'], analysis['kl_divergences']):
            if pos < max_steps:
                failed_avg[pos] += kl
                failed_count[pos] += 1
    
    # Normalize
    successful_avg = np.divide(successful_avg, successful_count, 
                              out=np.zeros_like(successful_avg), 
                              where=successful_count!=0)
    failed_avg = np.divide(failed_avg, failed_count,
                          out=np.zeros_like(failed_avg),
                          where=failed_count!=0)
    
    x_positions = np.arange(max_steps)
    ax1.plot(x_positions, successful_avg, 'g-', label='Successful', linewidth=2, alpha=0.8)
    ax1.plot(x_positions, failed_avg, 'r-', label='Failed', linewidth=2, alpha=0.8)
    ax1.fill_between(x_positions, 0, successful_avg, color='green', alpha=0.2)
    ax1.fill_between(x_positions, 0, failed_avg, color='red', alpha=0.2)
    
    ax1.set_xlabel('Position from Repeater Start')
    ax1.set_ylabel('KL Divergence')
    ax1.set_title('Average KL Divergence: Model vs Graph Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add vertical line at expected return position (average k)
    avg_k = np.mean([a['k_value'] for a in repeater_analyses])
    ax1.axvline(x=avg_k + 1, color='purple', linestyle='--', alpha=0.5, 
                label=f'Expected Return (k={avg_k:.1f})')
    
    # 2. Individual trajectories
    ax2 = axes[0, 1]
    
    # Plot sample of individual trajectories
    for analysis in successful_walks[:10]:
        ax2.plot(analysis['positions'], analysis['kl_divergences'], 
                'g-', alpha=0.3, linewidth=1)
    
    for analysis in failed_walks[:10]:
        ax2.plot(analysis['positions'], analysis['kl_divergences'], 
                'r-', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Position from Repeater Start')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('Individual Walk Trajectories (Sample)')
    ax2.grid(True, alpha=0.3)
    
    # 3. KL divergence at critical point (return position)
    ax3 = axes[1, 0]
    
    successful_return_kl = []
    failed_return_kl = []
    
    for analysis in successful_walks:
        return_pos = analysis['k_value'] + 1
        if return_pos < len(analysis['kl_divergences']):
            successful_return_kl.append(analysis['kl_divergences'][return_pos])
    
    for analysis in failed_walks:
        return_pos = analysis['k_value'] + 1
        if return_pos < len(analysis['kl_divergences']):
            failed_return_kl.append(analysis['kl_divergences'][return_pos])
    
    ax3.boxplot([successful_return_kl, failed_return_kl], 
                labels=['Successful', 'Failed'])
    ax3.set_ylabel('KL Divergence at Return Position')
    ax3.set_title('KL Divergence at Expected Return to Repeater')
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
REPEATER WALK ANALYSIS
{'='*30}

Total Walks Analyzed: {len(repeater_analyses)}
  Successful: {len(successful_walks)} ({100*len(successful_walks)/len(repeater_analyses):.1f}%)
  Failed: {len(failed_walks)} ({100*len(failed_walks)/len(repeater_analyses):.1f}%)

KL Divergence Statistics:
  
  At Repeater Start:
    Successful: {np.mean([a['kl_divergences'][0] for a in successful_walks if a['kl_divergences']]):.3f}
    Failed: {np.mean([a['kl_divergences'][0] for a in failed_walks if a['kl_divergences']]):.3f}
  
  At Return Position:
    Successful: {np.mean(successful_return_kl) if successful_return_kl else 0:.3f}
    Failed: {np.mean(failed_return_kl) if failed_return_kl else 0:.3f}

Average k-value: {avg_k:.1f}
"""
    
    ax4.text(0.1, 0.9, stats_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('KL Divergence Analysis: Repeater Walk Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to {output_path}")
    
    return output_path


def main():
    """Run repeater walk KL divergence analysis."""
    print("🔍 Repeater Walk KL Divergence Analysis")
    print("="*60)
    
    # Configuration
    context_window = 16  # From your config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths - using your Aug 31 experiment
    experiment_dir = "experiments/run_20250831_121356"
    model_path = os.path.join(experiment_dir, "model.pth")
    vocab_path = os.path.join(experiment_dir, "data", "vocab.pkl")
    graph_path = os.path.join(experiment_dir, "data", "graph.pkl")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please update the paths in the script")
        return
    
    # Load graph (pickle format)
    print(f"\n📊 Loading graph from {graph_path}...")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    
    # Debug: check graph attributes
    print(f"Graph attributes: {[attr for attr in dir(graph) if not attr.startswith('_') and not callable(getattr(graph, attr))][:10]}")
    
    # Load experiment metadata for rules
    metadata_path = os.path.join(experiment_dir, "data", "experiment_metadata.pkl")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    # Extract rule info from graph's rule objects
    # The graph object should have the rules attached
    if hasattr(graph, 'rules'):
        rules_from_graph = graph.rules
        rule_info = {
            "ascender_nodes": [],
            "even_nodes": [],
            "repeater_nodes_dict": {}
        }
        for rule in rules_from_graph:
            if hasattr(rule, 'is_ascender_rule'):
                rule_info["ascender_nodes"] = rule.member_nodes
            elif hasattr(rule, 'is_even_rule'):
                rule_info["even_nodes"] = rule.member_nodes
            elif hasattr(rule, 'is_repeater_rule'):
                rule_info["repeater_nodes_dict"] = rule.members_nodes_dict
    else:
        # Try to extract from metadata analysis
        if hasattr(metadata, 'analysis') and hasattr(metadata.analysis, 'rule_nodes'):
            rule_nodes = metadata.analysis.rule_nodes
            rule_info = {
                "ascender_nodes": rule_nodes.get('ascender', []),
                "even_nodes": rule_nodes.get('even', []),
                "repeater_nodes_dict": rule_nodes.get('repeater', {})
            }
        else:
            # Fall back to empty rules
            print("Warning: Could not find rule information")
            rule_info = {
                "ascender_nodes": [],
                "even_nodes": [],
                "repeater_nodes_dict": {}
            }
    
    rules = [
        AscenderRule(rule_info["ascender_nodes"]),
        EvenRule(rule_info["even_nodes"]),
        RepeaterRule(rule_info["repeater_nodes_dict"])
    ]
    
    print(f"✅ Graph: {graph.n} nodes")
    print(f"✅ Repeater nodes: {len(rule_info['repeater_nodes_dict'])}")
    
    # Load model
    print(f"\n🧠 Loading model...")
    model, vocab, model_config = load_model_and_vocab(model_path, vocab_path, device)
    print(f"✅ Model loaded on {device}")
    
    # Collect and analyze repeater walks
    print(f"\n📈 Analyzing repeater walks...")
    repeater_analyses = collect_repeater_walks(
        model=model,
        graph=graph,
        vocab=vocab,
        rules=rules,
        num_walks=2000,  # Generate enough walks to find repeater patterns
        min_length=30,
        max_length=60,
        device=device
    )
    
    # Collect and analyze even walks
    print(f"\n📈 Analyzing even rule walks...")
    even_analyses = collect_even_walks(
        model=model,
        graph=graph,
        vocab=vocab,
        rules=rules,
        num_walks=2000,  # Generate enough walks to find even patterns
        min_length=30,
        max_length=60,
        device=device
    )
    
    # Generate visualizations
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if repeater_analyses:
        plot_path = plot_kl_divergence_comparison(
            repeater_analyses,
            os.path.join(output_dir, "repeater_kl_divergence.png")
        )
    else:
        print("❌ No repeater walks found!")
    
    if even_analyses:
        even_plot_path = plot_even_kl_divergence(
            even_analyses,
            os.path.join(output_dir, "even_kl_divergence.png")
        )
    else:
        print("❌ No even walks found!")
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()