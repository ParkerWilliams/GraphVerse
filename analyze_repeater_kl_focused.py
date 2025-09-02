#!/usr/bin/env python3
"""
Focused KL divergence analysis for repeater walks.
Simplified version that correctly loads rules from node_attributes.
"""

import os
import sys
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
from graphverse.llm.model_enhanced import EnhancedWalkTransformer


def load_model_and_vocab(model_path, vocab_path, device='cpu'):
    """Load model and vocabulary."""
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # It's just the state dict
    state_dict = checkpoint
    
    # Infer model dimensions from state dict
    if 'embedding.weight' in state_dict:
        vocab_size = state_dict['embedding.weight'].shape[0]
        hidden_size = state_dict['embedding.weight'].shape[1]
    else:
        raise ValueError("Cannot determine model dimensions from checkpoint")
    
    # Determine max_seq_len
    if 'pos_encoding.pe' in state_dict:
        max_len = state_dict['pos_encoding.pe'].shape[1]
    else:
        max_len = 1024
    
    # Count layers
    num_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('layers.') and '.' in k], default=-1) + 1
    if num_layers == 0:
        num_layers = 4
    
    # Determine num_heads
    if hidden_size == 384:
        num_heads = 6  # 384 / 64 = 6
    else:
        num_heads = 8
    
    print(f"Model config: vocab={vocab_size}, hidden={hidden_size}, layers={num_layers}, heads={num_heads}")
    
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
    
    return model, vocab


def compute_kl_divergence(model_probs, graph_probs):
    """Compute KL divergence: KL(model || graph)"""
    eps = 1e-10
    model_probs = model_probs + eps
    graph_probs = graph_probs + eps
    
    model_probs = model_probs / model_probs.sum()
    graph_probs = graph_probs / graph_probs.sum()
    
    kl_div = torch.sum(model_probs * torch.log(model_probs / graph_probs))
    return kl_div.item()


def get_graph_edge_distribution(vertex, graph, vocab):
    """Get the graph's edge probability distribution from a vertex."""
    vocab_size = len(vocab.token2idx)
    probs = torch.zeros(vocab_size)
    
    if vertex < graph.n:
        neighbors = graph.get_neighbors(vertex)
        
        if len(neighbors) > 0:
            # Uniform distribution over neighbors for simplicity
            neighbor_prob = 1.0 / len(neighbors)
            
            for neighbor in neighbors:
                # Map vertex to vocab index
                neighbor_token = str(neighbor)
                if neighbor_token in vocab.token2idx:
                    neighbor_idx = vocab.token2idx[neighbor_token]
                    probs[neighbor_idx] = neighbor_prob
    
    return probs


def analyze_repeater_walk(model, graph, vocab, walk, repeater_nodes, device='cpu'):
    """
    Analyze KL divergence for a single repeater walk.
    
    Returns:
        Dictionary with analysis results or None if no repeater
    """
    # Find first repeater encounter
    repeater_start_idx = None
    repeater_vertex = None
    k_value = None
    
    for idx, vertex in enumerate(walk):
        if vertex in repeater_nodes:
            repeater_start_idx = idx
            repeater_vertex = vertex
            # Get k value from repeater_cycles
            if vertex in graph.repeater_cycles:
                cycle = graph.repeater_cycles[vertex]
                k_value = len(cycle) if isinstance(cycle, list) else cycle
            else:
                k_value = 3  # Default
            break
    
    if repeater_start_idx is None:
        return None
    
    # Track KL divergences from repeater start
    kl_divergences = []
    positions = []
    
    # Convert walk to input tensor up to repeater start
    input_ids = [vocab.token2idx.get(str(v), vocab.token2idx.get("<PAD>", 0)) 
                 for v in walk[:repeater_start_idx + 1]]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Track through the k-cycle
    max_steps = k_value + 2
    
    for step in range(max_steps):
        if repeater_start_idx + step >= len(walk):
            break
        
        current_vertex = walk[repeater_start_idx + step]
        
        # Get model prediction
        with torch.no_grad():
            logits = model(input_tensor)
            model_probs = F.softmax(logits[0, -1], dim=-1)
        
        # Get graph distribution
        graph_probs = get_graph_edge_distribution(current_vertex, graph, vocab).to(device)
        
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
            
            # Keep tensor manageable
            if input_tensor.size(1) > 100:
                input_tensor = input_tensor[:, -100:]
    
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


def collect_repeater_walks(model, graph, vocab, repeater_nodes, num_walks=1000, device='cpu'):
    """Generate walks and collect those with repeater patterns."""
    import random
    
    repeater_analyses = []
    successful_count = 0
    failed_count = 0
    
    print(f"Generating {num_walks} walks to find repeater patterns...")
    print(f"Total repeater nodes: {len(repeater_nodes)}")
    
    for _ in tqdm(range(num_walks)):
        # Sometimes start from repeater nodes to increase chances
        if random.random() < 0.3 and repeater_nodes:
            start_vertex = random.choice(list(repeater_nodes))
        else:
            start_vertex = random.randint(0, graph.n - 1)
        
        walk_length = random.randint(30, 60)
        
        walk = [start_vertex]
        current = start_vertex
        
        for _ in range(walk_length - 1):
            neighbors = graph.get_neighbors(current)
            if len(neighbors) == 0:
                break
            current = random.choice(neighbors)
            walk.append(current)
        
        # Analyze if contains repeater
        analysis = analyze_repeater_walk(model, graph, vocab, walk, repeater_nodes, device)
        
        if analysis:
            repeater_analyses.append(analysis)
            if analysis['successful']:
                successful_count += 1
            else:
                failed_count += 1
            
            if successful_count >= 50 and failed_count >= 50:
                break
    
    print(f"Found {len(repeater_analyses)} repeater walks")
    print(f"  Successful: {successful_count}")
    print(f"  Failed: {failed_count}")
    
    return repeater_analyses


def plot_repeater_kl_analysis(repeater_analyses, output_path="repeater_kl_focused.png"):
    """Plot focused KL divergence analysis for repeater walks."""
    if not repeater_analyses:
        print("No repeater walks to plot")
        return
    
    successful_walks = [a for a in repeater_analyses if a['successful']]
    failed_walks = [a for a in repeater_analyses if not a['successful']]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Average KL divergence trajectory
    ax1 = axes[0, 0]
    
    max_k = max(a['k_value'] for a in repeater_analyses)
    max_steps = max_k + 2
    
    # Calculate averages
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
    with np.errstate(divide='ignore', invalid='ignore'):
        successful_avg = np.where(successful_count > 0, 
                                  successful_avg / successful_count, np.nan)
        failed_avg = np.where(failed_count > 0,
                             failed_avg / failed_count, np.nan)
    
    x_positions = np.arange(max_steps)
    
    # Plot with nan handling
    mask_s = ~np.isnan(successful_avg)
    mask_f = ~np.isnan(failed_avg)
    
    if np.any(mask_s):
        ax1.plot(x_positions[mask_s], successful_avg[mask_s], 'g-', 
                label='Successful', linewidth=2, alpha=0.8)
    if np.any(mask_f):
        ax1.plot(x_positions[mask_f], failed_avg[mask_f], 'r-', 
                label='Failed', linewidth=2, alpha=0.8)
    
    avg_k = np.mean([a['k_value'] for a in repeater_analyses])
    ax1.axvline(x=0, color='blue', linestyle='-', alpha=0.3, label='Repeater Start')
    ax1.axvline(x=avg_k + 1, color='purple', linestyle='--', alpha=0.5, 
                label=f'Expected Return (k≈{avg_k:.1f})')
    
    ax1.set_xlabel('Position from Repeater Start')
    ax1.set_ylabel('KL Divergence')
    ax1.set_title('Average KL Divergence: Model vs Graph Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Individual trajectories
    ax2 = axes[0, 1]
    
    # Plot sample trajectories
    for i, analysis in enumerate(successful_walks[:10]):
        ax2.plot(analysis['positions'], analysis['kl_divergences'], 
                'g-', alpha=0.3, linewidth=1)
    
    for i, analysis in enumerate(failed_walks[:10]):
        ax2.plot(analysis['positions'], analysis['kl_divergences'], 
                'r-', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Position from Repeater Start')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('Individual Walk Trajectories (Sample)')
    ax2.grid(True, alpha=0.3)
    
    # 3. KL at return position
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
    
    if successful_return_kl and failed_return_kl:
        ax3.boxplot([successful_return_kl, failed_return_kl], 
                    labels=['Successful', 'Failed'])
        ax3.set_ylabel('KL Divergence at Return Position')
        ax3.set_title('KL Divergence at Expected Return to Repeater')
        ax3.grid(True, alpha=0.3)
    
    # 4. Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Distribution of k values
    k_values = [a['k_value'] for a in repeater_analyses]
    unique_k, k_counts = np.unique(k_values, return_counts=True)
    
    stats_text = f"""
REPEATER WALK ANALYSIS
{'='*30}

Total Walks: {len(repeater_analyses)}
  Successful: {len(successful_walks)} ({100*len(successful_walks)/len(repeater_analyses):.1f}%)
  Failed: {len(failed_walks)} ({100*len(failed_walks)/len(repeater_analyses):.1f}%)

k-value Distribution:
"""
    for k, count in zip(unique_k, k_counts):
        stats_text += f"  k={k}: {count} walks\n"
    
    if successful_return_kl and failed_return_kl:
        stats_text += f"""
KL at Return Position:
  Successful: {np.mean(successful_return_kl):.3f} ± {np.std(successful_return_kl):.3f}
  Failed: {np.mean(failed_return_kl):.3f} ± {np.std(failed_return_kl):.3f}
"""
    
    ax4.text(0.1, 0.9, stats_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Repeater Walk KL Divergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to {output_path}")
    
    return output_path


def main():
    """Run focused repeater KL divergence analysis."""
    print("🔍 Focused Repeater KL Divergence Analysis")
    print("="*60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    experiment_dir = "experiments/run_20250831_121356"
    
    # Load graph
    print(f"\n📊 Loading graph...")
    with open(os.path.join(experiment_dir, "data", "graph.pkl"), "rb") as f:
        graph = pickle.load(f)
    
    # Extract repeater nodes from node_attributes
    repeater_nodes = set()
    for node, attrs in graph.node_attributes.items():
        if attrs.get('rule') == 'repeater':
            repeater_nodes.add(node)
    
    print(f"✅ Graph: {graph.n} nodes")
    print(f"✅ Repeater nodes: {len(repeater_nodes)}")
    print(f"✅ Repeater cycles: {len(graph.repeater_cycles)}")
    
    # Load model and vocab
    print(f"\n🧠 Loading model...")
    model_path = os.path.join(experiment_dir, "model.pth")
    vocab_path = os.path.join(experiment_dir, "data", "vocab.pkl")
    model, vocab = load_model_and_vocab(model_path, vocab_path, device)
    print(f"✅ Model loaded on {device}")
    
    # Collect and analyze repeater walks
    print(f"\n📈 Analyzing repeater walks...")
    repeater_analyses = collect_repeater_walks(
        model=model,
        graph=graph,
        vocab=vocab,
        repeater_nodes=repeater_nodes,
        num_walks=2000,
        device=device
    )
    
    if not repeater_analyses:
        print("❌ No repeater walks found!")
        return
    
    # Generate visualization
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = plot_repeater_kl_analysis(
        repeater_analyses,
        os.path.join(output_dir, "repeater_kl_focused.png")
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()