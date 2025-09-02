"""Model architecture visualization and enhanced evaluation plots."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def create_model_architecture_diagram(model_config: Dict, output_path: str = "model_architecture.png"):
    """Create a visual diagram of the transformer model architecture."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(5, 19, 'Enhanced Walk Transformer Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Model parameters text
    params_text = f"Parameters: {model_config.get('total_params', 'N/A'):,}\n"
    params_text += f"Hidden: {model_config.get('hidden_size', 384)} | "
    params_text += f"Layers: {model_config.get('num_layers', 4)} | "
    params_text += f"Heads: {model_config.get('num_heads', 6)}"
    ax.text(5, 18.2, params_text, fontsize=10, ha='center', style='italic')
    
    # Component heights and positions
    y_pos = 16.5
    box_height = 1.2
    box_width = 6
    spacing = 0.3
    
    # Color scheme
    colors = {
        'embedding': '#E8F4F8',
        'positional': '#D4E6F1',
        'transformer': '#AED6F1',
        'norm': '#85C1E2',
        'output': '#5DADE2',
        'arrow': '#2874A6'
    }
    
    def draw_component(y, height, width, text, color, subtext=None):
        """Draw a component box with text."""
        box = FancyBboxPatch(
            (5 - width/2, y), width, height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='#34495E',
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(5, y + height/2, text, fontsize=12, fontweight='bold', 
                ha='center', va='center')
        if subtext:
            ax.text(5, y + height/2 - 0.3, subtext, fontsize=9, 
                    ha='center', va='center', style='italic')
        return y - spacing - height
    
    def draw_arrow(y1, y2):
        """Draw an arrow between components."""
        arrow = FancyArrowPatch(
            (5, y1), (5, y2),
            arrowstyle='->,head_width=0.3,head_length=0.2',
            color=colors['arrow'],
            linewidth=2
        )
        ax.add_patch(arrow)
    
    # Input
    ax.text(5, y_pos + 0.5, 'Input Tokens [B, L]', fontsize=11, ha='center')
    y_pos -= 0.8
    
    # Token Embedding
    next_y = draw_component(y_pos, box_height, box_width, 
                           'Token Embedding', colors['embedding'],
                           f'vocab_size × {model_config.get("hidden_size", 384)}')
    draw_arrow(y_pos + box_height, y_pos)
    y_pos = next_y
    
    # Positional Encoding
    next_y = draw_component(y_pos, box_height, box_width,
                           'Sinusoidal + Learnable PE', colors['positional'],
                           'Hybrid positional encoding')
    draw_arrow(y_pos + box_height, y_pos)
    y_pos = next_y
    
    # Dropout
    y_pos -= 0.2
    ax.text(5, y_pos, '+ Dropout (0.1)', fontsize=9, ha='center', style='italic')
    y_pos -= 0.5
    
    # Transformer Layers
    num_layers = model_config.get('num_layers', 4)
    for i in range(num_layers):
        # Layer container
        layer_y = y_pos
        layer_height = 3.5
        
        # Draw layer box
        layer_box = FancyBboxPatch(
            (2, layer_y - layer_height), 6, layer_height,
            boxstyle="round,pad=0.05",
            facecolor='white',
            edgecolor='#2874A6',
            linewidth=2,
            linestyle='--'
        )
        ax.add_patch(layer_box)
        
        # Layer label
        ax.text(2.3, layer_y - 0.3, f'Layer {i+1}', fontsize=10, 
                fontweight='bold', color='#2874A6')
        
        # Components within layer
        inner_y = layer_y - 0.7
        
        # Pre-Norm 1
        ax.add_patch(FancyBboxPatch((2.5, inner_y - 0.4), 5, 0.4,
                                    facecolor=colors['norm'],
                                    edgecolor='#34495E'))
        ax.text(5, inner_y - 0.2, 'LayerNorm (pre)', fontsize=9, ha='center')
        
        # Multi-Head Attention
        inner_y -= 0.7
        ax.add_patch(FancyBboxPatch((2.5, inner_y - 0.5), 5, 0.5,
                                    facecolor=colors['transformer'],
                                    edgecolor='#34495E'))
        ax.text(5, inner_y - 0.25, f'Multi-Head Attention ({model_config.get("num_heads", 6)} heads)',
                fontsize=9, ha='center')
        
        # Gated Residual
        inner_y -= 0.4
        ax.text(5, inner_y, '+ Gated Residual (α₁)', fontsize=8, ha='center', style='italic')
        
        # Pre-Norm 2
        inner_y -= 0.3
        ax.add_patch(FancyBboxPatch((2.5, inner_y - 0.4), 5, 0.4,
                                    facecolor=colors['norm'],
                                    edgecolor='#34495E'))
        ax.text(5, inner_y - 0.2, 'LayerNorm (pre)', fontsize=9, ha='center')
        
        # Feedforward
        inner_y -= 0.7
        ax.add_patch(FancyBboxPatch((2.5, inner_y - 0.5), 5, 0.5,
                                    facecolor=colors['transformer'],
                                    edgecolor='#34495E'))
        ax.text(5, inner_y - 0.25, 'Feedforward (2048 dim, ReLU)',
                fontsize=9, ha='center')
        
        # Gated Residual
        inner_y -= 0.4
        ax.text(5, inner_y, '+ Gated Residual (α₂)', fontsize=8, ha='center', style='italic')
        
        # Arrow to next layer
        if i < num_layers - 1:
            draw_arrow(layer_y - layer_height - 0.1, layer_y - layer_height - 0.3)
        
        y_pos = layer_y - layer_height - 0.5
    
    # Final Layer Norm
    next_y = draw_component(y_pos, box_height * 0.8, box_width,
                           'Final LayerNorm', colors['norm'])
    draw_arrow(y_pos + box_height * 0.8, y_pos)
    y_pos = next_y
    
    # Output Projection
    next_y = draw_component(y_pos, box_height, box_width,
                           'Output Projection', colors['output'],
                           f'{model_config.get("hidden_size", 384)} × vocab_size')
    draw_arrow(y_pos + box_height, y_pos)
    y_pos = next_y
    
    # Temperature Scaling
    y_pos -= 0.2
    ax.text(5, y_pos, '÷ Temperature τ (learnable)', fontsize=9, ha='center', style='italic')
    y_pos -= 0.5
    
    # Output
    ax.text(5, y_pos, 'Logits [B, L, V]', fontsize=11, ha='center')
    
    # Add legend for enhancements
    legend_y = 2
    ax.text(1, legend_y, 'Key Enhancements:', fontsize=10, fontweight='bold')
    enhancements = [
        '• Pre-norm architecture',
        '• Gated residual connections',
        '• Hybrid positional encoding',
        '• ReLU activation (not GELU)',
        '• Learnable temperature',
        '• Label smoothing',
        '• Xavier initialization'
    ]
    for i, enh in enumerate(enhancements):
        ax.text(1, legend_y - 0.3 * (i + 1), enh, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Model architecture diagram saved to {output_path}")
    return output_path


def plot_training_dynamics(epoch_losses: List[float], learning_rates: List[float] = None,
                          output_path: str = "training_dynamics.png"):
    """Plot training loss and learning rate dynamics."""
    
    has_lr = learning_rates is not None and len(learning_rates) > 0
    fig, axes = plt.subplots(1, 2 if has_lr else 1, figsize=(12 if has_lr else 6, 5))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Loss plot
    ax1 = axes[0]
    epochs = range(1, len(epoch_losses) + 1)
    ax1.plot(epochs, epoch_losses, 'o-', linewidth=2, markersize=6, color='#2E86C1')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Progression')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Add annotations for key points
    min_loss = min(epoch_losses)
    min_epoch = epoch_losses.index(min_loss) + 1
    ax1.annotate(f'Min: {min_loss:.3f}',
                xy=(min_epoch, min_loss),
                xytext=(min_epoch + 1, min_loss + 0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9)
    
    # Learning rate plot if provided
    if has_lr and len(axes) > 1:
        ax2 = axes[1]
        steps = range(len(learning_rates))
        ax2.plot(steps, learning_rates, linewidth=1, color='#E74C3C')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        # Mark warmup end
        if len(learning_rates) > 10:
            warmup_steps = int(len(learning_rates) * 0.1)
            ax2.axvline(x=warmup_steps, color='gray', linestyle='--', alpha=0.5)
            ax2.text(warmup_steps, max(learning_rates) * 0.9, 'Warmup End',
                    rotation=90, va='top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training dynamics plot saved to {output_path}")
    return output_path


def plot_attention_patterns(attention_weights: torch.Tensor, tokens: List[str] = None,
                           layer_idx: int = 0, head_idx: int = 0,
                           output_path: str = "attention_patterns.png"):
    """Visualize attention patterns from the model."""
    
    if attention_weights is None or len(attention_weights) == 0:
        print("⚠️  No attention weights available to visualize")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot attention for first 6 heads
    for i in range(min(6, attention_weights.shape[1])):
        ax = axes[i]
        
        # Get attention matrix for this head
        attn = attention_weights[layer_idx, i].detach().cpu().numpy()
        
        # Plot heatmap
        im = ax.imshow(attn, cmap='Blues', aspect='auto')
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Keys')
        ax.set_ylabel('Queries')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Add token labels if provided
        if tokens and len(tokens) <= 20:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
    
    plt.suptitle(f'Attention Patterns - Layer {layer_idx + 1}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Attention patterns saved to {output_path}")
    return output_path


def plot_probability_distribution_evolution(model_distributions: List[np.ndarray],
                                           positions: List[int] = None,
                                           output_path: str = "prob_evolution.png"):
    """Plot how probability distributions evolve during walk generation."""
    
    if not model_distributions or len(model_distributions) == 0:
        print("⚠️  No distribution data available")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sample 4 positions to visualize
    n_dists = len(model_distributions)
    if n_dists < 4:
        sample_positions = list(range(n_dists))
    else:
        sample_positions = [0, n_dists//3, 2*n_dists//3, n_dists-1]
    
    for idx, (ax, pos) in enumerate(zip(axes.flatten(), sample_positions)):
        if pos < len(model_distributions):
            dist = model_distributions[pos]
            
            # Get top-k probabilities
            top_k = 20
            if isinstance(dist, torch.Tensor):
                dist = dist.detach().cpu().numpy()
            
            top_indices = np.argsort(dist)[-top_k:][::-1]
            top_probs = dist[top_indices]
            
            # Plot bar chart
            ax.bar(range(len(top_probs)), top_probs, color='#2E86C1')
            ax.set_xlabel('Token Rank')
            ax.set_ylabel('Probability')
            ax.set_title(f'Step {pos + 1}: Top-{top_k} Probabilities')
            ax.set_ylim([0, max(0.1, max(top_probs) * 1.1)])
            
            # Add entropy annotation
            entropy = -np.sum(dist * np.log(dist + 1e-10))
            ax.text(0.95, 0.95, f'H = {entropy:.2f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.suptitle('Probability Distribution Evolution During Walk', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Probability evolution saved to {output_path}")
    return output_path


def plot_rule_violation_heatmap(evaluation_results: Dict, rules: List,
                               output_path: str = "rule_violations_heatmap.png"):
    """Create a heatmap showing rule violations by position in walk."""
    
    # Create violation matrix
    max_position = 50
    rule_types = ['repeater', 'ascender', 'even', 'graph']
    violation_matrix = np.zeros((len(rule_types), max_position))
    
    # Aggregate violations by position
    if 'walks' in evaluation_results:
        for walk_data in evaluation_results['walks'][:1000]:  # Sample first 1000
            if 'violations' in walk_data:
                for pos, vtype in walk_data['violations'].items():
                    if pos < max_position and vtype in rule_types:
                        violation_matrix[rule_types.index(vtype), pos] += 1
    
    # Normalize by number of walks
    violation_matrix = violation_matrix / max(1, len(evaluation_results.get('walks', [1])))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(violation_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(0, max_position, 5))
    ax.set_xticklabels(np.arange(0, max_position, 5))
    ax.set_yticks(np.arange(len(rule_types)))
    ax.set_yticklabels([r.capitalize() for r in rule_types])
    
    # Labels
    ax.set_xlabel('Position in Walk')
    ax.set_ylabel('Rule Type')
    ax.set_title('Rule Violation Frequency by Position')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Violation Rate', rotation=270, labelpad=15)
    
    # Add grid
    ax.set_xticks(np.arange(max_position) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rule_types)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Rule violation heatmap saved to {output_path}")
    return output_path


def plot_perplexity_by_rule_exposure(token_level_data: List[Dict],
                                    output_path: str = "perplexity_by_rules.png"):
    """Plot model perplexity based on rule exposure in context."""
    
    if not token_level_data:
        print("⚠️  No token-level data available")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    rule_types = ['repeater', 'ascender', 'even']
    colors = ['#E74C3C', '#28B463', '#3498DB']
    
    for ax, rule_type, color in zip(axes, rule_types, colors):
        # Collect perplexities based on rule presence
        with_rule = []
        without_rule = []
        
        for token_data in token_level_data[:5000]:  # Sample data
            if 'perplexity' not in token_data:
                if 'confidence' in token_data:
                    perplexity = 1.0 / max(0.001, token_data['confidence'])
                else:
                    continue
            else:
                perplexity = token_data['perplexity']
            
            # Check if rule is in context
            context = token_data.get('context_tokens', [])
            has_rule = any(rule_type in str(t).lower() for t in context)
            
            if has_rule:
                with_rule.append(min(perplexity, 100))  # Cap at 100
            else:
                without_rule.append(min(perplexity, 100))
        
        # Create violin plot
        data = [with_rule, without_rule] if with_rule and without_rule else [[1], [1]]
        parts = ax.violinplot(data, positions=[1, 2], widths=0.7,
                             showmeans=True, showmedians=True)
        
        # Color the violins
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['With Rule', 'Without Rule'])
        ax.set_ylabel('Perplexity')
        ax.set_title(f'{rule_type.capitalize()} Rule')
        ax.set_ylim([0, 50])
        ax.grid(True, alpha=0.3)
        
        # Add sample sizes
        ax.text(1, ax.get_ylim()[1] * 0.95, f'n={len(with_rule)}',
               ha='center', fontsize=8)
        ax.text(2, ax.get_ylim()[1] * 0.95, f'n={len(without_rule)}',
               ha='center', fontsize=8)
    
    plt.suptitle('Model Perplexity by Rule Exposure in Context', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Perplexity by rule exposure saved to {output_path}")
    return output_path


def create_comprehensive_evaluation_report(
    model_config: Dict,
    training_metrics: Dict,
    evaluation_results: Dict,
    token_level_data: List[Dict] = None,
    output_dir: str = "."
):
    """Generate all visualization plots for model evaluation."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    generated_plots = []
    
    # 1. Model Architecture Diagram
    try:
        path = create_model_architecture_diagram(
            model_config,
            os.path.join(output_dir, "model_architecture.png")
        )
        generated_plots.append(path)
    except Exception as e:
        print(f"⚠️  Failed to create architecture diagram: {e}")
    
    # 2. Training Dynamics
    if 'epoch_losses' in training_metrics:
        try:
            path = plot_training_dynamics(
                training_metrics['epoch_losses'],
                training_metrics.get('learning_rates'),
                os.path.join(output_dir, "training_dynamics.png")
            )
            generated_plots.append(path)
        except Exception as e:
            print(f"⚠️  Failed to create training dynamics: {e}")
    
    # 3. Probability Distribution Evolution
    if 'model_distributions' in evaluation_results:
        try:
            path = plot_probability_distribution_evolution(
                evaluation_results['model_distributions'][:100],
                output_path=os.path.join(output_dir, "prob_evolution.png")
            )
            if path:
                generated_plots.append(path)
        except Exception as e:
            print(f"⚠️  Failed to create probability evolution: {e}")
    
    # 4. Rule Violation Heatmap
    try:
        path = plot_rule_violation_heatmap(
            evaluation_results,
            [],
            os.path.join(output_dir, "rule_violations_heatmap.png")
        )
        generated_plots.append(path)
    except Exception as e:
        print(f"⚠️  Failed to create violation heatmap: {e}")
    
    # 5. Perplexity by Rule Exposure
    if token_level_data:
        try:
            path = plot_perplexity_by_rule_exposure(
                token_level_data,
                os.path.join(output_dir, "perplexity_by_rules.png")
            )
            if path:
                generated_plots.append(path)
        except Exception as e:
            print(f"⚠️  Failed to create perplexity plot: {e}")
    
    print(f"\n📊 Generated {len(generated_plots)} visualization plots")
    return generated_plots