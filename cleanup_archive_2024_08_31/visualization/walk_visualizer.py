"""
Walk visualization tools for GraphVerse.
Shows text and graph representations of walks with rule annotations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


def visualize_walk_with_text(
    walk: List[int],
    graph,
    rules: List,
    title: str = "Walk Visualization",
    figsize: Tuple[int, int] = (16, 10),
    rule_violations: Optional[Dict] = None,
    kl_divergences: Optional[List[float]] = None
):
    """
    Create a comprehensive visualization showing both text and graph representation of a walk.
    
    Args:
        walk: The walk sequence
        graph: Graph object
        rules: List of rule objects
        title: Title for the visualization
        figsize: Figure size
        rule_violations: Dict of violations {position: violation_type}
        kl_divergences: KL divergence at each step
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], width_ratios=[1, 1])
    
    # Text representation (top)
    ax_text = fig.add_subplot(gs[0, :])
    _add_walk_text(ax_text, walk, rules, rule_violations)
    
    # Graph visualization (middle left)
    ax_graph = fig.add_subplot(gs[1, 0])
    _draw_walk_graph(ax_graph, walk, graph, rules)
    
    # Walk path visualization (middle right)
    ax_path = fig.add_subplot(gs[1, 1])
    _draw_walk_path(ax_path, walk, rules, rule_violations)
    
    # KL divergence over time (bottom)
    if kl_divergences:
        ax_kl = fig.add_subplot(gs[2, :])
        _plot_kl_divergence(ax_kl, kl_divergences, walk, rules)
    else:
        ax_rules = fig.add_subplot(gs[2, :])
        _add_rule_legend(ax_rules, rules, walk)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def _add_walk_text(ax, walk, rules, violations=None):
    """Add formatted text representation of walk with rule annotations."""
    ax.axis('off')
    
    # Identify rule nodes
    rule_nodes = _identify_rule_nodes(rules)
    
    # Track which ascender positions actually trigger the rule
    ascender_triggers = set()
    for i in range(len(walk) - 1):
        if walk[i] in rule_nodes['ascender']:
            ascender_triggers.add(i)
    
    # Build formatted text
    text_parts = []
    colors = []
    
    for i, node in enumerate(walk):
        # Determine color based on rules and violations
        color = 'black'
        annotation = ''
        
        if node in rule_nodes['repeater']:
            color = 'blue'
            k = rule_nodes['repeater'][node]
            annotation = f'(R{k})'
        elif node in rule_nodes['ascender'] and i in ascender_triggers:
            # Only mark as ascender if this position triggers the rule
            color = 'green'
            annotation = '(A→)' # Arrow indicates ascending requirement
        elif node in rule_nodes['even']:
            color = 'purple'
            annotation = '(E)'
        
        if violations and i in violations:
            color = 'red'
            annotation += '✗'
        
        text_parts.append(f"{node}{annotation}")
        colors.append(color)
    
    # Format walk text
    walk_str = "Walk: [" + ", ".join(text_parts) + "]"
    
    # Display with color coding
    x_offset = 0.05
    y_offset = 0.6
    
    # Title and walk length info
    ax.text(x_offset, y_offset + 0.25, f"Walk Sequence (length={len(walk)}):", fontsize=12, fontweight='bold')
    
    # Display walk in chunks for readability
    items_per_line = 12
    for line_num, start_idx in enumerate(range(0, len(walk), items_per_line)):
        end_idx = min(start_idx + items_per_line, len(walk))
        line_y = y_offset - line_num * 0.12
        
        current_x = x_offset + 0.05
        for i in range(start_idx, end_idx):
            part = text_parts[i]
            color = colors[i]
            
            # Add step number above
            ax.text(current_x, line_y + 0.04, f"{i}", fontsize=7, color='gray', ha='left')
            # Add node with annotation
            ax.text(current_x, line_y, part, fontsize=10, color=color, 
                   fontweight='bold' if color != 'black' else 'normal')
            current_x += 0.07
    
    # Add legend with better spacing
    legend_y = 0.15
    ax.text(x_offset, legend_y, "Legend:", fontsize=10, fontweight='bold')
    legend_items = [
        ("Rk=Repeater(k)", 'blue'),
        ("A→=Ascender(must go up)", 'green'),
        ("E=Even(next must be odd)", 'purple'),
        ("✗=Violation", 'red')
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.text(x_offset + 0.1 + i * 0.2, legend_y - 0.05, label, color=color, fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_walk_graph(ax, walk, graph, rules):
    """Draw the graph with the walk path highlighted."""
    # Create NetworkX graph with only walk nodes for clarity
    G = nx.DiGraph()
    
    # Add only nodes that appear in the walk
    unique_walk_nodes = list(set(walk))
    G.add_nodes_from(unique_walk_nodes)
    
    # Add edges from walk
    for i in range(len(walk) - 1):
        G.add_edge(walk[i], walk[i+1])
    
    # Use hierarchical layout for better readability
    try:
        # Try to create a more structured layout
        pos = nx.shell_layout(G)
    except:
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # Identify rule nodes
    rule_nodes = _identify_rule_nodes(rules)
    
    # Draw all nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in rule_nodes['repeater']:
            node_colors.append('blue')
            node_sizes.append(500 if node in walk else 200)
        elif node in rule_nodes['ascender']:
            node_colors.append('green')
            node_sizes.append(500 if node in walk else 200)
        elif node in rule_nodes['even']:
            node_colors.append('purple')
            node_sizes.append(500 if node in walk else 200)
        elif node in walk:
            node_colors.append('lightgray')
            node_sizes.append(400)
        else:
            node_colors.append('white')
            node_sizes.append(100)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          edgecolors='black', linewidths=1, ax=ax)
    
    # Draw walk edges with arrows
    walk_edges = [(walk[i], walk[i+1]) for i in range(len(walk)-1)]
    nx.draw_networkx_edges(G, pos, walk_edges, edge_color='red', width=2,
                          arrows=True, arrowsize=20, arrowstyle='->', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # Add start and end markers
    if walk:
        start_pos = pos[walk[0]]
        end_pos = pos[walk[-1]]
        ax.scatter(*start_pos, s=700, c='none', edgecolors='red', linewidths=3, marker='o')
        ax.scatter(*end_pos, s=700, c='none', edgecolors='red', linewidths=3, marker='s')
        ax.text(start_pos[0], start_pos[1] - 0.15, 'START', fontsize=8, ha='center', fontweight='bold')
        ax.text(end_pos[0], end_pos[1] - 0.15, 'END', fontsize=8, ha='center', fontweight='bold')
    
    ax.set_title("Graph Representation", fontsize=11, fontweight='bold')
    ax.axis('off')


def _draw_walk_path(ax, walk, rules, violations=None):
    """Draw walk as a linear path showing rule activations."""
    ax.set_title("Walk Path Timeline", fontsize=11, fontweight='bold')
    
    rule_nodes = _identify_rule_nodes(rules)
    
    # Create timeline
    y_base = 0.5
    x_spacing = 0.8 / max(len(walk) - 1, 1)
    
    # Draw path line
    x_positions = [0.1 + i * x_spacing for i in range(len(walk))]
    ax.plot(x_positions, [y_base] * len(walk), 'k-', alpha=0.3, linewidth=1)
    
    # Draw nodes
    for i, (x, node) in enumerate(zip(x_positions, walk)):
        # Determine node appearance
        color = 'lightgray'
        size = 100
        marker = 'o'
        
        if node in rule_nodes['repeater']:
            color = 'blue'
            size = 200
            marker = 's'  # Square for repeater
        elif node in rule_nodes['ascender']:
            color = 'green'
            size = 200
            marker = '^'  # Triangle for ascender
        elif node in rule_nodes['even']:
            color = 'purple'
            size = 200
            marker = 'D'  # Diamond for even
        
        # Mark violations
        edge_color = 'red' if violations and i in violations else 'black'
        edge_width = 3 if violations and i in violations else 1
        
        ax.scatter(x, y_base, s=size, c=color, marker=marker,
                  edgecolors=edge_color, linewidths=edge_width, zorder=5)
        
        # Add node label
        ax.text(x, y_base - 0.1, str(node), fontsize=8, ha='center')
        
        # Add step number
        ax.text(x, y_base + 0.1, f"t={i}", fontsize=7, ha='center', color='gray')
    
    # Draw arrows between nodes
    for i in range(len(walk) - 1):
        ax.annotate('', xy=(x_positions[i+1], y_base), xytext=(x_positions[i], y_base),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1))
    
    # Mark repeater cycles
    for node, k in rule_nodes['repeater'].items():
        if node in walk:
            positions = [i for i, n in enumerate(walk) if n == node]
            if len(positions) >= 2:
                for j in range(len(positions) - 1):
                    start_idx = positions[j]
                    end_idx = positions[j + 1]
                    
                    # Draw arc for cycle
                    x_start = x_positions[start_idx]
                    x_end = x_positions[end_idx]
                    arc_height = 0.15
                    
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (x_start, y_base - 0.02), x_end - x_start, arc_height,
                        boxstyle="round,pad=0.01", edgecolor='blue', facecolor='none',
                        linestyle='--', alpha=0.5))
                    
                    # Label cycle
                    x_mid = (x_start + x_end) / 2
                    ax.text(x_mid, y_base + arc_height, f"k={k}", fontsize=7, 
                           ha='center', color='blue')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def _plot_kl_divergence(ax, kl_divergences, walk, rules=None):
    """Plot KL divergence vs steps until violation."""
    ax.set_title("Model Uncertainty Analysis", fontsize=11, fontweight='bold')
    
    # Calculate steps until next violation for each position
    violations = _find_all_violations(walk, rules) if rules else {}
    steps_to_violation = []
    
    for i in range(len(kl_divergences)):
        # Find next violation after position i
        next_violation = None
        for v_pos in sorted(violations.keys()):
            if v_pos > i:
                next_violation = v_pos
                break
        
        if next_violation is not None:
            steps_to_violation.append(next_violation - i)
        else:
            steps_to_violation.append(len(walk) - i)  # Steps to end
    
    # Create scatter plot: KL divergence vs steps until violation
    scatter = ax.scatter(steps_to_violation[:len(kl_divergences)], kl_divergences, 
                        c=range(len(kl_divergences)), cmap='viridis', 
                        s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar to show progression through walk
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Walk Step', fontsize=9)
    
    # Add reference lines for KL divergence interpretation
    ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.3, label='Uniform-like')
    ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.3, label='Moderate certainty')
    ax.axhline(y=4.0, color='r', linestyle='--', alpha=0.3, label='High certainty')
    
    # Labels and formatting
    ax.set_xlabel("Steps Until Next Violation (or End)", fontsize=10)
    ax.set_ylabel("KL Divergence from Uniform", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=8)
    
    # Add annotations for interesting points
    if len(kl_divergences) > 0:
        max_kl_idx = np.argmax(kl_divergences)
        ax.annotate(f'Peak uncertainty\nat step {max_kl_idx}',
                   xy=(steps_to_violation[max_kl_idx], kl_divergences[max_kl_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, ha='left',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                                 color='red', alpha=0.7))


def _add_rule_legend(ax, rules, walk):
    """Add detailed rule legend."""
    ax.axis('off')
    
    rule_nodes = _identify_rule_nodes(rules)
    
    # Create legend content
    y_pos = 0.7
    x_pos = 0.1
    
    ax.text(x_pos, y_pos, "Active Rules in Walk:", fontsize=12, fontweight='bold')
    
    y_pos -= 0.15
    
    # Repeater rules
    if rule_nodes['repeater']:
        ax.text(x_pos, y_pos, "Repeaters:", fontsize=10, fontweight='bold', color='blue')
        for node, k in rule_nodes['repeater'].items():
            if node in walk:
                count = walk.count(node)
                ax.text(x_pos + 0.1, y_pos - 0.08, 
                       f"• Node {node}: k={k} (appears {count}x)", 
                       fontsize=9, color='blue')
                y_pos -= 0.08
        y_pos -= 0.05
    
    # Ascender rules
    if rule_nodes['ascender']:
        active_ascenders = [n for n in rule_nodes['ascender'] if n in walk]
        if active_ascenders:
            ax.text(x_pos, y_pos, "Ascenders:", fontsize=10, fontweight='bold', color='green')
            for node in active_ascenders:
                idx = walk.index(node)
                ax.text(x_pos + 0.1, y_pos - 0.08,
                       f"• Node {node}: activated at step {idx}",
                       fontsize=9, color='green')
                y_pos -= 0.08
            y_pos -= 0.05
    
    # Even rules
    if rule_nodes['even']:
        active_evens = [n for n in rule_nodes['even'] if n in walk]
        if active_evens:
            ax.text(x_pos, y_pos, "Even Rules:", fontsize=10, fontweight='bold', color='purple')
            for node in active_evens:
                idx = walk.index(node)
                ax.text(x_pos + 0.1, y_pos - 0.08,
                       f"• Node {node}: activated at step {idx}",
                       fontsize=9, color='purple')
                y_pos -= 0.08
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _identify_rule_nodes(rules):
    """Extract rule node information from rules list."""
    rule_nodes = {
        'repeater': {},
        'ascender': set(),
        'even': set()
    }
    
    for rule in rules:
        if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            rule_nodes['repeater'].update(rule.members_nodes_dict)
        elif hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
            rule_nodes['ascender'].update(rule.member_nodes)
        elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
            rule_nodes['even'].update(rule.member_nodes)
    
    return rule_nodes


def _find_all_violations(walk, rules):
    """Find all rule violations in a walk."""
    violations = {}
    
    if not rules:
        return violations
    
    for rule in rules:
        if hasattr(rule, 'get_violation_position'):
            if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                pos = rule.get_violation_position(walk)
            else:
                pos = rule.get_violation_position(None, walk)  # graph parameter
            
            if pos is not None:
                violation_type = 'repeater' if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule else \
                               'ascender' if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule else \
                               'even'
                violations[pos] = violation_type
    
    return violations


def visualize_rule_breaking_walks(
    walks_by_category: Dict[str, List[Tuple[List[int], List[float], Dict]]],
    graph,
    rules: List,
    figsize: Tuple[int, int] = (20, 12)
):
    """
    Visualize rule-breaking walks grouped by uncertainty patterns.
    Shows model confidence relative to uniform, default, and certain distributions.
    
    Args:
        walks_by_category: Dict with categories as keys:
            - 'peaked_correct': High confidence on correct predictions
            - 'peaked_default': Following default edge weights
            - 'uniform': Uniform/uncertain predictions
            Each value is list of (walk, kl_divergences, violations) tuples
        graph: Graph object
        rules: List of rule objects
    """
    categories = ['peaked_correct', 'peaked_default', 'uniform']
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    category_titles = {
        'peaked_correct': 'High Confidence (Correct)',
        'peaked_default': 'Default Edge Weights', 
        'uniform': 'Uncertain (Near-Uniform)'
    }
    
    # Calculate global KL range for normalization
    all_kl_values = []
    for category in categories:
        if category in walks_by_category:
            for walk, kl_divs, violations in walks_by_category[category][:3]:
                if kl_divs:
                    all_kl_values.extend(kl_divs)
    
    global_kl_range = (0, max(all_kl_values) * 1.1) if all_kl_values else (0, 5)
    
    for row, category in enumerate(categories):
        if category not in walks_by_category or not walks_by_category[category]:
            continue
        
        # Take up to 3 examples per category
        examples = walks_by_category[category][:3]
        
        for col, (walk, kl_divs, violations) in enumerate(examples):
            ax = axes[row, col]
            
            # Plot walk path with violations marked and normalized KL range
            _plot_violation_walk(ax, walk, rules, violations, kl_divs, global_kl_range)
            
            if col == 0:
                ax.set_ylabel(category_titles[category], fontsize=11, fontweight='bold')
    
    fig.suptitle("Rule-Breaking Walks by KL Divergence Pattern", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def _plot_violation_walk(ax, walk, rules, violations, kl_divergences, global_kl_range=None):
    """Plot a single walk with violations and uncertainty metrics."""
    # Create two y-axes
    ax2 = ax.twinx()
    
    # Plot KL divergence as main focus (not walk nodes)
    steps = list(range(len(walk)))
    
    # Plot uncertainty (KL divergence) as primary data
    if kl_divergences:
        ax.plot(steps[:len(kl_divergences)], kl_divergences, 'b-', alpha=0.8, linewidth=2)
        ax.fill_between(steps[:len(kl_divergences)], kl_divergences, alpha=0.3, color='blue')
        
        # Set consistent y-axis range across all plots if provided
        if global_kl_range:
            ax.set_ylim(global_kl_range)
        
        # Shade regions of high uncertainty
        high_uncertainty = [i for i, kl in enumerate(kl_divergences) if kl > 3.0]
        for i in high_uncertainty:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.4, color='orange', zorder=1)
    
    # Mark violations with different markers and colors at the bottom
    violation_markers = {'repeater': 'X', 'ascender': '^', 'even': 's', 'graph': 'o'}
    violation_colors = {'repeater': 'blue', 'ascender': 'green', 'even': 'purple', 'graph': 'red'}
    violation_names = {'repeater': 'R', 'ascender': 'A', 'even': 'E', 'graph': 'G'}
    
    y_bottom = ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.2
    
    for pos, vtype in violations.items():
        marker = violation_markers.get(vtype, 'x')
        color = violation_colors.get(vtype, 'red')
        name = violation_names.get(vtype, vtype[0].upper())
        
        # Place violation markers at bottom of plot
        ax.scatter(pos, y_bottom, s=150, c=color, marker=marker, 
                  edgecolors='black', linewidths=1, zorder=10)
        # Add violation label
        ax.text(pos, y_bottom - 0.1, name, fontsize=8, 
               ha='center', color=color, fontweight='bold')
    
    # Add reference lines for KL divergence interpretation
    if global_kl_range:
        ax.axhline(y=0.5, color='g', linestyle=':', alpha=0.3)
        ax.axhline(y=2.0, color='orange', linestyle=':', alpha=0.3)
        ax.axhline(y=4.0, color='r', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Walk Step', fontsize=8)
    ax.set_ylabel('Model Uncertainty (KL div)', fontsize=8, color='blue')
    ax.tick_params(axis='both', labelsize=7)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.2)
    
    # Format walk sequence for title (show first few and last few nodes)
    if len(walk) <= 8:
        walk_str = '→'.join(map(str, walk))
    else:
        walk_str = f"{walk[0]}→{walk[1]}→{walk[2]}→...→{walk[-2]}→{walk[-1]} (len={len(walk)})"
    
    # Add violation types to title
    violation_types = list(set(violations.values()))
    vtype_str = ','.join(violation_names.get(v, v) for v in violation_types)
    
    title = f"{walk_str}\nViolations: {vtype_str} ({len(violations)} total)"
    ax.set_title(title, fontsize=8, pad=10)


def create_kl_divergence_heatmap(
    walks: List[List[int]],
    kl_divergences: List[List[float]],
    max_length: int = 50,
    figsize: Tuple[int, int] = (14, 8),
    violation_positions: Optional[List[Dict]] = None
):
    """
    Create improved visualization showing KL divergence patterns and violations.
    
    Args:
        walks: List of walk sequences
        kl_divergences: Corresponding KL divergences for each walk
        max_length: Maximum walk length to display
        figsize: Figure size
        violation_positions: List of violation dicts for each walk
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Create scatter plot: KL divergence vs steps to violation
    all_kl_values = []
    all_steps_to_violation = []
    walk_indices = []
    
    for walk_idx, (walk, kl_seq) in enumerate(zip(walks, kl_divergences)):
        violations = violation_positions[walk_idx] if violation_positions else {}
        
        for step, kl_val in enumerate(kl_seq[:max_length]):
            # Find next violation
            next_violation = None
            for v_pos in sorted(violations.keys()):
                if v_pos > step:
                    next_violation = v_pos
                    break
            
            steps_to_viol = (next_violation - step) if next_violation else (len(walk) - step)
            
            all_kl_values.append(kl_val)
            all_steps_to_violation.append(steps_to_viol)
            walk_indices.append(walk_idx)
    
    # Main scatter plot
    scatter = ax1.scatter(all_steps_to_violation, all_kl_values, 
                         c=walk_indices, cmap='tab20', s=30, alpha=0.6)
    
    # Add trend line
    if len(all_steps_to_violation) > 1:
        z = np.polyfit(all_steps_to_violation, all_kl_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, max(all_steps_to_violation), 100)
        ax1.plot(x_trend, p(x_trend), 'r--', alpha=0.5, label=f'Trend: KL = {z[0]:.3f}*steps + {z[1]:.3f}')
    
    # Reference lines
    ax1.axhline(y=0.5, color='g', linestyle=':', alpha=0.3, label='Low uncertainty')
    ax1.axhline(y=2.0, color='orange', linestyle=':', alpha=0.3, label='Medium uncertainty')
    ax1.axhline(y=4.0, color='r', linestyle=':', alpha=0.3, label='High uncertainty')
    
    ax1.set_xlabel('Steps Until Next Violation', fontsize=11)
    ax1.set_ylabel('KL Divergence (Model Uncertainty)', fontsize=11)
    ax1.set_title('Model Uncertainty vs Distance to Rule Violations', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.2)
    
    # Histogram of steps to violation
    ax2.hist(all_steps_to_violation, bins=20, alpha=0.7, color='steelblue')
    ax2.set_xlabel('Steps Until Next Violation', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Distribution of Steps to Violations', fontsize=10)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    return fig


def analyze_kl_patterns(kl_divergences: List[float], threshold_peaked: float = 2.0, 
                        threshold_uniform: float = 0.5) -> str:
    """
    Analyze KL divergence pattern and classify it.
    
    Args:
        kl_divergences: KL divergence values over steps
        threshold_peaked: Threshold for peaked distribution
        threshold_uniform: Threshold for uniform distribution
        
    Returns:
        Classification string: 'peaked_correct', 'peaked_default', or 'uniform'
    """
    if not kl_divergences:
        return 'uniform'
    
    max_kl = max(kl_divergences)
    mean_kl = np.mean(kl_divergences)
    
    if max_kl > threshold_peaked:
        # Determine if peaked on correct or default
        # This would need actual model predictions to determine
        # For now, using a heuristic based on position
        peak_position = kl_divergences.index(max_kl)
        if peak_position < len(kl_divergences) // 2:
            return 'peaked_correct'
        else:
            return 'peaked_default'
    elif mean_kl < threshold_uniform:
        return 'uniform'
    else:
        return 'peaked_default'


def find_last_rule_encounter(walk, viol_type, viol_pos, rules):
    """
    Find the last position where a rule-setting vertex was encountered before the violation.
    
    Args:
        walk: The walk sequence
        viol_type: Type of violation ('repeater', 'ascender', 'even', 'graph')
        viol_pos: Position of the violation
        rules: List of rule objects
    
    Returns:
        Position of last rule vertex encounter, or -1 if none found
    """
    # Extract rule nodes
    rule_nodes = {
        'repeater': set(),
        'ascender': set(),
        'even': set()
    }
    
    for rule in rules:
        if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            rule_nodes['repeater'].update(rule.members_nodes_dict.keys())
        elif hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
            rule_nodes['ascender'].update(rule.member_nodes)
        elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
            rule_nodes['even'].update(rule.member_nodes)
    
    # For graph violations, we consider any step as valid
    if viol_type == 'graph':
        return 0
    
    # Find last encounter of relevant rule vertex before violation
    for pos in range(viol_pos - 1, -1, -1):
        if walk[pos] in rule_nodes.get(viol_type, set()):
            return pos
    
    return -1


def create_comprehensive_violation_analysis(
    walks_with_violations: List[Tuple[List[int], List[float], Dict, Optional[List[float]]]],
    max_lookback: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    model_distributions: Optional[List[List[np.ndarray]]] = None,
    graph = None,
    output_dir: str = '.',
    rules: Optional[List] = None,
    only_after_rule_encounter: bool = True
):
    """
    Create comprehensive plots for all metrics and violation types.
    Generates:
    - One plot per metric per violation type
    - One aggregated plot per metric across all violations
    
    Args:
        walks_with_violations: List of (walk, kl_divergences, violations, wasserstein_distances)
        max_lookback: Maximum steps before violation to analyze
        figsize: Figure size for individual plots
        model_distributions: Optional list of model predicted distributions for each walk
        graph: Optional graph object to compute edge distributions
        output_dir: Directory to save plots
        rules: List of rule objects (needed for finding rule encounters)
        only_after_rule_encounter: If True, only analyze steps after rule vertex was encountered
    
    Returns:
        Dict of generated figures
    """
    if not (model_distributions and graph):
        print("Model distributions and graph required for analysis")
        return None
    
    if only_after_rule_encounter and not rules:
        print("Rules required for post-encounter analysis")
        return None
    
    # Define all metrics to compute
    metrics = {
        'kl': ('KL Divergence', compute_kl_divergence),
        'js': ('JS Divergence', compute_js_divergence),
        'tv': ('Total Variation', compute_total_variation),
        'hellinger': ('Hellinger Distance', compute_hellinger_distance),
        'cross_entropy': ('Cross Entropy', compute_cross_entropy),
        'model_entropy': ('Model Entropy', lambda p, q: compute_entropy(p)),
        'edge_entropy': ('Edge Distribution Entropy', lambda p, q: compute_entropy(q))
    }
    
    # Organize data by violation type and metric
    data_by_type_and_metric = {
        'repeater': {metric: defaultdict(list) for metric in metrics},
        'ascender': {metric: defaultdict(list) for metric in metrics},
        'even': {metric: defaultdict(list) for metric in metrics},
        'graph': {metric: defaultdict(list) for metric in metrics},
        'all': {metric: defaultdict(list) for metric in metrics}  # Aggregated across all types
    }
    
    # Process all walks
    for walk_idx, (walk, kl_divs, violations, _) in enumerate(walks_with_violations):
        if walk_idx >= len(model_distributions) or not violations:
            continue
        
        model_dists = model_distributions[walk_idx]
        
        # Process each violation
        for viol_pos, viol_type in violations.items():
            if viol_type not in ['repeater', 'ascender', 'even', 'graph']:
                continue
            
            # Find last rule encounter if filtering is enabled
            if only_after_rule_encounter:
                last_rule_pos = find_last_rule_encounter(walk, viol_type, viol_pos, rules)
                if last_rule_pos < 0:
                    # No rule vertex encountered before violation, skip
                    continue
            else:
                last_rule_pos = 0
                
            # Look back from violation position, but only after rule encounter
            for steps_before in range(1, min(viol_pos + 1, max_lookback + 1)):
                pos = viol_pos - steps_before
                
                # Skip positions before the rule encounter
                if pos <= last_rule_pos:
                    continue
                    
                if pos >= 0 and pos < len(model_dists) and pos < len(walk) - 1:
                    current_vertex = walk[pos]
                    
                    # Get distributions
                    model_dist = model_dists[pos]
                    edge_dist = compute_edge_distribution(graph, current_vertex)
                    
                    if edge_dist is not None and len(model_dist) == len(edge_dist):
                        # Calculate all metrics
                        for metric_name, (_, compute_func) in metrics.items():
                            value = compute_func(model_dist, edge_dist)
                            data_by_type_and_metric[viol_type][metric_name][steps_before].append(value)
                            data_by_type_and_metric['all'][metric_name][steps_before].append(value)
    
    # Generate plots
    figures = {}
    colors = {
        'repeater': 'blue',
        'ascender': 'green',
        'even': 'purple',
        'graph': 'red',
        'all': 'black'
    }
    
    # For each metric, create plots for each violation type and one aggregated
    for metric_name, (metric_label, _) in metrics.items():
        for viol_type in ['repeater', 'ascender', 'even', 'graph', 'all']:
            metrics_by_steps = data_by_type_and_metric[viol_type][metric_name]
            
            if not metrics_by_steps:
                continue
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Calculate means and std errors
            steps_before_list = sorted(metrics_by_steps.keys(), reverse=True)
            means = []
            stds = []
            
            for steps in steps_before_list:
                values = metrics_by_steps[steps]
                means.append(np.mean(values))
                stds.append(np.std(values) / np.sqrt(len(values)))
            
            # Plot
            color = colors[viol_type]
            label = f'{viol_type.capitalize()} violations' if viol_type != 'all' else 'All violations'
            ax.plot(steps_before_list, means, color=color, linewidth=2.5, label=label)
            ax.fill_between(steps_before_list,
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.3, color=color)
            
            # Formatting
            ax.set_xlabel('Steps Before Violation (countdown)', fontsize=12)
            ax.set_ylabel(metric_label, fontsize=12)
            title_suffix = f'{viol_type.capitalize()} Violations' if viol_type != 'all' else 'All Violations'
            ax.set_title(f'{metric_label}: Model vs Edge Distribution\n{title_suffix}', 
                        fontsize=14, fontweight='bold')
            
            # Add violation marker
            ax.axvline(x=1, color='red', linestyle='-', alpha=0.5, linewidth=2)
            ax.fill_betweenx([ax.get_ylim()[0], ax.get_ylim()[1]], 0, 3, alpha=0.2, color='red')
            
            # Reference line for divergence metrics
            if metric_name in ['kl', 'js', 'tv', 'hellinger', 'cross_entropy']:
                ax.axhline(y=0, color='black', linestyle=':', alpha=0.3, label='Perfect alignment')
            
            # Add sample count
            sample_counts = [len(metrics_by_steps[s]) for s in steps_before_list]
            avg_samples = np.mean(sample_counts) if sample_counts else 0
            ax.text(0.98, 0.02, f'Avg samples: {avg_samples:.0f}', 
                   transform=ax.transAxes, fontsize=10, ha='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Trend annotation
            if len(means) > 5:
                early_mean = np.mean(means[:len(means)//3])
                late_mean = np.mean(means[-len(means)//3:])
                trend = "increasing" if late_mean > early_mean else "decreasing"
                percent_change = 100 * (late_mean - early_mean) / early_mean if early_mean > 0 else 0
                ax.text(0.02, 0.95, f'Trend: {trend} ({percent_change:+.1f}%)',
                       transform=ax.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.invert_xaxis()
            
            plt.tight_layout()
            
            # Save figure
            filename = f'{output_dir}/{metric_name}_{viol_type}_violations.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            figures[f'{metric_name}_{viol_type}'] = fig
            plt.close(fig)
    
    return figures


def create_violation_analysis_by_type(
    walks_with_violations: List[Tuple[List[int], List[float], Dict, Optional[List[float]]]],
    max_lookback: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    model_distributions: Optional[List[List[np.ndarray]]] = None,
    graph = None,
    metric: str = 'kl',
    output_dir: str = '.'
):
    """
    Create individual plots for each violation type and metric.
    
    Args:
        walks_with_violations: List of (walk, kl_divergences, violations, wasserstein_distances)
        max_lookback: Maximum steps before violation to analyze
        figsize: Figure size for individual plots
        model_distributions: Optional list of model predicted distributions for each walk
        graph: Optional graph object to compute edge distributions
        metric: Which metric to plot ('kl', 'js', 'tv', 'hellinger')
        output_dir: Directory to save plots
    
    Returns:
        Dict of figures by violation type
    """
    if not (model_distributions and graph):
        print("Model distributions and graph required for analysis")
        return None
    
    # Organize data by violation type
    metrics_by_violation_type = {
        'repeater': defaultdict(list),
        'ascender': defaultdict(list),
        'even': defaultdict(list),
        'graph': defaultdict(list)
    }
    
    for walk_idx, (walk, kl_divs, violations, _) in enumerate(walks_with_violations):
        if walk_idx >= len(model_distributions) or not violations:
            continue
        
        model_dists = model_distributions[walk_idx]
        
        # Process each violation by type
        for viol_pos, viol_type in violations.items():
            if viol_type not in metrics_by_violation_type:
                continue
                
            # Look back from violation position
            for steps_before in range(1, min(viol_pos + 1, max_lookback + 1)):
                pos = viol_pos - steps_before
                if pos >= 0 and pos < len(model_dists) and pos < len(walk) - 1:
                    current_vertex = walk[pos]
                    
                    # Get distributions
                    model_dist = model_dists[pos]
                    edge_dist = compute_edge_distribution(graph, current_vertex)
                    
                    if edge_dist is not None and len(model_dist) == len(edge_dist):
                        # Calculate requested metric
                        if metric == 'kl':
                            value = compute_kl_divergence(model_dist, edge_dist)
                        elif metric == 'js':
                            value = compute_js_divergence(model_dist, edge_dist)
                        elif metric == 'tv':
                            value = compute_total_variation(model_dist, edge_dist)
                        elif metric == 'hellinger':
                            value = compute_hellinger_distance(model_dist, edge_dist)
                        else:
                            continue
                        
                        metrics_by_violation_type[viol_type][steps_before].append(value)
    
    # Create plots for each violation type
    figures = {}
    metric_labels = {
        'kl': 'KL Divergence',
        'js': 'JS Divergence',
        'tv': 'Total Variation',
        'hellinger': 'Hellinger Distance'
    }
    
    colors = {
        'repeater': 'blue',
        'ascender': 'green',
        'even': 'purple',
        'graph': 'red'
    }
    
    for viol_type, metrics_by_steps in metrics_by_violation_type.items():
        if not metrics_by_steps:
            continue
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate means and std errors
        steps_before_list = sorted(metrics_by_steps.keys(), reverse=True)
        means = []
        stds = []
        
        for steps in steps_before_list:
            values = metrics_by_steps[steps]
            means.append(np.mean(values))
            stds.append(np.std(values) / np.sqrt(len(values)))
        
        # Plot
        color = colors[viol_type]
        ax.plot(steps_before_list, means, color=color, linewidth=2.5, 
                label=f'{viol_type.capitalize()} violations')
        ax.fill_between(steps_before_list,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.3, color=color)
        
        # Formatting
        ax.set_xlabel('Steps Before Violation (countdown)', fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f'{metric_labels[metric]}: Model vs Edge Distribution\n{viol_type.capitalize()} Violations', 
                    fontsize=14, fontweight='bold')
        
        # Add violation marker
        ax.axvline(x=1, color='red', linestyle='-', alpha=0.5, linewidth=2)
        ax.fill_betweenx([0, max(means) * 1.1], 0, 3, alpha=0.2, color='red')
        
        # Reference line
        ax.axhline(y=0, color='black', linestyle=':', alpha=0.3, label='Perfect alignment')
        
        # Add sample count
        sample_counts = [len(metrics_by_steps[s]) for s in steps_before_list]
        avg_samples = np.mean(sample_counts)
        ax.text(0.98, 0.02, f'Avg samples: {avg_samples:.0f}', 
                transform=ax.transAxes, fontsize=10, ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Trend annotation
        if len(means) > 5:
            early_mean = np.mean(means[:len(means)//3])
            late_mean = np.mean(means[-len(means)//3:])
            trend = "increasing" if late_mean > early_mean else "decreasing"
            percent_change = 100 * (late_mean - early_mean) / early_mean if early_mean > 0 else 0
            ax.text(0.02, 0.95, f'Trend: {trend} ({percent_change:+.1f}%)',
                    transform=ax.transAxes, fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.invert_xaxis()
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{output_dir}/{metric}_{viol_type}_violations.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        figures[viol_type] = fig
    
    return figures


def create_aggregated_violation_analysis(
    walks_with_violations: List[Tuple[List[int], List[float], Dict, Optional[List[float]]]],
    max_lookback: int = 20,
    figsize: Tuple[int, int] = (14, 8),
    model_distributions: Optional[List[List[np.ndarray]]] = None,
    graph = None
):
    """
    Create aggregated analysis of model vs edge distribution divergence before violations.
    
    Args:
        walks_with_violations: List of (walk, kl_divergences, violations, wasserstein_distances)
        max_lookback: Maximum steps before violation to analyze
        figsize: Figure size
        model_distributions: Optional list of model predicted distributions for each walk
        graph: Optional graph object to compute edge distributions
    
    Returns:
        Figure with aggregated metrics
    """
    # Create figure with 2 subplots for model vs edge distribution analysis
    if not (model_distributions and graph):
        print("Model distributions and graph required for analysis")
        return None
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1])
    
    # Process model vs edge distribution comparisons
    if model_distributions and graph:
        js_model_vs_edge_by_steps = defaultdict(list)
        tv_model_vs_edge_by_steps = defaultdict(list)
        hellinger_model_vs_edge_by_steps = defaultdict(list)
        kl_model_vs_edge_by_steps = defaultdict(list)
        
        for walk_idx, (walk, kl_divs, violations, _) in enumerate(walks_with_violations):
            if walk_idx >= len(model_distributions) or not violations:
                continue
            
            model_dists = model_distributions[walk_idx]
            
            # Process each violation
            for viol_pos in violations.keys():
                # Look back from violation position
                for steps_before in range(1, min(viol_pos + 1, max_lookback + 1)):
                    pos = viol_pos - steps_before
                    if pos >= 0 and pos < len(model_dists) and pos < len(walk) - 1:
                        current_vertex = walk[pos]
                        
                        # Get model distribution at this position (over all n nodes)
                        model_dist = model_dists[pos]
                        
                        # Compute edge distribution from graph (weights to all n nodes, 0 for non-edges)
                        edge_dist = compute_edge_distribution(graph, current_vertex)
                        
                        if edge_dist is not None and len(model_dist) == len(edge_dist):
                            # Calculate various divergence metrics
                            js_div = compute_js_divergence(model_dist, edge_dist)
                            js_model_vs_edge_by_steps[steps_before].append(js_div)
                            
                            tv_dist = compute_total_variation(model_dist, edge_dist)
                            tv_model_vs_edge_by_steps[steps_before].append(tv_dist)
                            
                            hellinger = compute_hellinger_distance(model_dist, edge_dist)
                            hellinger_model_vs_edge_by_steps[steps_before].append(hellinger)
                            
                            kl_div = compute_kl_divergence(model_dist, edge_dist)
                            kl_model_vs_edge_by_steps[steps_before].append(kl_div)
        
        # Plot model vs edge distribution divergence metrics
        if kl_model_vs_edge_by_steps:
            steps_before_list = sorted(kl_model_vs_edge_by_steps.keys(), reverse=True)
            kl_model_edge_means = []
            kl_model_edge_stds = []
            js_model_edge_means = []
            js_model_edge_stds = []
            tv_model_edge_means = []
            hellinger_model_edge_means = []
            
            for steps in steps_before_list:
                kl_values = kl_model_vs_edge_by_steps[steps]
                kl_model_edge_means.append(np.mean(kl_values))
                kl_model_edge_stds.append(np.std(kl_values) / np.sqrt(len(kl_values)))
                
                if steps in js_model_vs_edge_by_steps:
                    js_values = js_model_vs_edge_by_steps[steps]
                    js_model_edge_means.append(np.mean(js_values))
                    js_model_edge_stds.append(np.std(js_values) / np.sqrt(len(js_values)))
                
                if steps in tv_model_vs_edge_by_steps:
                    tv_model_edge_means.append(np.mean(tv_model_vs_edge_by_steps[steps]))
                    
                if steps in hellinger_model_vs_edge_by_steps:
                    hellinger_model_edge_means.append(np.mean(hellinger_model_vs_edge_by_steps[steps]))
            
            # PLOT 1: KL divergence between model and edge distributions
            color = 'purple'
            ax1.plot(steps_before_list, kl_model_edge_means, color=color, linewidth=2, 
                    label='KL(Model || Edge Dist)')
            ax1.fill_between(steps_before_list,
                            np.array(kl_model_edge_means) - np.array(kl_model_edge_stds),
                            np.array(kl_model_edge_means) + np.array(kl_model_edge_stds),
                            alpha=0.3, color=color)
            
            ax1.set_xlabel('Steps Before Violation (countdown)', fontsize=11)
            ax1.set_ylabel('KL(Model || Edge Distribution)', fontsize=11, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_title('Model Distribution vs Graph Edge Distribution (KL & JS Divergence)', fontsize=12, fontweight='bold')
            
            # Secondary axis for JS divergence on plot 1
            if js_model_edge_means:
                ax1_twin = ax1.twinx()
                ax1_twin.plot(steps_before_list, js_model_edge_means, 'orange', linestyle='--', 
                             linewidth=1.5, label='JS Divergence', alpha=0.7)
                ax1_twin.set_ylabel('JS Divergence', fontsize=11, color='orange')
                ax1_twin.tick_params(axis='y', labelcolor='orange')
                ax1_twin.set_ylim([0, max(js_model_edge_means) * 1.2 if js_model_edge_means else 1])
            
            # Add violation marker and reference lines for plot 1
            ax1.axvline(x=1, color='red', linestyle='-', alpha=0.5, linewidth=2)
            ax1.fill_betweenx([0, max(kl_model_edge_means) * 1.1], 0, 3, alpha=0.2, color='red')
            ax1.axhline(y=0, color='black', linestyle=':', alpha=0.3, label='Perfect alignment')
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.2)
            ax1.invert_xaxis()
            
            # Trend annotation for plot 1
            if len(kl_model_edge_means) > 5:
                early_mean = np.mean(kl_model_edge_means[:len(kl_model_edge_means)//3])
                late_mean = np.mean(kl_model_edge_means[-len(kl_model_edge_means)//3:])
                trend = "diverging from" if late_mean > early_mean else "converging to"
                ax1.text(0.02, 0.95, f"Model {trend} edge distribution near violations",
                        transform=ax1.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # PLOT 2: Multiple distance metrics between model and edge distributions
            if tv_model_edge_means and hellinger_model_edge_means:
                # Plot Total Variation distance
                ax2.plot(steps_before_list, tv_model_edge_means, 'darkblue', linewidth=2, 
                        label='Total Variation', alpha=0.8)
                
                # Plot Hellinger distance  
                ax2.plot(steps_before_list, hellinger_model_edge_means, 'green', linewidth=2,
                        label='Hellinger Distance', alpha=0.8)
                
                # Also plot JS divergence for comparison (all bounded [0,1])
                if js_model_edge_means:
                    ax2.plot(steps_before_list, js_model_edge_means, 'orange', linewidth=2,
                            label='JS Divergence', alpha=0.8)
                
                ax2.set_xlabel('Steps Before Violation (countdown)', fontsize=11)
                ax2.set_ylabel('Distance/Divergence', fontsize=11)
                ax2.set_title('Distribution Distance Metrics (Model vs Edge Distribution)', fontsize=12, fontweight='bold')
                ax2.set_ylim([0, max(max(tv_model_edge_means), max(hellinger_model_edge_means), 
                                    max(js_model_edge_means) if js_model_edge_means else 0) * 1.2])
                
                # Add violation marker and reference lines for plot 2
                ax2.axvline(x=1, color='red', linestyle='-', alpha=0.5, linewidth=2)
                ax2.fill_betweenx([0, ax2.get_ylim()[1]], 0, 3, alpha=0.2, color='red')
                ax2.axhline(y=0, color='black', linestyle=':', alpha=0.3, label='Perfect alignment')
                ax2.legend(loc='upper left', fontsize=9)
                ax2.grid(True, alpha=0.2)
                ax2.invert_xaxis()
                
                # Trend annotation for plot 2
                if len(tv_model_edge_means) > 5:
                    early_tv = np.mean(tv_model_edge_means[:len(tv_model_edge_means)//3])
                    late_tv = np.mean(tv_model_edge_means[-len(tv_model_edge_means)//3:])
                    early_hell = np.mean(hellinger_model_edge_means[:len(hellinger_model_edge_means)//3])
                    late_hell = np.mean(hellinger_model_edge_means[-len(hellinger_model_edge_means)//3:])
                    
                    tv_trend = "increasing" if late_tv > early_tv else "decreasing"
                    hell_trend = "increasing" if late_hell > early_hell else "decreasing"
                    
                    ax2.text(0.02, 0.95, f"TV {tv_trend}, Hellinger {hell_trend} near violations",
                            transform=ax2.transAxes, fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def compute_edge_distribution(graph, vertex):
    """Compute normalized edge weight distribution from a vertex."""
    if graph is None or not hasattr(graph, 'n'):
        return None
    
    try:
        n = graph.n
        
        # Get edge weights from adjacency matrix - all n vertices
        weights = np.zeros(n)
        for neighbor in range(n):
            if neighbor != vertex:
                weight = graph.adjacency[vertex, neighbor]
                weights[neighbor] = weight if weight > 0 else 0
        
        # Normalize to probability distribution
        total = weights.sum()
        if total > 0:
            return weights / total
        else:
            # Uniform over non-self neighbors if no weights
            weights = np.ones(n)
            weights[vertex] = 0
            return weights / weights.sum()
    except:
        return None


def compute_kl_divergence(p, q, epsilon=1e-10):
    """Compute KL divergence KL(p || q)."""
    # Add small epsilon to avoid log(0)
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    
    # Renormalize
    p = p / p.sum()
    q = q / q.sum()
    
    return np.sum(p * np.log(p / q))


def compute_distribution_alignment(p, q):
    """Compute cosine similarity between two distributions."""
    p = np.array(p)
    q = np.array(q)
    
    dot_product = np.dot(p, q)
    norm_p = np.linalg.norm(p)
    norm_q = np.linalg.norm(q)
    
    if norm_p == 0 or norm_q == 0:
        return 0
    
    return dot_product / (norm_p * norm_q)


def compute_js_divergence(p, q):
    """Compute Jensen-Shannon divergence (symmetric, bounded [0,1])."""
    p = np.array(p)
    q = np.array(q)
    
    # Ensure proper normalization
    p = p / p.sum()
    q = q / q.sum()
    
    # JS divergence is square of JS distance from scipy
    return jensenshannon(p, q) ** 2


def compute_total_variation(p, q):
    """Compute Total Variation distance (L1/2)."""
    p = np.array(p)
    q = np.array(q)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    return 0.5 * np.sum(np.abs(p - q))


def compute_hellinger_distance(p, q):
    """Compute Hellinger distance."""
    p = np.array(p)
    q = np.array(q)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Hellinger distance
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def compute_cross_entropy(p, q, epsilon=1e-10):
    """Compute cross entropy H(p, q)."""
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    return -np.sum(p * np.log(q))


def compute_entropy(p, epsilon=1e-10):
    """Compute Shannon entropy H(p)."""
    p = np.array(p) + epsilon
    p = p / p.sum()
    
    return -np.sum(p * np.log(p))