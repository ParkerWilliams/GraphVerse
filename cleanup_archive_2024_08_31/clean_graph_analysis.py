#!/usr/bin/env python3
"""
Clean Graph Analysis Script

Creates readable visualizations with:
1. Single repeater cycles with all vertices labeled  
2. Walk visualizations with clear walk output at top
3. Proper vertex labeling throughout

Usage:
    python clean_graph_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import json
import sys
import os
import glob
from pathlib import Path
from collections import defaultdict

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from graphverse.vis.graph_visualizer import GraphVisualizer
except ImportError:
    print("❌ Error: Could not import GraphVerse modules")
    sys.exit(1)


class CleanGraphVisualizer(GraphVisualizer):
    """Clean visualizer with proper labeling and walk output."""
    
    def __init__(self, graph, node_attributes, repeater_cycles):
        super().__init__(graph)
        self.node_attributes = node_attributes
        self.repeater_cycles = repeater_cycles
        self.rule_colors = {
            'ascender': '#FF4444',    # Bright Red
            'even': '#44AA44',        # Green  
            'repeater': '#FF8800',    # Orange
            'none': '#CCCCCC'         # Light Gray
        }
    
    def format_walk_output(self, walk, target_rule):
        """Format walk for clear display at top of analysis."""
        print(f"\n" + "="*80)
        print(f"WALK TARGETING {target_rule.upper()} NODES")
        print(f"="*80)
        
        # Basic walk info
        print(f"WALK: {' → '.join(map(str, walk))}")
        print(f"LENGTH: {len(walk)} steps")
        
        # Find target hits
        target_hits = []
        rule_counts = defaultdict(int)
        
        for i, node in enumerate(walk):
            node_attrs = self.node_attributes.get(str(node), {})
            rule = node_attrs.get('rule', 'none')
            rule_counts[rule] += 1
            
            if rule == target_rule:
                target_hits.append((i, node))
        
        print(f"TARGET HITS: {len(target_hits)} {target_rule} nodes")
        if target_hits:
            hit_list = [f"step {step}→node {node}" for step, node in target_hits]
            print(f"POSITIONS: {', '.join(hit_list)}")
        
        # Rule summary
        print(f"RULE COUNTS: ", end="")
        rule_parts = []
        for rule, count in sorted(rule_counts.items()):
            rule_parts.append(f"{rule}({count})")
        print(", ".join(rule_parts))
        
        # Annotated walk (show rules inline)
        print(f"\nANNOTATED WALK:")
        annotated = []
        for node in walk:
            node_attrs = self.node_attributes.get(str(node), {})
            rule = node_attrs.get('rule', 'none')
            if rule == target_rule:
                annotated.append(f"{node}★{rule.upper()}")
            elif rule != 'none':
                annotated.append(f"{node}•{rule}")
            else:
                annotated.append(str(node))
        
        # Print in chunks for readability
        chunk_size = 10
        for i in range(0, len(annotated), chunk_size):
            chunk = annotated[i:i + chunk_size]
            print(f"  {' → '.join(chunk)}")
        
        print(f"="*80)
        return target_hits
    
    def format_cycle_output(self, repeater_node, cycle_idx=0):
        """Format cycle information for clear display."""
        cycles = self.repeater_cycles.get(str(repeater_node), [])
        if cycle_idx >= len(cycles):
            print(f"❌ Cycle {cycle_idx} not found for repeater {repeater_node}")
            return None
        
        cycle = cycles[cycle_idx]
        attrs = self.node_attributes.get(str(repeater_node), {})
        k_value = attrs.get('repetitions', 2)
        
        print(f"\n" + "="*80)
        print(f"REPEATER CYCLE k={k_value}")
        print(f"="*80)
        print(f"REPEATER NODE: {repeater_node}")
        print(f"CYCLE: {' → '.join(map(str, cycle))} → {cycle[0]}")
        print(f"LENGTH: {len(cycle)} nodes")
        
        # Show rule for each node in cycle
        print(f"\nCYCLE NODES:")
        for i, node in enumerate(cycle):
            node_attrs = self.node_attributes.get(str(node), {})
            rule = node_attrs.get('rule', 'none').upper()
            print(f"  {node}: {rule}")
        
        print(f"="*80)
        return cycle, k_value
    
    def draw_cycle_clean(self, repeater_node, cycle_idx=0, figsize=(12, 10)):
        """Draw clean cycle visualization with all vertices labeled."""
        cycle_info = self.format_cycle_output(repeater_node, cycle_idx)
        if cycle_info is None:
            return None, None
        
        cycle, k_value = cycle_info
        
        # Use circular layout
        positions = self.circular_layout()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw cycle edges with thick red lines
        cycle_lines = []
        for i in range(len(cycle)):
            curr_node = cycle[i]
            next_node = cycle[(i + 1) % len(cycle)]
            x1, y1 = positions[curr_node]
            x2, y2 = positions[next_node]
            cycle_lines.append([(x1, y1), (x2, y2)])
        
        if cycle_lines:
            line_collection = LineCollection(cycle_lines, linewidths=5.0,
                                           colors='red', alpha=0.9, zorder=2)
            ax.add_collection(line_collection)
        
        # Draw ALL nodes with labels
        node_radius = 1.2  # Scale with larger layout
        cycle_set = set(cycle)
        
        for i in range(self.graph.n):
            x, y = positions[i]
            attrs = self.node_attributes.get(str(i), {})
            rule = attrs.get('rule', 'none')
            color = self.rule_colors[rule]
            
            # Highlight cycle nodes
            if i in cycle_set:
                edge_color = 'darkred'
                linewidth = 4
                alpha = 1.0
                radius = node_radius * 1.2
                text_color = 'white'
                fontweight = 'bold'
                fontsize = 10
            else:
                edge_color = 'gray'
                linewidth = 1
                alpha = 0.7
                radius = node_radius * 0.8
                text_color = 'black'
                fontweight = 'normal'
                fontsize = 8
            
            circle = patches.Circle((x, y), radius=radius,
                                  facecolor=color, edgecolor=edge_color,
                                  linewidth=linewidth, alpha=alpha, zorder=3)
            ax.add_patch(circle)
            
            # Label ALL nodes
            ax.text(x, y, str(i), ha='center', va='center',
                   fontsize=fontsize, fontweight=fontweight, 
                   color=text_color, zorder=4)
        
        # Set plot bounds with good margins - larger for 100 nodes
        ax.set_xlim(-45, 45)
        ax.set_ylim(-45, 45)
        ax.set_aspect('equal')
        
        title = f"Repeater Cycle k={k_value} from Node {repeater_node} (length={len(cycle)})"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=40)
        ax.axis('off')
        
        # Add complete cycle text at top of graph
        cycle_text = f"CYCLE: {' → '.join(map(str, cycle))} → {cycle[0]}"
        ax.text(0, 48, cycle_text, ha='center', va='top',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.9),
               transform=ax.transData, zorder=10)
        
        # Comprehensive legend
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=5, label=f'k={k_value} cycle path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444',
                   markersize=14, markeredgecolor='darkred', markeredgewidth=2, 
                   label='Ascender nodes'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#44AA44',
                   markersize=14, markeredgecolor='darkred', markeredgewidth=2,
                   label='Even nodes'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8800',
                   markersize=14, markeredgecolor='darkred', markeredgewidth=2,
                   label='Repeater nodes'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#CCCCCC',
                   markersize=12, markeredgecolor='gray', markeredgewidth=1,
                   label='Regular nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0.02, 0.98), fontsize=12)
        
        plt.tight_layout()
        return fig, ax
    
    def draw_walk_clean(self, walk, target_rule, figsize=(14, 12)):
        """Draw clean walk visualization with all vertices labeled."""
        target_hits = self.format_walk_output(walk, target_rule)
        
        # Use circular layout
        positions = self.circular_layout()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw walk path with thick blue lines
        walk_lines = []
        for i in range(len(walk) - 1):
            u, v = walk[i], walk[i + 1]
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            walk_lines.append([(x1, y1), (x2, y2)])
        
        if walk_lines:
            line_collection = LineCollection(walk_lines, linewidths=4.0,
                                           colors='blue', alpha=0.8, zorder=2)
            ax.add_collection(line_collection)
        
        # Draw ALL nodes with labels  
        node_radius = 1.0  # Scale with larger layout
        walk_set = set(walk)
        target_nodes = set([node for _, node in target_hits])
        
        for i in range(self.graph.n):
            x, y = positions[i]
            attrs = self.node_attributes.get(str(i), {})
            rule = attrs.get('rule', 'none')
            
            if i in walk_set:
                if i in target_nodes:
                    # Target nodes - bright with star
                    color = self.rule_colors[target_rule]
                    edge_color = 'darkred'
                    linewidth = 4
                    alpha = 1.0
                    radius = node_radius * 1.3
                    text_color = 'white'
                    fontweight = 'bold'
                    fontsize = 11
                elif rule != 'none':
                    # Other rule nodes in walk
                    color = self.rule_colors[rule]
                    edge_color = 'darkblue'
                    linewidth = 3
                    alpha = 0.9
                    radius = node_radius * 1.1
                    text_color = 'white'
                    fontweight = 'bold'
                    fontsize = 9
                else:
                    # Regular walk nodes
                    color = '#BBDDFF'
                    edge_color = 'blue'
                    linewidth = 2
                    alpha = 0.8
                    radius = node_radius
                    text_color = 'black'
                    fontweight = 'normal'
                    fontsize = 8
            else:
                # Non-walk nodes - very faded but still visible
                color = '#F0F0F0'
                edge_color = 'lightgray'
                linewidth = 1
                alpha = 0.4
                radius = node_radius * 0.7
                text_color = 'gray'
                fontweight = 'normal'
                fontsize = 7
            
            circle = patches.Circle((x, y), radius=radius,
                                  facecolor=color, edgecolor=edge_color,
                                  linewidth=linewidth, alpha=alpha, zorder=3)
            ax.add_patch(circle)
            
            # Label ALL nodes
            ax.text(x, y, str(i), ha='center', va='center',
                   fontsize=fontsize, fontweight=fontweight, 
                   color=text_color, zorder=4)
        
        # Add start/end markers  
        if len(walk) > 0:
            # Start node - green S
            start_x, start_y = positions[walk[0]]
            ax.text(start_x, start_y + 0.8, 'START', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='green',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                   zorder=5)
            
            # End node - red E
            end_x, end_y = positions[walk[-1]]
            ax.text(end_x, end_y + 0.8, 'END', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8),
                   zorder=5)
        
        # Set plot bounds - larger for 100 nodes
        ax.set_xlim(-45, 45)
        ax.set_ylim(-45, 45)
        ax.set_aspect('equal')
        
        title = f"Walk Targeting {target_rule.title()} Nodes (length={len(walk)}, hits={len(target_hits)})"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=60)
        ax.axis('off')
        
        # Add complete walk text at top of graph
        walk_text = f"WALK: {' → '.join(map(str, walk))}"
        
        # Split long walk into multiple lines if needed
        if len(walk_text) > 120:
            # Split into chunks for display
            walk_parts = walk_text.split(' → ')
            lines = []
            current_line = "WALK: "
            
            for i, part in enumerate(walk_parts):
                if i == 0:
                    current_line += part
                else:
                    test_line = current_line + " → " + part
                    if len(test_line) > 80:  # Start new line
                        lines.append(current_line)
                        current_line = "      " + part  # Indent continuation
                    else:
                        current_line = test_line
            
            if current_line.strip():
                lines.append(current_line)
            
            # Display multiple lines
            for i, line in enumerate(lines):
                ax.text(0, 50 - i*4, line, ha='center', va='top',
                       fontsize=10, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
                       transform=ax.transData, zorder=10)
        else:
            # Single line display
            ax.text(0, 50, walk_text, ha='center', va='top',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
                   transform=ax.transData, zorder=10)
        
        # Add target hits info
        if target_hits:
            hits_text = f"TARGET HITS: {', '.join([f'step {pos}→node {node}' for pos, node in target_hits])}"
            if len(hits_text) > 80:
                hits_text = f"TARGET HITS: {len(target_hits)} {target_rule} nodes at steps {[pos for pos, _ in target_hits]}"
            
            ax.text(0, 42, hits_text, ha='center', va='top',
                   fontsize=9, fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9),
                   transform=ax.transData, zorder=10)
        
        # Comprehensive legend
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=4, label='Walk path'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=self.rule_colors[target_rule],
                   markersize=16, markeredgecolor='darkred', markeredgewidth=2,
                   label=f'{target_rule.title()} targets ★'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#BBDDFF',
                   markersize=12, markeredgecolor='blue', markeredgewidth=2,
                   label='Other walk nodes'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen',
                   markersize=10, markeredgecolor='green', markeredgewidth=1,
                   label='Start/End markers'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#F0F0F0',
                   markersize=8, markeredgecolor='lightgray', markeredgewidth=1,
                   label='Non-walk nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0.02, 0.98), fontsize=11)
        
        plt.tight_layout()
        return fig, ax


def load_graph_data():
    """Load graph data from files."""
    try:
        adjacency = np.load("small_graph_100.npy")
        with open("small_graph_100_attrs.json", 'r') as f:
            attrs_data = json.load(f)
        return adjacency, attrs_data
    except FileNotFoundError as e:
        print(f"❌ Error loading graph data: {e}")
        sys.exit(1)


def create_graph_object(adjacency):
    """Create a Graph object from adjacency matrix."""
    class SimpleGraph:
        def __init__(self, adj_matrix):
            self.adjacency = adj_matrix
            self.n = adj_matrix.shape[0]
            
        def get_degree(self, node):
            return np.sum(self.adjacency[node] > 0)
    
    return SimpleGraph(adjacency)


def load_walks_data():
    """Load walks data from available files."""
    # Try to load valid rule-following walks first
    try:
        ascender_walks = list(np.load('small_results/valid_ascender_walks.npy'))
        even_walks = list(np.load('small_results/valid_even_walks.npy'))
        print(f"✅ Loaded {len(ascender_walks)} valid ascender walks and {len(even_walks)} valid even walks")
        return ascender_walks + even_walks
    except:
        print("❌ No valid walk files found")
        return []


def main():
    print("🎨 CLEAN GRAPH ANALYSIS")
    print("="*60)
    
    # Load data
    print("📊 Loading graph data...")
    adjacency, attrs_data = load_graph_data()
    graph = create_graph_object(adjacency)
    
    visualizer = CleanGraphVisualizer(
        graph, 
        attrs_data['node_attributes'],
        attrs_data['repeater_cycles']
    )
    
    print(f"📈 Graph: {graph.n} nodes, {np.sum(adjacency > 0)} edges")
    
    # 1. Create 3 cycle visualizations
    print(f"\n🔄 CREATING CYCLE VISUALIZATIONS")
    print(f"="*60)
    
    # Find different k-values
    k_repeaters = defaultdict(list)
    for node_id, attrs in attrs_data['node_attributes'].items():
        if attrs.get('rule') == 'repeater':
            k_value = attrs.get('repetitions', 2)
            k_repeaters[k_value].append(int(node_id))
    
    # Create 3 different cycles
    cycle_count = 0
    for k_value in sorted(k_repeaters.keys()):
        if cycle_count >= 3:
            break
        
        repeater_node = k_repeaters[k_value][0]
        
        fig, ax = visualizer.draw_cycle_clean(repeater_node)
        if fig:
            filename = f"cycle_k{k_value}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved: {filename}")
            cycle_count += 1
    
    # 2. Create walk visualizations
    print(f"\n🚶 CREATING WALK VISUALIZATIONS")
    print(f"="*60)
    
    walks = load_walks_data()
    
    if walks:
        # Load the specific valid walk types
        try:
            ascender_walks = list(np.load('small_results/valid_ascender_walks.npy'))[:2]
            even_walks = list(np.load('small_results/valid_even_walks.npy'))[:2]
        except:
            ascender_walks = []
            even_walks = []
        
        # Create ascender walk visualizations
        for i, walk in enumerate(ascender_walks):
            fig, ax = visualizer.draw_walk_clean(walk, 'ascender')
            filename = f"walk_ascender_{i+1}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved: {filename}")
        
        # Create even walk visualizations
        for i, walk in enumerate(even_walks):
            fig, ax = visualizer.draw_walk_clean(walk, 'even')
            filename = f"walk_even_{i+1}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved: {filename}")
    
    print(f"\n🎉 Clean analysis complete!")
    print(f"All visualizations have proper vertex labeling and clear output.")


if __name__ == "__main__":
    main()