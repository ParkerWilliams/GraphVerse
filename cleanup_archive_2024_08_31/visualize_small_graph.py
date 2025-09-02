#!/usr/bin/env python3
"""
GraphVerse Small Graph Visualization Script

Creates comprehensive visualizations of the small 100-node graph with:
- Rule-based node coloring (ascenders, evens, repeaters, regular)
- Repeater cycle highlighting with distinct colors
- Multiple layout algorithms for optimal viewing
- Detailed legends and statistics

Usage:
    python visualize_small_graph.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import json
import sys
import os
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from graphverse.graph.base import Graph
    from graphverse.vis.graph_visualizer import GraphVisualizer
except ImportError:
    print("❌ Error: Could not import GraphVerse modules")
    print("Make sure you're running this from the GraphVerse root directory")
    sys.exit(1)


class EnhancedGraphVisualizer(GraphVisualizer):
    """Enhanced visualizer with rule-aware coloring and cycle highlighting."""
    
    def __init__(self, graph, node_attributes, repeater_cycles):
        """
        Initialize enhanced visualizer.
        
        Args:
            graph: Graph object
            node_attributes: Dict of node attributes from JSON
            repeater_cycles: Dict of repeater cycles from JSON
        """
        super().__init__(graph)
        self.node_attributes = node_attributes
        self.repeater_cycles = repeater_cycles
        self.rule_colors = self._define_rule_colors()
        self.k_colors = self._define_k_colors()
        
    def _define_rule_colors(self):
        """Define color scheme for different rule types."""
        return {
            'ascender': '#FF6B6B',    # Red
            'even': '#4ECDC4',        # Teal
            'repeater': '#FFA500',    # Orange (base, will be modified by k-value)
            'none': '#E0E0E0'         # Light gray
        }
    
    def _define_k_colors(self):
        """Define color gradient for different k-values."""
        # Orange to Purple gradient for k=2 to k=12
        colors = [
            '#FFA500',  # k=2: Orange
            '#FF8C00',  # k=4: Dark Orange  
            '#FF7F50',  # k=6: Coral
            '#FF6347',  # k=7: Tomato
            '#FF4500',  # k=9: Orange Red
            '#DC143C',  # k=10: Crimson
            '#800080'   # k=12: Purple
        ]
        k_values = [2, 4, 6, 7, 9, 10, 12]
        return dict(zip(k_values, colors))
    
    def get_node_color(self, node_id):
        """Get color for a specific node based on its rule."""
        attrs = self.node_attributes.get(str(node_id), {})
        rule = attrs.get('rule', 'none')
        
        if rule == 'repeater':
            # Use k-value specific color for repeaters
            k_value = attrs.get('repetitions', 2)
            return self.k_colors.get(k_value, self.rule_colors['repeater'])
        else:
            return self.rule_colors[rule]
    
    def set_rule_based_colors(self):
        """Set node colors based on rule types."""
        colors = []
        for i in range(self.graph.n):
            colors.append(self.get_node_color(i))
        self.node_colors = colors
    
    def draw_with_rules(self, layout='spring', figsize=(16, 12), node_size=800, 
                       edge_width=0.5, title="GraphVerse Small Graph with Rules",
                       show_cycles=True, show_legend=True):
        """
        Draw graph with rule-based coloring and optional cycle highlighting.
        
        Args:
            layout: Layout algorithm
            figsize: Figure size
            node_size: Size of nodes
            edge_width: Width of regular edges
            title: Plot title
            show_cycles: Whether to highlight repeater cycles
            show_legend: Whether to show legend
        """
        # Set rule-based colors
        self.set_rule_based_colors()
        
        # Compute layout with improved parameters for 100-node graph
        if layout == 'spring':
            positions = self.spring_layout(iterations=200, k=1.5, repulsion=0.5)
        elif layout == 'circular':
            positions = self.circular_layout()
        elif layout == 'spectral':
            positions = self.spectral_layout()
        elif layout == 'grid':
            positions = self.grid_layout()
        else:
            positions = self.spring_layout(iterations=200, k=1.5, repulsion=0.5)
        
        # Create figure with extra space for legend
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw regular edges first (lighter)
        edge_lines = []
        for i in range(self.graph.n):
            for j in range(self.graph.n):
                if self.graph.adjacency[i, j] > 0:
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    edge_lines.append([(x1, y1), (x2, y2)])
        
        if edge_lines:
            line_collection = LineCollection(edge_lines, linewidths=edge_width, 
                                           colors='lightgray', alpha=0.3, zorder=1)
            ax.add_collection(line_collection)
        
        # Draw repeater cycles if requested
        if show_cycles:
            self._draw_repeater_cycles(ax, positions)
        
        # Draw nodes
        for i in range(self.graph.n):
            x, y = positions[i]
            color = self.node_colors[i]
            
            # Calculate appropriate radius and font size based on plot scale
            plot_range = max(positions.max() - positions.min(), 1.0)
            radius = min(0.25, plot_range / (np.sqrt(self.graph.n) * 6))  # Adaptive radius
            font_size = max(6, min(10, int(radius * 30)))  # Scale font with radius
            
            # Node circle
            circle = patches.Circle((x, y), radius=radius, 
                                  facecolor=color, edgecolor='black', 
                                  linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            
            # Node label - only show if radius is large enough
            if radius > 0.1:
                ax.text(x, y, str(i), ha='center', va='center', 
                       fontsize=font_size, fontweight='bold', zorder=4)
        
        # Set plot properties with adaptive margins
        plot_range = max(positions.max() - positions.min(), 1.0)
        margin = max(0.5, plot_range * 0.15)  # 15% margin, minimum 0.5
        ax.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
        ax.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        if show_legend:
            self._add_comprehensive_legend(ax)
        
        plt.tight_layout()
        return fig, ax
    
    def _draw_repeater_cycles(self, ax, positions):
        """Draw repeater cycles with distinct colors."""
        cycle_width = 3.0
        alpha = 0.7
        
        for repeater_node, cycles in self.repeater_cycles.items():
            repeater_id = int(repeater_node)
            attrs = self.node_attributes.get(repeater_node, {})
            k_value = attrs.get('repetitions', 2)
            
            # Get color for this k-value
            cycle_color = self.k_colors.get(k_value, '#FFA500')
            
            # Draw each cycle for this repeater
            for cycle_idx, cycle in enumerate(cycles):
                cycle_lines = []
                for i in range(len(cycle)):
                    curr_node = cycle[i]
                    next_node = cycle[(i + 1) % len(cycle)]  # Wrap around for cycles
                    
                    x1, y1 = positions[curr_node]
                    x2, y2 = positions[next_node]
                    cycle_lines.append([(x1, y1), (x2, y2)])
                
                if cycle_lines:
                    # Vary alpha slightly for different cycles of same repeater
                    cycle_alpha = alpha - (cycle_idx * 0.1)
                    line_collection = LineCollection(cycle_lines, linewidths=cycle_width,
                                                   colors=cycle_color, alpha=cycle_alpha, 
                                                   zorder=2)
                    ax.add_collection(line_collection)
    
    def _add_comprehensive_legend(self, ax):
        """Add comprehensive legend showing rules and k-values."""
        legend_elements = []
        
        # Rule type legend
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=self.rule_colors['ascender'], 
                                    markersize=12, label='Ascender nodes (10)'))
        
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=self.rule_colors['even'],
                                    markersize=12, label='Even nodes (15)'))
        
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=self.rule_colors['none'],
                                    markersize=12, label='Regular nodes (60)'))
        
        # Add separator
        legend_elements.append(Line2D([0], [0], color='white', alpha=0, label=''))
        legend_elements.append(Line2D([0], [0], color='black', alpha=0, 
                                    label='Repeater nodes (15):'))
        
        # K-value specific repeater legend
        k_counts = self._count_repeater_k_values()
        for k_value in sorted(k_counts.keys()):
            count = k_counts[k_value]
            color = self.k_colors.get(k_value, self.rule_colors['repeater'])
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=12,
                                        label=f'  k={k_value} repeaters ({count})'))
        
        # Add cycle legend
        legend_elements.append(Line2D([0], [0], color='white', alpha=0, label=''))
        legend_elements.append(Line2D([0], [0], color='black', linewidth=3, alpha=0.7,
                                    label='Repeater cycles'))
        
        # Position legend outside plot area
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                 title='Graph Rules & Structure', title_fontsize=12, fontsize=10,
                 frameon=True, fancybox=True, shadow=True)
    
    def _count_repeater_k_values(self):
        """Count how many repeaters have each k-value."""
        k_counts = {}
        for node_id in range(self.graph.n):
            attrs = self.node_attributes.get(str(node_id), {})
            if attrs.get('rule') == 'repeater':
                k_value = attrs.get('repetitions', 2)
                k_counts[k_value] = k_counts.get(k_value, 0) + 1
        return k_counts
    
    def generate_statistics_summary(self):
        """Generate comprehensive statistics about the graph."""
        stats = {}
        
        # Basic graph stats
        n = self.graph.n
        edges = np.sum(self.graph.adjacency > 0)
        density = edges / (n * (n - 1)) if n > 1 else 0
        
        # Rule distribution
        rule_counts = {'ascender': 0, 'even': 0, 'repeater': 0, 'none': 0}
        k_value_counts = {}
        
        for i in range(n):
            attrs = self.node_attributes.get(str(i), {})
            rule = attrs.get('rule', 'none')
            rule_counts[rule] += 1
            
            if rule == 'repeater':
                k_value = attrs.get('repetitions', 2)
                k_value_counts[k_value] = k_value_counts.get(k_value, 0) + 1
        
        # Cycle statistics
        total_cycles = sum(len(cycles) for cycles in self.repeater_cycles.values())
        avg_cycle_length = np.mean([
            len(cycle) for cycles in self.repeater_cycles.values() 
            for cycle in cycles
        ]) if total_cycles > 0 else 0
        
        stats.update({
            'nodes': n,
            'edges': edges,
            'density': density,
            'rule_distribution': rule_counts,
            'k_value_distribution': k_value_counts,
            'total_repeater_cycles': total_cycles,
            'average_cycle_length': avg_cycle_length
        })
        
        return stats


def load_graph_data(graph_path="small_graph_100"):
    """Load graph data from files."""
    try:
        # Load adjacency matrix
        adjacency = np.load(f"{graph_path}.npy")
        
        # Load attributes
        with open(f"{graph_path}_attrs.json", 'r') as f:
            attrs_data = json.load(f)
        
        # Load rules
        with open(f"{graph_path}_rules.json", 'r') as f:
            rules_data = json.load(f)
        
        return adjacency, attrs_data, rules_data
        
    except FileNotFoundError as e:
        print(f"❌ Error loading graph data: {e}")
        print("Make sure the graph files exist:")
        print(f"  - {graph_path}.npy")
        print(f"  - {graph_path}_attrs.json") 
        print(f"  - {graph_path}_rules.json")
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


def main():
    """Main visualization function."""
    print("🎨 GraphVerse Small Graph Visualization")
    print("=" * 50)
    
    # Load graph data
    print("📊 Loading graph data...")
    adjacency, attrs_data, rules_data = load_graph_data()
    
    # Create graph object
    graph = create_graph_object(adjacency)
    
    # Create enhanced visualizer
    visualizer = EnhancedGraphVisualizer(
        graph, 
        attrs_data['node_attributes'],
        attrs_data['repeater_cycles']
    )
    
    # Generate statistics
    print("📈 Generating statistics...")
    stats = visualizer.generate_statistics_summary()
    
    print(f"\n📊 Graph Statistics:")
    print(f"   Nodes: {stats['nodes']}")
    print(f"   Edges: {stats['edges']}")
    print(f"   Density: {stats['density']:.3f}")
    print(f"   Rule Distribution:")
    for rule, count in stats['rule_distribution'].items():
        print(f"     {rule}: {count}")
    print(f"   K-value Distribution:")
    for k, count in sorted(stats['k_value_distribution'].items()):
        print(f"     k={k}: {count} repeaters")
    print(f"   Total Repeater Cycles: {stats['total_repeater_cycles']}")
    print(f"   Average Cycle Length: {stats['average_cycle_length']:.1f}")
    
    # Create visualizations with different layouts
    layouts = ['spring', 'circular', 'spectral']
    
    for layout in layouts:
        print(f"\n🎨 Generating {layout} layout...")
        
        fig, ax = visualizer.draw_with_rules(
            layout=layout,
            figsize=(20, 14),
            node_size=1000,
            title=f"GraphVerse Small Graph - {layout.title()} Layout",
            show_cycles=True,
            show_legend=True
        )
        
        # Save high-resolution images
        output_file = f"small_graph_{layout}_layout"
        
        # PNG for viewing
        plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"   ✅ Saved: {output_file}.png")
        
        # PDF for publication
        plt.savefig(f"{output_file}.pdf", dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"   ✅ Saved: {output_file}.pdf")
        
        plt.close(fig)
    
    # Create a summary visualization with all layouts
    print(f"\n🎨 Creating summary visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle("GraphVerse Small Graph - Multiple Layout Comparison", 
                fontsize=20, fontweight='bold', y=0.95)
    
    for idx, layout in enumerate(layouts):
        # Set current axis
        plt.sca(axes[idx])
        
        # Draw without legend (will add one master legend)
        visualizer.draw_with_rules(
            layout=layout,
            figsize=(10, 8),
            node_size=600,
            title=f"{layout.title()} Layout",
            show_cycles=True,
            show_legend=(idx == 2)  # Only show legend on last plot
        )
    
    plt.tight_layout()
    
    # Save summary
    plt.savefig("small_graph_summary.png", dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig("small_graph_summary.pdf", dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"   ✅ Saved: small_graph_summary.png")
    print(f"   ✅ Saved: small_graph_summary.pdf")
    
    plt.close(fig)
    
    print("\n🎉 Visualization complete!")
    print("Generated files:")
    for layout in layouts:
        print(f"   • small_graph_{layout}_layout.png")
        print(f"   • small_graph_{layout}_layout.pdf")
    print(f"   • small_graph_summary.png")
    print(f"   • small_graph_summary.pdf")


if __name__ == "__main__":
    main()