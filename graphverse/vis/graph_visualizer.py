import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import math


class GraphVisualizer:
    """
    A graph visualization tool that creates 2D layouts without external dependencies.
    Uses matplotlib for rendering and implements several layout algorithms.
    """
    
    def __init__(self, graph):
        """
        Initialize visualizer with a graph.
        
        Args:
            graph: Graph object with adjacency matrix and methods
        """
        self.graph = graph
        self.node_positions = None
        self.node_colors = None
        self.edge_colors = None
        
    def circular_layout(self):
        """Create a circular layout for nodes with better spacing."""
        n = self.graph.n
        positions = np.zeros((n, 2))
        
        # Scale radius based on number of nodes to prevent overlap
        # For 100 nodes, need much larger radius for proper spacing
        radius = max(5.0, n * 0.4)  # Minimum radius of 5, scale more aggressively
        
        for i in range(n):
            angle = 2 * np.pi * i / n
            positions[i] = [radius * np.cos(angle), radius * np.sin(angle)]
            
        self.node_positions = positions
        return positions
    
    def spring_layout(self, iterations=50, k=1.0, repulsion=0.1):
        """
        Simple spring-force layout algorithm with improved spacing.
        
        Args:
            iterations: Number of layout iterations
            k: Spring constant
            repulsion: Repulsion strength between non-connected nodes
        """
        n = self.graph.n
        
        # Initialize random positions in reasonable range
        np.random.seed(42)  # For reproducible layouts
        positions = np.random.random((n, 2)) * 2 - 1  # Start in [-1, 1] range
        
        # Conservative parameters to prevent explosion
        optimal_edge_length = 1.0
        repulsion_strength = 0.1
        max_force = 0.1  # Cap maximum force magnitude
        
        for iteration in range(iterations):
            forces = np.zeros((n, 2))
            
            # Calculate forces between all pairs of nodes
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist = max(0.01, np.sqrt(dx*dx + dy*dy))  # Prevent division by zero
                        
                        # Unit vector from i to j
                        ux, uy = dx / dist, dy / dist
                        
                        if self.graph.adjacency[i, j] > 0:  # Connected nodes
                            # Attractive spring force - only if too far apart
                            if dist > optimal_edge_length:
                                force_mag = min(k * (dist - optimal_edge_length) * 0.1, max_force)
                                forces[i, 0] += force_mag * ux
                                forces[i, 1] += force_mag * uy
                        
                        # Repulsive force for all pairs (prevents overlap)
                        if dist < optimal_edge_length * 2:
                            force_mag = min(repulsion_strength / (dist + 0.1), max_force)
                            forces[i, 0] -= force_mag * ux
                            forces[i, 1] -= force_mag * uy
            
            # Update positions with small step size and cooling
            cooling = max(0.1, 1.0 - (iteration / iterations) * 0.8)
            step_size = 0.01 * cooling  # Much smaller step size
            
            # Clamp forces to prevent explosion
            forces = np.clip(forces, -max_force, max_force)
            positions += forces * step_size
            
            # Keep positions in reasonable bounds
            positions = np.clip(positions, -10, 10)
        
        self.node_positions = positions
        return positions
    
    def spectral_layout(self):
        """Use spectral embedding for layout if available."""
        try:
            embedding = self.graph.get_spectral_embedding(k=2)
            self.node_positions = embedding.real
            return self.node_positions
        except:
            # Fallback to circular layout
            return self.circular_layout()
    
    def grid_layout(self):
        """Create a grid layout with better spacing."""
        n = self.graph.n
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # Scale spacing based on graph size
        spacing = max(1.5, np.sqrt(n) * 0.2)
        
        positions = np.zeros((n, 2))
        for i in range(n):
            row = i // cols
            col = i % cols
            positions[i] = [
                (col - (cols-1)/2) * spacing, 
                ((rows-1)/2 - row) * spacing
            ]
            
        self.node_positions = positions
        return positions
    
    def set_node_colors(self, colors=None, attribute=None):
        """
        Set node colors.
        
        Args:
            colors: List/array of colors for each node
            attribute: Node attribute name to use for coloring
        """
        if colors is not None:
            self.node_colors = colors
        elif attribute is not None and attribute in self.graph.node_attributes:
            # Color by attribute values
            attrs = [self.graph.node_attributes[attribute].get(str(i), 0) 
                    for i in range(self.graph.n)]
            self.node_colors = attrs
        else:
            # Default blue
            self.node_colors = ['skyblue'] * self.graph.n
    
    def draw(self, layout='spring', figsize=(10, 8), node_size=300, 
             edge_width=1.0, node_labels=True, title="Graph Visualization", show_legend=True):
        """
        Draw the graph with specified layout.
        
        Args:
            layout: Layout algorithm ('spring', 'circular', 'spectral', 'grid')
            figsize: Figure size tuple
            node_size: Size of nodes
            edge_width: Width of edges
            node_labels: Whether to show node labels
            title: Plot title
            show_legend: Whether to show legend for nodes
        """
        
        # Compute layout
        if layout == 'spring':
            positions = self.spring_layout()
        elif layout == 'circular':
            positions = self.circular_layout()
        elif layout == 'spectral':
            positions = self.spectral_layout()
        elif layout == 'grid':
            positions = self.grid_layout()
        else:
            positions = self.spring_layout()  # Default
        
        # Set default colors if not set
        if self.node_colors is None:
            self.set_node_colors()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw edges
        edge_lines = []
        for i in range(self.graph.n):
            for j in range(self.graph.n):
                if self.graph.adjacency[i, j] > 0:  # Only positive edges
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    edge_lines.append([(x1, y1), (x2, y2)])
        
        if edge_lines:
            line_collection = LineCollection(edge_lines, linewidths=edge_width, 
                                           colors='gray', alpha=0.6)
            ax.add_collection(line_collection)
        
        # Draw nodes and collect legend info
        node_handles = []
        node_labels_list = []
        
        for i in range(self.graph.n):
            x, y = positions[i]
            color = self.node_colors[i] if isinstance(self.node_colors[i], str) else 'skyblue'
            # Calculate appropriate radius based on plot scale and node count
            plot_range = max(positions.max() - positions.min(), 1.0)
            radius = min(0.15, plot_range / (np.sqrt(self.graph.n) * 8))  # Adaptive radius
            circle = Circle((x, y), radius=radius, 
                          facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
            
            # Collect for legend instead of adding text
            if node_labels and show_legend:
                node_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=color, markersize=8))
                node_labels_list.append(f'Node {i}')
        
        # Set plot properties with better margins
        plot_range = max(positions.max() - positions.min(), 1.0)
        margin = max(0.5, plot_range * 0.1)  # 10% margin, minimum 0.5
        ax.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
        ax.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add legend if requested and there are labels
        if show_legend and node_labels and node_handles:
            # Limit legend entries if too many nodes
            if len(node_handles) > 10:
                ax.legend(node_handles[:10], node_labels_list[:10], 
                         loc='center left', bbox_to_anchor=(1, 0.5),
                         title=f'Nodes (showing 10/{self.graph.n})')
            else:
                ax.legend(node_handles, node_labels_list, 
                         loc='center left', bbox_to_anchor=(1, 0.5),
                         title='Nodes')
        
        plt.tight_layout()
        return fig, ax
    
    def draw_walk(self, walk, layout='spring', figsize=(10, 8), 
                  highlight_color='red', title="Graph Walk Visualization"):
        """
        Draw the graph with a walk highlighted.
        
        Args:
            walk: List of node indices representing the walk
            layout: Layout algorithm
            figsize: Figure size
            highlight_color: Color for walk edges
            title: Plot title
        """
        fig, ax = self.draw(layout=layout, figsize=figsize, title=title, show_legend=False)
        
        # Highlight walk edges
        if len(walk) > 1:
            walk_lines = []
            for i in range(len(walk) - 1):
                u, v = walk[i], walk[i + 1]
                x1, y1 = self.node_positions[u]
                x2, y2 = self.node_positions[v]
                walk_lines.append([(x1, y1), (x2, y2)])
            
            if walk_lines:
                walk_collection = LineCollection(walk_lines, linewidths=3.0, 
                                               colors=highlight_color, alpha=0.8)
                ax.add_collection(walk_collection)
        
        # Highlight walk nodes
        walk_handles = []
        walk_labels = []
        plot_range = max(self.node_positions.max() - self.node_positions.min(), 1.0)
        walk_radius = min(0.2, plot_range / (np.sqrt(self.graph.n) * 6))  # Slightly larger than regular nodes
        
        for i, node in enumerate(walk):
            x, y = self.node_positions[node]
            circle = Circle((x, y), radius=walk_radius, 
                          facecolor=highlight_color, edgecolor='darkred', 
                          linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            
            # Collect for legend instead of adding text
            if i == 0:
                walk_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=highlight_color, markersize=10))
                walk_labels.append(f'Start: Node {node}')
            elif i == len(walk) - 1:
                walk_handles.append(plt.Line2D([0], [0], marker='s', color='w',
                                              markerfacecolor=highlight_color, markersize=10))
                walk_labels.append(f'End: Node {node}')
        
        # Add walk sequence to legend
        walk_sequence = ' → '.join([str(n) for n in walk[:10]])
        if len(walk) > 10:
            walk_sequence += f'... ({len(walk)} nodes total)'
        
        # Add legend with walk information
        if walk_handles:
            ax.legend(walk_handles, walk_labels, loc='upper left', title=f'Walk: {walk_sequence}')
        
        return fig, ax
    
    def save_visualization(self, filename, layout='spring', **kwargs):
        """Save visualization to file."""
        fig, ax = self.draw(layout=layout, **kwargs)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    def show_graph_stats(self):
        """Display basic graph statistics."""
        n = self.graph.n
        edges = np.sum(self.graph.adjacency > 0)
        density = edges / (n * (n - 1)) if n > 1 else 0
        
        degrees = [self.graph.get_degree(i) for i in range(n)]
        avg_degree = np.mean(degrees)
        
        print(f"Graph Statistics:")
        print(f"  Nodes: {n}")
        print(f"  Edges: {edges}")
        print(f"  Density: {density:.3f}")
        print(f"  Average Degree: {avg_degree:.2f}")
        print(f"  Max Degree: {max(degrees) if degrees else 0}")
        print(f"  Min Degree: {min(degrees) if degrees else 0}")