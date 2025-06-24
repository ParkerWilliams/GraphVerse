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
        """Create a circular layout for nodes."""
        n = self.graph.n
        positions = np.zeros((n, 2))
        
        for i in range(n):
            angle = 2 * np.pi * i / n
            positions[i] = [np.cos(angle), np.sin(angle)]
            
        self.node_positions = positions
        return positions
    
    def spring_layout(self, iterations=50, k=1.0, repulsion=0.1):
        """
        Simple spring-force layout algorithm.
        
        Args:
            iterations: Number of layout iterations
            k: Spring constant
            repulsion: Repulsion strength between non-connected nodes
        """
        n = self.graph.n
        
        # Initialize random positions
        positions = np.random.random((n, 2)) * 2 - 1
        
        for _ in range(iterations):
            forces = np.zeros((n, 2))
            
            # Spring forces between connected nodes
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist > 0:
                            if self.graph.adjacency[i, j] > 0:  # Connected nodes
                                # Attractive force
                                force_mag = k * (dist - 1.0)
                                forces[i, 0] += force_mag * dx / dist
                                forces[i, 1] += force_mag * dy / dist
                            else:  # Unconnected nodes
                                # Repulsive force
                                force_mag = repulsion / (dist * dist)
                                forces[i, 0] -= force_mag * dx / dist
                                forces[i, 1] -= force_mag * dy / dist
            
            # Update positions
            positions += forces * 0.1
            
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
        """Create a grid layout for small graphs."""
        n = self.graph.n
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        positions = np.zeros((n, 2))
        for i in range(n):
            row = i // cols
            col = i % cols
            positions[i] = [col - (cols-1)/2, (rows-1)/2 - row]
            
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
             edge_width=1.0, node_labels=True, title="Graph Visualization"):
        """
        Draw the graph with specified layout.
        
        Args:
            layout: Layout algorithm ('spring', 'circular', 'spectral', 'grid')
            figsize: Figure size tuple
            node_size: Size of nodes
            edge_width: Width of edges
            node_labels: Whether to show node labels
            title: Plot title
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
        
        # Draw nodes
        for i in range(self.graph.n):
            x, y = positions[i]
            color = self.node_colors[i] if isinstance(self.node_colors[i], str) else 'skyblue'
            circle = Circle((x, y), radius=node_size/10000, 
                          facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
            
            # Add node labels
            if node_labels:
                ax.text(x, y, str(i), ha='center', va='center', 
                       fontsize=8, fontweight='bold')
        
        # Set plot properties
        ax.set_xlim(positions[:, 0].min() - 0.2, positions[:, 0].max() + 0.2)
        ax.set_ylim(positions[:, 1].min() - 0.2, positions[:, 1].max() + 0.2)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
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
        fig, ax = self.draw(layout=layout, figsize=figsize, title=title)
        
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
        for i, node in enumerate(walk):
            x, y = self.node_positions[node]
            circle = Circle((x, y), radius=400/10000, 
                          facecolor=highlight_color, edgecolor='darkred', 
                          linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            
            # Add step numbers
            ax.text(x, y + 0.1, f"#{i}", ha='center', va='center', 
                   fontsize=6, color='white', fontweight='bold')
        
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