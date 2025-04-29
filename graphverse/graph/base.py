import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import json
import os

class Graph:
    def __init__(self, n):
        self.n = n
        self.adjacency = np.zeros((n, n), dtype=np.float32)
        self.node_attributes = {}
        
    def add_edge(self, u, v, weight=1.0):
        self.adjacency[u, v] = weight
        self.adjacency[v, u] = -1*weight
        
    def get_neighbors(self, node):
        return np.where(self.adjacency[node] > 0)[0]
        
    def get_degree(self, node):
        return np.sum(self.adjacency[node] > 0)
        
    def get_laplacian(self):
        """Compute the graph Laplacian matrix"""
        # Only consider positive edges for out-degree
        D = np.diag(np.sum(self.adjacency > 0, axis=1))
        return D - self.adjacency
        
    def get_spectral_embedding(self, k=2):
        """Compute spectral embedding using k smallest non-zero eigenvectors"""
        L = self.get_laplacian()
        # Get k+1 smallest eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigs(L, k=k+1, which='SM')
        # Sort by eigenvalue
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Return k smallest non-zero eigenvectors
        return eigenvectors[:, 1:k+1]
        
    def is_connected(self):
        """Check if graph is connected using spectral properties"""
        L = self.get_laplacian()
        eigenvalues = eigs(L, k=2, which='SM')[0]
        return np.abs(eigenvalues[1]) > 1e-10  # Second smallest eigenvalue > 0
        
    def get_edge_probabilities(self, node):
        """Get transition probabilities for a node, only considering positive edges"""
        neighbors = self.get_neighbors(node)  # This already only gets positive edges
        if len(neighbors) == 0:
            return np.array([]), np.array([])
        weights = self.adjacency[node, neighbors]  # These weights are guaranteed positive
        probs = weights / np.sum(weights)
        return neighbors, probs

    def save_graph(self, filepath):
        """Save the graph to a file.
        
        Args:
            filepath (str): Path to save the graph data. Will create two files:
                           - filepath.npy for adjacency matrix
                           - filepath_attrs.json for node attributes
        """
        # Save adjacency matrix
        np.save(filepath, self.adjacency)
        
        # Save node attributes
        attrs_file = f"{filepath}_attrs.json"
        with open(attrs_file, 'w') as f:
            json.dump(self.node_attributes, f)
            
    @classmethod
    def load_graph(cls, filepath):
        """Load a graph from a file.
        
        Args:
            filepath (str): Path to the graph data files (without extensions)
            
        Returns:
            Graph: A new Graph instance with loaded data
        """
        # Load adjacency matrix
        adjacency = np.load(f"{filepath}.npy")
        n = adjacency.shape[0]
        
        # Create new graph instance
        graph = cls(n)
        graph.adjacency = adjacency
        
        # Load node attributes
        attrs_file = f"{filepath}_attrs.json"
        if os.path.exists(attrs_file):
            with open(attrs_file, 'r') as f:
                graph.node_attributes = json.load(f)
                
        return graph 

    def has_edge(self, u, v):
        """Return True if there is a positive edge from u to v."""
        return self.adjacency[u, v] > 0 