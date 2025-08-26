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
        self.edge_count = 0  # Track number of undirected edges
        self.repeater_cycles = {}  # Store k-cycle paths for repeater nodes
        
    def add_edge(self, u, v, weight=1.0):
        # For undirected graph, only store the edge once (in both directions with same positive weight)
        # Only increment if this is a new edge
        if self.adjacency[u, v] == 0 and self.adjacency[v, u] == 0:
            self.edge_count += 1
        self.adjacency[u, v] = weight
        self.adjacency[v, u] = weight  # Same positive weight in both directions
        
    def get_neighbors(self, node):
        return np.where(self.adjacency[node] > 0)[0]
    
    def add_repeater_cycle(self, repeater_node, cycle_path):
        """
        Store a k-cycle path for a repeater node. Supports multiple cycles per repeater.
        
        Args:
            repeater_node: The repeater node
            cycle_path: List of nodes forming the k-cycle (including start/end repeater)
        """
        if repeater_node not in self.repeater_cycles:
            self.repeater_cycles[repeater_node] = []
        self.repeater_cycles[repeater_node].append(cycle_path)
    
    def get_repeater_cycle(self, repeater_node):
        """
        Get a k-cycle path for a repeater node. Randomly selects from multiple cycles if available.
        
        Args:
            repeater_node: The repeater node
            
        Returns:
            List of nodes in the k-cycle or None if not found
        """
        cycles = self.repeater_cycles.get(repeater_node, None)
        if cycles is None or len(cycles) == 0:
            return None
        
        # Randomly select one of the available cycles
        import random
        return random.choice(cycles)
    
    def get_all_repeater_cycles(self, repeater_node):
        """
        Get all k-cycle paths for a repeater node.
        
        Args:
            repeater_node: The repeater node
            
        Returns:
            List of k-cycles or None if not found
        """
        return self.repeater_cycles.get(repeater_node, None)
    
    def is_repeater_node(self, node):
        """Check if a node is a repeater node."""
        return node in self.repeater_cycles and len(self.repeater_cycles.get(node, [])) > 0
        
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
        """Check if graph is connected using spectral properties with Schur decomposition"""
        if self.n == 0:
            return True
        
        try:
            L = self.get_laplacian()
            
            # Convert to sparse matrix for better performance
            L_sparse = csr_matrix(L)
            
            # Use Schur decomposition first to improve convergence
            from scipy.linalg import schur
            try:
                # For small matrices, use dense decomposition
                if self.n <= 1000:
                    T, Z = schur(L.astype(np.float64))
                    eigenvalues = np.diag(T)
                    # Sort eigenvalues
                    eigenvalues = np.sort(eigenvalues)
                    # Second smallest eigenvalue should be > 0 for connected graph
                    return np.abs(eigenvalues[1]) > 1e-10
                else:
                    # For larger matrices, use sparse eigenvalue solver with better parameters
                    eigenvalues = eigs(L_sparse, k=2, which='SM', maxiter=1000, tol=1e-6)[0]
                    eigenvalues = np.sort(eigenvalues.real)
                    return np.abs(eigenvalues[1]) > 1e-10
                    
            except:
                # Fallback to BFS if spectral methods fail
                return self._is_connected_bfs()
                
        except:
            # Final fallback to BFS
            return self._is_connected_bfs()
    
    def _is_connected_bfs(self):
        """Fallback connectivity check using BFS traversal"""
        if self.n == 0:
            return True
        
        # Use BFS to check if all nodes are reachable from node 0
        visited = set()
        queue = [0]
        visited.add(0)
        
        while queue:
            current = queue.pop(0)
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # For undirected connectivity, we only need to check one direction
        # since our graph has bidirectional edges (positive and negative weights)
        return len(visited) == self.n
        
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
        
        # Save node attributes and repeater cycles
        attrs_file = f"{filepath}_attrs.json"
        graph_data = {
            'node_attributes': self.node_attributes,
            'repeater_cycles': self.repeater_cycles
        }
        with open(attrs_file, 'w') as f:
            json.dump(graph_data, f, default=lambda x: int(x) if hasattr(x, 'item') else str(x))
            
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
        
        # Recalculate edge count from adjacency matrix (count each undirected edge only once)
        # For undirected graph, only count upper triangle to avoid double counting
        upper_triangle = np.triu(adjacency > 0, k=1)
        graph.edge_count = np.sum(upper_triangle)
        
        # Load node attributes and repeater cycles
        attrs_file = f"{filepath}_attrs.json"
        if os.path.exists(attrs_file):
            with open(attrs_file, 'r') as f:
                graph_data = json.load(f)
                
                # Handle both old format (just node_attributes) and new format
                if isinstance(graph_data, dict) and 'node_attributes' in graph_data:
                    graph.node_attributes = graph_data['node_attributes']
                    # Load repeater cycles, converting string keys back to integers
                    if 'repeater_cycles' in graph_data:
                        graph.repeater_cycles = {
                            int(k): v for k, v in graph_data['repeater_cycles'].items()
                        }
                else:
                    # Old format - just node attributes
                    graph.node_attributes = graph_data
                
        return graph 

    def has_edge(self, u, v):
        """Return True if there is a positive edge from u to v."""
        return self.adjacency[u, v] > 0 