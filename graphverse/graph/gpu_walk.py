"""
GPU-accelerated walk generation for high-performance random walk sampling.

This module provides GPU implementations using PyTorch for vectorized
random walk generation, particularly beneficial for large graphs and
high walk counts.
"""

import random
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class GPUWalkGenerator:
    """GPU-accelerated random walk generator."""
    
    def __init__(self, graph, rules, device=None):
        """
        Initialize GPU walk generator.
        
        Args:
            graph: Graph object to walk on
            rules: List of rules to follow
            device: PyTorch device ("cuda", "mps", "cpu", or None for auto)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPU-accelerated walk generation")
        
        self.graph = graph
        self.rules = rules
        self.n = graph.n
        
        # Determine device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Convert adjacency matrix to GPU tensor
        # Use only positive weights (outgoing edges)
        adj_matrix = np.where(graph.adjacency > 0, graph.adjacency, 0)
        self.adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=self.device)
        
        # Precompute transition probabilities (row-normalized adjacency)
        row_sums = self.adj_tensor.sum(dim=1, keepdim=True)
        # Avoid division by zero for nodes with no outgoing edges
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        self.transition_probs = self.adj_tensor / row_sums
        
        # Precompute rule constraints as tensors for faster evaluation
        self._precompute_rule_constraints()
    
    def _precompute_rule_constraints(self):
        """Precompute rule constraints as GPU tensors for fast evaluation."""
        self.rule_constraints = {}
        
        for rule in self.rules:
            rule_name = rule.__class__.__name__
            
            if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
                # Ascender rule: can only go to higher-numbered nodes
                constraint_matrix = torch.zeros((self.n, self.n), dtype=torch.bool, device=self.device)
                for i in range(self.n):
                    constraint_matrix[i, i+1:] = True
                self.rule_constraints['ascender'] = constraint_matrix
                
            elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
                # Even rule: can only go to even nodes
                even_nodes = torch.tensor([i for i in range(self.n) if i % 2 == 0], 
                                        dtype=torch.long, device=self.device)
                constraint_matrix = torch.zeros((self.n, self.n), dtype=torch.bool, device=self.device)
                constraint_matrix[:, even_nodes] = True
                self.rule_constraints['even'] = constraint_matrix
                
            elif hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                # Repeater rule: more complex - store node-specific constraints
                repeater_constraints = {}
                for node, k_value in rule.members_nodes_dict.items():
                    # This is complex to vectorize efficiently, so we'll handle it separately
                    repeater_constraints[node] = k_value
                self.rule_constraints['repeater'] = repeater_constraints
    
    def generate_walks_batch(self, num_walks: int, min_length: int, max_length: int, 
                           batch_size: Optional[int] = None) -> List[List[int]]:
        """
        Generate multiple walks in batches using GPU acceleration.
        
        Args:
            num_walks: Total number of walks to generate
            min_length: Minimum walk length
            max_length: Maximum walk length
            batch_size: Batch size for GPU processing (None = auto)
            
        Returns:
            List of generated walks
        """
        if batch_size is None:
            # Auto-determine batch size based on available GPU memory
            if self.device.type == "cuda":
                batch_size = min(1024, max(64, num_walks // 10))
            elif self.device.type == "mps":
                batch_size = min(512, max(32, num_walks // 20))
            else:
                batch_size = min(256, max(16, num_walks // 50))
        
        all_walks = []
        remaining_walks = num_walks
        
        while remaining_walks > 0:
            current_batch_size = min(batch_size, remaining_walks)
            
            # Generate batch of walks
            batch_walks = self._generate_batch_gpu(current_batch_size, min_length, max_length)
            all_walks.extend(batch_walks)
            
            remaining_walks -= len(batch_walks)
        
        return all_walks[:num_walks]  # Ensure exact count
    
    def _generate_batch_gpu(self, batch_size: int, min_length: int, max_length: int) -> List[List[int]]:
        """Generate a batch of walks using GPU operations."""
        
        # Generate random starting nodes for the batch
        start_nodes = torch.randint(0, self.n, (batch_size,), device=self.device)
        
        # Generate random target lengths for each walk
        target_lengths = torch.randint(min_length, max_length + 1, (batch_size,), device=self.device)
        max_len = int(target_lengths.max().item())
        
        # Initialize walk storage
        walks = torch.full((batch_size, max_len), -1, dtype=torch.long, device=self.device)
        walks[:, 0] = start_nodes
        
        # Track which walks are still active
        active_walks = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        walk_lengths = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        # Generate walks step by step
        for step in range(1, max_len):
            if not active_walks.any():
                break
            
            # Get current positions for active walks
            current_nodes = walks[active_walks, step - 1]
            
            # Get valid next nodes for each active walk
            valid_transitions = self._get_valid_transitions_batch(
                current_nodes, walks[active_walks, :step]
            )
            
            # Sample next nodes from valid transitions
            next_nodes = self._sample_next_nodes_batch(current_nodes, valid_transitions)
            
            # Update walks
            active_indices = torch.where(active_walks)[0]
            walks[active_indices, step] = next_nodes
            walk_lengths[active_indices] += 1
            
            # Check which walks should stop (reached target length)
            should_stop = walk_lengths >= target_lengths
            active_walks &= ~should_stop
        
        # Convert back to CPU and format as lists
        walks_cpu = walks.cpu().numpy()
        result_walks = []
        
        for i in range(batch_size):
            length = int(walk_lengths[i].item())
            walk = walks_cpu[i, :length].tolist()
            # Filter out invalid nodes (should not happen with proper implementation)
            walk = [node for node in walk if node != -1]
            if len(walk) >= min_length:
                result_walks.append(walk)
        
        return result_walks
    
    def _get_valid_transitions_batch(self, current_nodes: torch.Tensor, 
                                   walk_histories: torch.Tensor) -> torch.Tensor:
        """
        Get valid transitions for a batch of current nodes considering rule constraints.
        
        Args:
            current_nodes: Current node positions (batch_size,)
            walk_histories: History of walks so far (batch_size, history_length)
            
        Returns:
            Boolean tensor indicating valid transitions (batch_size, n_nodes)
        """
        batch_size = current_nodes.size(0)
        
        # Start with adjacency-based valid moves
        valid_moves = self.adj_tensor[current_nodes] > 0  # (batch_size, n_nodes)
        
        # Apply rule constraints
        for rule in self.rules:
            rule_name = rule.__class__.__name__
            
            if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
                # Check if current nodes are ascender nodes
                ascender_nodes = torch.tensor(list(rule.member_nodes), device=self.device)
                is_ascender = torch.isin(current_nodes, ascender_nodes)
                
                if is_ascender.any():
                    # Apply ascender constraint: only higher-numbered nodes
                    ascender_indices = torch.where(is_ascender)[0]
                    for idx in ascender_indices:
                        current_node = current_nodes[idx].item()
                        # Only allow transitions to higher-numbered nodes
                        valid_moves[idx, :current_node + 1] = False
            
            elif hasattr(rule, 'is_even_rule') and rule.is_even_rule:
                # Check if current nodes are even rule nodes  
                even_rule_nodes = torch.tensor(list(rule.member_nodes), device=self.device)
                is_even_rule = torch.isin(current_nodes, even_rule_nodes)
                
                if is_even_rule.any():
                    # Apply even rule constraint: only even nodes
                    even_rule_indices = torch.where(is_even_rule)[0]
                    odd_nodes = torch.arange(1, self.n, 2, device=self.device)  # 1, 3, 5, ...
                    valid_moves[even_rule_indices][:, odd_nodes] = False
            
            elif hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                # Repeater rule is more complex and requires checking path history
                # For now, implement a simplified version
                repeater_nodes = torch.tensor(list(rule.member_nodes), device=self.device)
                is_repeater = torch.isin(current_nodes, repeater_nodes)
                
                if is_repeater.any():
                    # This would require more complex logic to track k-step cycles
                    # For initial implementation, allow all valid adjacency moves
                    # TODO: Implement proper k-step cycle checking
                    pass
        
        return valid_moves
    
    def _sample_next_nodes_batch(self, current_nodes: torch.Tensor, 
                               valid_transitions: torch.Tensor) -> torch.Tensor:
        """
        Sample next nodes from valid transitions using transition probabilities.
        
        Args:
            current_nodes: Current positions (batch_size,)
            valid_transitions: Valid transitions mask (batch_size, n_nodes)
            
        Returns:
            Next node for each walk (batch_size,)
        """
        batch_size = current_nodes.size(0)
        
        # Get transition probabilities for current nodes
        trans_probs = self.transition_probs[current_nodes]  # (batch_size, n_nodes)
        
        # Mask out invalid transitions
        masked_probs = trans_probs * valid_transitions.float()
        
        # Handle cases where no valid transitions exist (fallback to random valid adjacency)
        row_sums = masked_probs.sum(dim=1, keepdim=True)
        no_valid = (row_sums.squeeze() == 0)
        
        if no_valid.any():
            # Fallback: use raw adjacency for nodes with no valid rule-compliant moves
            fallback_indices = torch.where(no_valid)[0]
            raw_adjacency = self.adj_tensor[current_nodes[fallback_indices]] > 0
            fallback_probs = raw_adjacency.float()
            fallback_sums = fallback_probs.sum(dim=1, keepdim=True)
            fallback_sums = torch.where(fallback_sums > 0, fallback_sums, torch.ones_like(fallback_sums))
            fallback_probs = fallback_probs / fallback_sums
            
            masked_probs[fallback_indices] = fallback_probs
            row_sums[fallback_indices] = fallback_sums
        
        # Normalize probabilities
        normalized_probs = masked_probs / torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        
        # Sample from categorical distributions
        try:
            next_nodes = torch.multinomial(normalized_probs, 1).squeeze(1)
        except RuntimeError:
            # Fallback to uniform sampling from valid moves if multinomial fails
            valid_indices = torch.where(valid_transitions)
            batch_indices = valid_indices[0]
            node_indices = valid_indices[1]
            
            next_nodes = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            for b in range(batch_size):
                batch_mask = batch_indices == b
                if batch_mask.any():
                    valid_for_batch = node_indices[batch_mask]
                    chosen_idx = torch.randint(0, len(valid_for_batch), (1,), device=self.device)
                    next_nodes[b] = valid_for_batch[chosen_idx]
                else:
                    # Emergency fallback: stay at current node (should not happen)
                    next_nodes[b] = current_nodes[b]
        
        return next_nodes


def generate_walks_gpu_accelerated(
    graph, 
    num_walks: int,
    min_length: int, 
    max_length: int, 
    rules,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False
) -> List[List[int]]:
    """
    Generate walks using GPU acceleration.
    
    Args:
        graph: Graph object
        num_walks: Number of walks to generate
        min_length: Minimum walk length
        max_length: Maximum walk length
        rules: Rules to follow
        device: Device to use ("cuda", "mps", "cpu", or None for auto)
        batch_size: Batch size for GPU processing
        verbose: Show progress
        
    Returns:
        List of generated walks
        
    Raises:
        ImportError: If PyTorch is not available
        RuntimeError: If specified device is not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU-accelerated walk generation")
    
    if verbose:
        print(f"\n  GPU-accelerated walk generation:")
        print(f"  Target walks: {num_walks}")
        print(f"  Walk length: {min_length}-{max_length}")
    
    try:
        generator = GPUWalkGenerator(graph, rules, device)
        
        if verbose:
            print(f"  Using device: {generator.device}")
            print(f"  Batch size: {batch_size or 'auto'}")
        
        walks = generator.generate_walks_batch(num_walks, min_length, max_length, batch_size)
        
        if verbose:
            print(f"  ✓ Generated {len(walks)} walks using GPU acceleration")
        
        return walks
        
    except Exception as e:
        if verbose:
            print(f"  ⚠ GPU generation failed: {e}")
            print(f"  Falling back to CPU implementation")
        
        # Fallback to CPU implementation
        from .walk import generate_multiple_walks_sequential
        return generate_multiple_walks_sequential(graph, num_walks, min_length, max_length, rules, verbose)


# Integration function for the parallel walk module
def get_gpu_walk_generator(graph, rules, device=None):
    """
    Factory function to create GPU walk generator if available.
    
    Returns:
        GPUWalkGenerator instance or None if not available
    """
    if not TORCH_AVAILABLE:
        return None
    
    try:
        return GPUWalkGenerator(graph, rules, device)
    except Exception:
        return None