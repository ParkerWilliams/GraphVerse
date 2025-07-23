import torch
import numpy as np
import random

from ..graph.walk import generate_multiple_walks, generate_valid_walk


class WalkVocabulary:
    """
    Vocabulary for walks.
    """

    def __init__(self, walks):
        max_node = np.max(np.concatenate(walks))
        nodes = range(0, max_node + 1)
        self.token2idx = {
            **{str(x): x for x in nodes},
            **{"<PAD>": max_node + 1, "<START>": max_node + 2, "<END>": max_node + 3}
        }
        self.idx2token = {
            **{x: str(x) for x in nodes},
            **{max_node + 1: "<PAD>", max_node + 2: "<START>", max_node + 3: "<END>"}
        }
        # self. = {}
        # self.build_vocab(walks)

    # def build_vocab(self, walks):
    #     for walk in walks:
    #         for token in walk:
    #             if str(token) not in self.token2idx:
    #                 idx = len(self.token2idx) - 3
    #                 self.token2idx[str(token)] = idx
    #                 self.idx2token[idx] = str(token)

    def __len__(self):
        return len(self.token2idx)


def prepare_training_data(
    graph, num_walks, min_length, max_length, rules, verbose=False
):
    """
    Prepare training data for the model.
    
    Args:
        graph: The graph to generate walks on
        num_walks: Number of random walks to generate
        min_length: Minimum walk length
        max_length: Maximum walk length
        rules: Set of rules to follow
        verbose: Whether to print progress
        
    Returns:
        training_data: Tensor of shape (N, max_seq_len) containing input sequences
        vocab: WalkVocabulary object mapping node indices to tokens
    """
    # Generate walks
    if verbose:
        print(f"Generating {num_walks} random walks...")
    walks = generate_multiple_walks(
        graph, num_walks, min_length, max_length, rules, verbose=verbose
    )
    
    if verbose:
        print(f"Generating a walk starting from each node in the graph...")
    per_node_walks = []
    for node in range(graph.n):
        walk = generate_valid_walk(
            graph, node, min_length, max_length, rules, verbose=verbose
        )
        if walk:
            per_node_walks.append(walk)
    
    # Combine all walks
    all_walks = walks + per_node_walks
    
    # Create vocabulary using WalkVocabulary class
    vocab = WalkVocabulary(all_walks)
    
    # Create training data
    max_seq_len = max(len(walk) for walk in all_walks)
    training_sequences = []
    
    for walk in all_walks:
        # Convert walk to indices, including START and END tokens
        walk_indices = [vocab.token2idx["<START>"]] + [vocab.token2idx[str(node)] for node in walk] + [vocab.token2idx["<END>"]]
        # Pad sequence to max_seq_len + 2 (+2 for START and END tokens)
        padded = walk_indices + [vocab.token2idx["<PAD>"]] * (max_seq_len + 2 - len(walk_indices))
        training_sequences.append(padded)
    
    # Convert to tensor
    training_data = torch.tensor(training_sequences, dtype=torch.long)
    
    return training_data, vocab

def prepare_density_controlled_training_data(
    graph, 
    num_walks, 
    min_length, 
    max_length, 
    rules, 
    repeater_densities=None,
    verbose=False
):
    """
    Prepare training data with controlled repeater node density.
    
    Args:
        graph: The graph to generate walks on
        num_walks: Base number of random walks to generate
        min_length: Minimum walk length
        max_length: Maximum walk length
        rules: Set of rules to follow
        repeater_densities: Dict mapping repeater_node -> additional_walk_count
                           Controls how many extra walks include each repeater node
        verbose: Whether to print progress
        
    Returns:
        training_data: Tensor of shape (N, max_seq_len) containing input sequences
        vocab: WalkVocabulary object mapping node indices to tokens
        density_stats: Dictionary with statistics about repeater exposure
    """
    from ..graph.walk import generate_multiple_walks, generate_valid_walk
    
    # Generate base walks
    if verbose:
        print(f"Generating {num_walks} base random walks...")
    base_walks = generate_multiple_walks(
        graph, num_walks, min_length, max_length, rules, verbose=verbose
    )
    
    # Generate per-node walks (one from each node)
    if verbose:
        print(f"Generating walks starting from each node...")
    per_node_walks = []
    for node in range(graph.n):
        walk = generate_valid_walk(
            graph, node, min_length, max_length, rules, verbose=verbose
        )
        if walk:
            per_node_walks.append(walk)
    
    # Generate additional walks for specific repeater nodes if requested
    density_walks = []
    repeater_exposure_counts = {}
    
    if repeater_densities:
        # Find repeater nodes from rules
        repeater_nodes = set()
        for rule in rules:
            if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
                repeater_nodes.update(rule.member_nodes)
        
        if verbose:
            print(f"Generating additional walks for repeater density control...")
            print(f"Repeater nodes: {repeater_nodes}")
            print(f"Density targets: {repeater_densities}")
        
        for repeater_node, additional_count in repeater_densities.items():
            if repeater_node in repeater_nodes:
                node_walks = []
                attempts = 0
                max_attempts = additional_count * 10  # Allow more attempts
                
                while len(node_walks) < additional_count and attempts < max_attempts:
                    # Try starting from the repeater node
                    walk = generate_valid_walk(
                        graph, repeater_node, min_length, max_length, rules, verbose=False
                    )
                    if walk and repeater_node in walk:
                        node_walks.append(walk)
                    
                    # Also try random starts that might pass through the repeater
                    if len(node_walks) < additional_count:
                        start_node = random.choice(list(range(graph.n)))
                        walk = generate_valid_walk(
                            graph, start_node, min_length, max_length, rules, verbose=False
                        )
                        if walk and repeater_node in walk:
                            node_walks.append(walk)
                    
                    attempts += 1
                
                density_walks.extend(node_walks)
                if verbose:
                    print(f"Generated {len(node_walks)} additional walks for repeater {repeater_node}")
    
    # Combine all walks
    all_walks = base_walks + per_node_walks + density_walks
    
    # Calculate exposure statistics
    if repeater_densities:
        for repeater_node in repeater_densities.keys():
            count = sum(1 for walk in all_walks if repeater_node in walk)
            repeater_exposure_counts[repeater_node] = count
    
    # Create vocabulary
    vocab = WalkVocabulary(all_walks)
    
    # Create training data
    max_seq_len = max(len(walk) for walk in all_walks)
    training_sequences = []
    
    for walk in all_walks:
        # Convert walk to indices, including START and END tokens
        walk_indices = [vocab.token2idx["<START>"]] + [vocab.token2idx[str(node)] for node in walk] + [vocab.token2idx["<END>"]]
        # Pad sequence to max_seq_len + 2 (+2 for START and END tokens)
        padded = walk_indices + [vocab.token2idx["<PAD>"]] * (max_seq_len + 2 - len(walk_indices))
        training_sequences.append(padded)
    
    # Convert to tensor
    training_data = torch.tensor(training_sequences, dtype=torch.long)
    
    # Prepare density statistics
    density_stats = {
        "total_walks": len(all_walks),
        "base_walks": len(base_walks),
        "per_node_walks": len(per_node_walks),
        "density_walks": len(density_walks),
        "repeater_exposure_counts": repeater_exposure_counts,
        "repeater_densities_requested": repeater_densities or {}
    }
    
    if verbose:
        print(f"Training data prepared:")
        print(f"  Total walks: {len(all_walks)}")
        print(f"  Repeater exposures: {repeater_exposure_counts}")
    
    return training_data, vocab, density_stats
