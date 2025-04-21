import torch
import numpy as np

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
        training_data: List of (walk, label) pairs
        vocab: Vocabulary mapping node indices to tokens
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
        if verbose:
            print(f"Generating a walk starting from node {node}")
        walk = generate_valid_walk(
            graph, node, min_length, max_length, rules, verbose=verbose
        )
        if walk:
            per_node_walks.append(walk)
    
    # Combine all walks
    all_walks = walks + per_node_walks
    
    # Create vocabulary
    vocab = {i: str(i) for i in range(graph.n)}
    
    # Create training data
    training_data = []
    for walk in all_walks:
        # For each position in walk, predict next node
        for i in range(len(walk) - 1):
            context = walk[:i+1]
            target = walk[i+1]
            training_data.append((context, target))
    
    return training_data, vocab
