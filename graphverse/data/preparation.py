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
    """
    if verbose:
        print(f"Generating a walk starting from each node in the graph...")
    per_node_walks = []
    for node in graph.nodes:
        if verbose:
            print(f"Generating a walk starting from node {node}")
        valid_walk = generate_valid_walk(graph, node, min_length, max_length, rules)
        if valid_walk:
            per_node_walks.append(valid_walk)
        if verbose:
            print()  # Print a new line after each iteration

    # Generate walks
    if verbose:
        print(f"Generating {num_walks} walks...")
    walks = generate_multiple_walks(
        graph, num_walks, min_length, max_length, rules, verbose=verbose
    )

    walks = walks + per_node_walks

    # Create vocabulary
    vocab = WalkVocabulary(walks)

    tensor_data = []
    for walk in walks:
        tensor_walk = (
            [vocab.token2idx["<START>"]]
            + [vocab.token2idx[str(node)] for node in walk]
            + [vocab.token2idx["<END>"]]
        )
        tensor_data.append(torch.tensor(tensor_walk))

    if verbose:
        print(f"Number of walks: {len(walks)}")
        print(f"Vocabulary size: {len(vocab)}")
        print(
            f"Tensor data shape: {torch.nn.utils.rnn.pad_sequence(tensor_data, batch_first=True, padding_value=vocab.token2idx['<PAD>']).shape}"
        )

    return torch.nn.utils.rnn.pad_sequence(
        tensor_data, batch_first=True, padding_value=vocab.token2idx["<PAD>"]
    ), vocab
