import math
import random

import pandas as pd
import torch
import numpy as np

from graphverse.data.preparation import prepare_training_data
from graphverse.graph.graph_generation import (
    calculate_edge_density,
    generate_random_graph,
)
from graphverse.graph.rules import (
    AscenderRule,
    EvenRule,
    RepeaterRule,
    define_all_rules,
)
from graphverse.graph.walk import generate_multiple_walks
from graphverse.llm.evaluation import evaluate_model
from graphverse.llm.training import train_model
from graphverse.graph.base import Graph


def main(
    n,
    num_walks,
    min_walk_length,
    max_walk_length,
    num_ascenders,
    num_evens,
    num_repeaters,
    repeater_min_steps,
    repeater_max_steps,
    epochs,
    batch_size,
    learning_rate,
    verbose=False,
):
    # Define rule sets
    if verbose:
        print("Selecting vertices with rules")
    ascenders, evens, repeaters = define_all_rules(
        n, num_ascenders, num_evens, num_repeaters, repeater_min_steps, repeater_max_steps
    )

    # Create rule objects
    ascender_rule = AscenderRule(ascenders)
    even_rule = EvenRule(evens)
    repeater_rule = RepeaterRule(repeaters)

    # Set of rules
    rules = {ascender_rule, even_rule, repeater_rule}

    # Generate graph
    if verbose:
        print("Generating graph")
    G = generate_random_graph(
        n=n,
        rules=rules,
        num_walks=num_walks,
        min_walk_length=min_walk_length,
        max_walk_length=max_walk_length,
        verbose=verbose,
        save_walks=True,
        output_dir="walks"
    )

    if verbose:
        print(f"Graph created")
        print(f"Number of nodes: {G.n}")
        print(f"Number of edges: {np.sum(G.adjacency > 0)}")
        print(f"Is connected: {G.is_connected()}")
        print(f"Now preparing training data")

    # Prepare training data
    walks = generate_multiple_walks(
        G, num_walks, min_walk_length, max_walk_length, rules, verbose=verbose
    )
    training_data, vocab = prepare_training_data(
        G, num_walks, min_walk_length, max_walk_length, rules, verbose=verbose
    )
    if verbose:
        print(f"Training data prepared")

    # Train model
    if verbose:
        print(f"Training model")
    model = train_model(
        training_data, vocab, epochs, batch_size, learning_rate, verbose=verbose
    )
    if verbose:
        print(f"Model trained")

    return model, G, vocab


if __name__ == "__main__":
    n = 1000
    num_walks = 10000
    min_walk_length = 5
    max_walk_length = 20
    num_ascenders = 10
    num_evens = 10
    num_repeaters = 10
    repeater_min_steps = 3
    repeater_max_steps = 10
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    verbose = True

    model, G, vocab = main(
        n,
        num_walks,
        min_walk_length,
        max_walk_length,
        num_ascenders,
        num_evens,
        num_repeaters,
        repeater_min_steps,
        repeater_max_steps,
        epochs,
        batch_size,
        learning_rate,
        verbose=verbose,
    )
