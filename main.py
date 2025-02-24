import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch

from graphverse.data.preparation import prepare_training_data
from graphverse.graph.graph_generation import (
    calculate_edge_density,
    generate_random_graph,
)
from graphverse.graph.rules import (
    AscenderRule,
    DescenderRule,
    EvenRule,
    OddRule,
    RepeaterRule,
    define_all_rules,
)
from graphverse.graph.walk import generate_multiple_walks
from graphverse.llm.evaluation import evaluate_model
from graphverse.llm.training import train_model


def main(
    n,
    num_walks,
    min_walk_length,
    max_walk_length,
    num_ascenders,
    num_descenders,
    num_evens,
    num_odds,
    num_repeaters,
    repeater_min_steps,
    repeater_max_steps,
    epochs,
    batch_size,
    learning_rate,
    verbose=False,
):
    # Define rule sets
    #def define_all_rules(n, num_ascenders, num_descenders, num_evens, num_odds, num_repeaters, repeater_min_steps, repeater_max_steps):
    if verbose:
        print("Selecting vertices with rules")
    ascenders, descenders, evens, odds, repeaters = define_all_rules(
        n, num_ascenders, num_descenders, num_evens, num_odds, num_repeaters, repeater_min_steps, repeater_max_steps
    )

    # Create rule objects
    ascender_rule = AscenderRule(ascenders)
    descender_rule = DescenderRule(descenders)
    even_rule = EvenRule(evens)
    odd_rule = OddRule(odds)
    repeater_rule = RepeaterRule(repeaters)

    # Set of rules
    rules = {ascender_rule, descender_rule, even_rule, odd_rule, repeater_rule}

    # Generate graph
    if verbose:
        print("Generating graph")
    G = generate_random_graph(
        n, rules, verbose=verbose, save_walks=True, output_dir="walks"
    )

    if verbose:
        print(f"Graph created")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Is strongly connected: {nx.is_strongly_connected(G)}")
        print(f"Is weakly connected: {nx.is_weakly_connected(G)}")
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
    num_descenders = 10
    num_evens = 10
    num_odds = 10
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
        num_descenders,
        num_evens,
        num_odds,
        num_repeaters,
        repeater_min_steps,
        repeater_max_steps,
        epochs,
        batch_size,
        learning_rate,
        verbose=verbose,
    )
