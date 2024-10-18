import networkx as nx
import matplotlib.pyplot as plt
import torch
import pandas as pd
import math
import random

from graphverse.graph.graph_generation import generate_random_graph, calculate_edge_density
from graphverse.graph.rules import AscenderRule, DescenderRule, EvenRule, OddRule, RepeaterRule
from graphverse.graph.rules import define_all_rules
from graphverse.data.preparation import prepare_training_data
from graphverse.llm.training import train_model
from graphverse.llm.evaluation import evaluate_model

def main(n, num_walks, min_walk_length, max_walk_length, num_repeaters, repeater_min_steps, repeater_max_steps, epochs, batch_size, learning_rate, verbose=False):
    # Define rule sets
    if verbose:
        print('Selecting vertices with rules')
    ascenders, descenders, evens, odds, repeaters = define_all_rules(n, num_repeaters, repeater_min_steps, repeater_max_steps)
    rules = (ascenders, descenders, evens, odds, repeaters)

    # Generate graph
    if verbose:
        print('Generating graph')
    G = generate_random_graph(n, rules, num_walks, min_walk_length, max_walk_length, verbose=verbose)

    if verbose:
        print(f'Graph created')
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Is strongly connected: {nx.is_strongly_connected(G)}")
        print(f"Is weakly connected: {nx.is_weakly_connected(G)}")
        print(f'Now preparing training data')

    # Prepare training data
    walks = generate_multiple_walks(G, num_walks, min_walk_length, max_walk_length, rules, verbose=verbose)
    training_data, vocab = prepare_training_data(G, walks, rules, verbose=verbose)
    if verbose:
        print(f'Training data prepared')

    # Train model
    if verbose:
        print(f'Training model')
    model = train_model(training_data, vocab, epochs, batch_size, learning_rate, verbose=verbose)
    if verbose:
        print(f'Model trained')

    return model, G, vocab

if __name__ == "__main__":
    n = 1000
    num_walks = 10000
    min_walk_length = 5
    max_walk_length = 20
    num_repeaters = 3
    repeater_min_steps = 3
    repeater_max_steps = 10
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    verbose = True

    model, G, vocab = main(n, num_walks, min_walk_length, max_walk_length, num_repeaters, repeater_min_steps, repeater_max_steps, epochs, batch_size, learning_rate, verbose=verbose)
