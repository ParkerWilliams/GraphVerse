import matplotlib.pyplot as plt
import torch
import pandas as pd
import math
import random
import os
import pickle

from graphverse.graph.graph_generation import generate_random_graph, calculate_edge_density
from graphverse.graph.rules import AscenderRule, EvenRule, RepeaterRule
from graphverse.graph.rules import define_all_rules
from graphverse.data.preparation import prepare_training_data
from graphverse.llm.training import train_model
from graphverse.llm.evaluation import evaluate_model
from graphverse.graph.walk import generate_multiple_walks
from graphverse.utils.experiment_manager import (
    create_experiment_folder, save_config,
    save_error_summary, save_kl_divergence_series
)
from graphverse.llm.evaluation_vis import plot_error_summary, plot_kl_divergence_timeseries, plot_aggregate_kl

def main(n, num_walks, min_walk_length, max_walk_length, num_repeaters, repeater_min_steps, repeater_max_steps, epochs, batch_size, learning_rate, verbose=False, context_window_size=None, repeater_distance=None, seed=None):
    # --- New: Create experiment folder and save config ---
    experiment_folder = create_experiment_folder()
    config = {
        "n": n,
        "num_walks": num_walks,
        "min_walk_length": min_walk_length,
        "max_walk_length": max_walk_length,
        "num_repeaters": num_repeaters,
        "repeater_min_steps": repeater_min_steps,
        "repeater_max_steps": repeater_max_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "context_window_size": context_window_size,
        "repeater_distance": repeater_distance,
        "seed": seed,
    }
    save_config(config, experiment_folder)

    # Define rule sets
    if verbose:
        print('Selecting vertices with rules')
    ascenders, descenders, evens, odds, repeaters = define_all_rules(n, num_repeaters, repeater_min_steps, repeater_max_steps)

    # Create rule instances
    ascender_rule = AscenderRule(ascenders)
    descender_rule = DescenderRule(descenders)
    even_rule = EvenRule(evens)
    odd_rule = OddRule(odds)
    repeater_rule = RepeaterRule(repeaters)
    rule_instances = [ascender_rule, descender_rule, even_rule, odd_rule, repeater_rule]

    # Generate graph
    if verbose:
        print('Generating graph')
    G = generate_random_graph(n, (ascenders, descenders, evens, odds, repeaters), num_walks, min_walk_length, max_walk_length, verbose=verbose)

    if verbose:
        print(f'Graph created')
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Is strongly connected: {nx.is_strongly_connected(G)}")
        print(f"Is weakly connected: {nx.is_weakly_connected(G)}")
        print(f'Now preparing training data')

    # Prepare training data
    walks = generate_multiple_walks(G, num_walks, min_walk_length, max_walk_length, rule_instances, verbose=verbose)
    training_data, vocab = prepare_training_data(G, walks, verbose=verbose)
    if verbose:
        print(f'Training data prepared')

    # Train model
    if verbose:
        print(f'Training model')
    model = train_model(training_data, vocab, epochs, batch_size, learning_rate, verbose=verbose)
    if verbose:
        print(f'Model trained')

    # Save model
    torch.save(model.state_dict(), os.path.join(experiment_folder, "model.pth"))
    # Save dataset
    with open(os.path.join(experiment_folder, "data", "training_data.pt"), "wb") as f:
        torch.save(training_data, f)
    # Save graph
    with open(os.path.join(experiment_folder, "data", "graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    # Save vocabulary
    with open(os.path.join(experiment_folder, "data", "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    # --- New: Evaluate model and save metrics ---
    evaluation_results, error_summary, kl_divergence_series = evaluate_model(
        model, G, vocab, num_walks, min_walk_length, max_walk_length, rule_instances, verbose=verbose
    )
    save_error_summary(error_summary, experiment_folder)
    save_kl_divergence_series(kl_divergence_series, experiment_folder)
    # --- End new code ---

    return model, G, vocab, rule_instances

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

    model, G, vocab, rule_instances = main(n, num_walks, min_walk_length, max_walk_length, num_repeaters, repeater_min_steps, repeater_max_steps, epochs, batch_size, learning_rate, verbose=verbose)

    # Paths to your experiment outputs
    error_summary_path = "experiments/run_YYYYMMDD_HHMMSS/evaluation/error_summary.json"
    kl_csv_path = "experiments/run_YYYYMMDD_HHMMSS/evaluation/kl_divergence_timeseries.csv"

    # Plot error summary
    plot_error_summary(error_summary_path, output_path="experiments/run_YYYYMMDD_HHMMSS/evaluation/error_rates.png")

    # Plot KL divergence for a specific walk
    plot_kl_divergence_timeseries(kl_csv_path, walk_idx=0, output_path="experiments/run_YYYYMMDD_HHMMSS/evaluation/kl_walk0.png")

    # Plot aggregate KL divergence
    plot_aggregate_kl(kl_csv_path, output_path="experiments/run_YYYYMMDD_HHMMSS/evaluation/kl_aggregate.png")
