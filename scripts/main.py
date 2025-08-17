import torch
import numpy as np
import random
import os
import pickle

from graphverse.graph.graph_generation import generate_random_graph, calculate_edge_density
from graphverse.graph.rules import AscenderRule, EvenRule, RepeaterRule
from graphverse.graph.rules import define_all_rules, define_all_rules_by_percentage
from graphverse.data.preparation import prepare_training_data
from graphverse.analysis.metadata import GraphMetadata, ExperimentMetadata
from graphverse.llm.training import train_model
from graphverse.llm.evaluation import evaluate_model
from graphverse.graph.walk import generate_multiple_walks
from graphverse.utils.experiment_manager import (
    create_experiment_folder, save_config,
    save_error_summary, save_kl_divergence_series,
    save_token_level_data, save_token_summary_stats,
    save_rule_violation_analysis, save_baseline_performance_summary,
    save_exemplar_walks
)
from graphverse.llm.evaluation_vis import (
    plot_error_summary, plot_kl_divergence_timeseries, plot_aggregate_kl,
    plot_token_kl_heatmap, plot_token_entropy_vs_kl, plot_prediction_confidence_analysis
)

def main(n, num_walks, context_window_size, num_ascenders, num_evens, num_repeaters, repeater_min_steps, repeater_max_steps, epochs, batch_size, learning_rate, verbose=False, repeater_distance=None, seed=None, use_percentages=False, edge_concentration=0.8):
    # Calculate walk lengths based on context window (walks = 2x context window)
    min_walk_length = 2 * context_window_size
    max_walk_length = 2 * context_window_size
    
    # --- New: Create experiment folder and save config ---
    experiment_folder = create_experiment_folder()
    config = {
        "n": n,
        "num_walks": num_walks,
        "context_window_size": context_window_size,
        "min_walk_length": min_walk_length,
        "max_walk_length": max_walk_length,
        "walk_length_constraint": "2x_context_window",
        "num_ascenders": num_ascenders,
        "num_evens": num_evens,
        "num_repeaters": num_repeaters,
        "repeater_min_steps": repeater_min_steps,
        "repeater_max_steps": repeater_max_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "repeater_distance": repeater_distance,
        "seed": seed,
        "use_percentages": use_percentages,
        "rule_definition_method": "percentage_based" if use_percentages else "fixed_counts",
        "edge_concentration": edge_concentration,
        "edge_weight_strategy": "dirichlet"
    }
    save_config(config, experiment_folder)

    # Define rule sets
    if verbose:
        print("\n" + "="*60)
        print("EXPERIMENT CONFIGURATION")
        print("="*60)
        print(f"  Context window size: {context_window_size}")
        print(f"  Walk length: {min_walk_length} (2x context window)")
        print(f"  Graph size: {n} nodes")
        print()
        if use_percentages:
            print("PERCENTAGE-BASED RULE CONFIGURATION")
            print(f"  Ascender percentage: {num_ascenders}%")
            print(f"  Even rule percentage: {num_evens}%")
            print(f"  Repeater percentage: {num_repeaters}%")
        else:
            print("FIXED-COUNT RULE CONFIGURATION")
            print(f"  Ascender nodes: {num_ascenders}")
            print(f"  Even rule nodes: {num_evens}")
            print(f"  Repeater nodes: {num_repeaters}")
        print(f"  Repeater step range: {repeater_min_steps}-{repeater_max_steps}")
        print(f"  Context window vs repeater analysis:")
        print(f"    Context window size: {context_window_size}")
        print(f"    Learnable repeaters (k ≤ {context_window_size}): {repeater_min_steps}-{min(repeater_max_steps, context_window_size)}")
        if repeater_max_steps > context_window_size:
            print(f"    Challenging repeaters (k > {context_window_size}): {context_window_size + 1}-{repeater_max_steps}")
        else:
            print(f"    No challenging repeaters (all k ≤ {context_window_size})")
        print("="*60 + "\n")
        print('Selecting vertices with rules...')
    
    if use_percentages:
        ascenders, evens, repeaters = define_all_rules_by_percentage(
            n, ascender_percent=num_ascenders, even_percent=num_evens, repeater_percent=num_repeaters, 
            repeater_min_steps=repeater_min_steps, repeater_max_steps=repeater_max_steps, verbose=verbose
        )
    else:
        ascenders, evens, repeaters = define_all_rules(
            n, num_ascenders=num_ascenders, num_evens=num_evens, num_repeaters=num_repeaters, 
            repeater_min_steps=repeater_min_steps, repeater_max_steps=repeater_max_steps
        )
        
        if verbose:
            print(f"  Ascender nodes selected: {sorted(list(ascenders)[:5])}..." if len(ascenders) > 5 else f"  Ascender nodes: {sorted(list(ascenders))}")
            print(f"  Even rule nodes selected: {sorted(list(evens)[:5])}..." if len(evens) > 5 else f"  Even rule nodes: {sorted(list(evens))}")
            print(f"  Repeater nodes selected: {list(repeaters.keys())[:5]}..." if len(repeaters) > 5 else f"  Repeater nodes: {list(repeaters.keys())}")
            if repeaters:
                print(f"  Detailed repeater assignments:")
                for node, k in sorted(repeaters.items()):
                    learnable = "✓ Learnable" if k <= context_window_size else "✗ Challenging"
                    print(f"    Node {node}: repeat every {k} steps ({learnable})")

    # Create rule instances
    ascender_rule = AscenderRule(ascenders)
    even_rule = EvenRule(evens)
    repeater_rule = RepeaterRule(repeaters)
    rule_instances = [ascender_rule, even_rule, repeater_rule]

    # Generate graph
    if verbose:
        print("\n" + "="*60)
        print('GRAPH GENERATION')
        print("="*60)
    G = generate_random_graph(
        n, rule_instances, num_walks, min_walk_length, max_walk_length, 
        verbose=verbose, edge_concentration=edge_concentration
    )

    if verbose:
        print("\n" + "="*60)
        print('GRAPH STATISTICS')
        print("="*60)
        print(f"  Number of nodes: {G.n}")
        print(f"  Number of edges: {int(np.sum(G.adjacency > 0))}")
        print(f"  Edge density: {calculate_edge_density(G, verbose=False):.3f}")
        print(f"  Is connected (spectral): {G.is_connected()}")
        print("="*60 + "\n")

    # Prepare training data
    if verbose:
        print("\n" + "="*60)
        print('DATA PREPARATION')
        print("="*60)
        print(f"  Generating {num_walks} walks...")
    
    walks = generate_multiple_walks(G, num_walks, min_walk_length, max_walk_length, rule_instances, verbose=verbose)
    training_data, vocab, corpus_metadata = prepare_training_data(
        G, num_walks, min_walk_length, max_walk_length, rule_instances, verbose=verbose
    )
    
    if verbose:
        print(f"\n  Training data prepared:")
        print(f"    Total walks: {len(walks)}")
        print(f"    Vocabulary size: {len(vocab)}")
        print(f"    Training tensor shape: {training_data.shape}")
        print("="*60 + "\n")

    # Train model
    if verbose:
        print(f'Training model')
    hidden_size = 256
    num_heads = 8
    model = train_model(
        training_data, vocab,
        hidden_size=hidden_size,
        num_layers=4,
        num_heads=num_heads,
        dropout=0.1,
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=learning_rate,
        context_window_size=context_window_size,
        verbose=verbose
    )
    if verbose:
        print(f'Model trained')

    # Create comprehensive experiment metadata
    graph_metadata = GraphMetadata(G, rule_instances)
    experiment_metadata = ExperimentMetadata(
        graph_metadata=graph_metadata,
        corpus_metadata=corpus_metadata,
        config=config
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(experiment_folder, "model.pth"))
    # Save dataset
    with open(os.path.join(experiment_folder, "data", "training_data.pkl"), "wb") as f:
        pickle.dump(training_data, f)
    # Save graph
    with open(os.path.join(experiment_folder, "data", "graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    # Save vocabulary
    with open(os.path.join(experiment_folder, "data", "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    # Save comprehensive metadata
    with open(os.path.join(experiment_folder, "data", "experiment_metadata.pkl"), "wb") as f:
        pickle.dump(experiment_metadata, f)

    # --- Enhanced: Evaluate model with token-level tracking and save metrics ---
    # NEW: Now returns trajectories as 7th return value
    evaluation_results, error_summary, kl_divergence_series, token_level_data, progressive_analysis, exemplar_walks, evaluation_trajectories = evaluate_model(
        model, G, vocab, num_walks, min_walk_length, max_walk_length, rule_instances, 
        verbose=verbose, track_token_details=True
    )
    save_error_summary(error_summary, experiment_folder)
    save_kl_divergence_series(kl_divergence_series, experiment_folder)
    save_token_level_data(token_level_data, experiment_folder)
    save_token_summary_stats(token_level_data, experiment_folder)
    save_rule_violation_analysis(token_level_data, experiment_folder)
    save_baseline_performance_summary(token_level_data, progressive_analysis, experiment_folder)
    save_exemplar_walks(exemplar_walks, experiment_folder)
    
    # NEW: Save trajectory metadata if available
    if evaluation_trajectories:
        from graphverse.utils.experiment_manager import save_trajectory_metadata
        save_trajectory_metadata(evaluation_trajectories, experiment_folder)
    # --- End enhanced code ---

    return model, G, vocab, rule_instances, token_level_data, experiment_metadata

if __name__ == "__main__":
    # Graph parameters
    n = 100  # 100 nodes for test
    
    # Context window and walk length (walks will be 2x context window)
    context_window_size = 5  # Context window size
    # Walk lengths will be automatically calculated as 2x context window = 10
    
    # Toggle between percentage-based and fixed-count rule definition
    use_percentages = True  # Set to False for fixed counts
    
    if use_percentages:
        # Rule parameters as percentages (0.0 to 100.0)
        num_ascenders = 2.0   # 2% of nodes are ascenders
        num_evens = 3.0       # 3% of nodes have even rules
        num_repeaters = 1.0   # 1% of nodes are repeaters
    else:
        # Rule parameters as fixed counts
        num_ascenders = 2     # Exactly 2 ascender nodes
        num_evens = 3         # Exactly 3 even rule nodes
        num_repeaters = 1     # Exactly 1 repeater node
    
    # Set repeater lengths to be distributed around the context window
    # This allows testing both learnable (k <= context_window) and unlearnable (k > context_window) patterns
    repeater_min_steps = max(2, context_window_size - 2)  # Start 2 below context window (min 2)
    repeater_max_steps = context_window_size + 2          # End 2 above context window
    
    # Edge weight configuration
    edge_concentration = 0.8  # Slightly non-uniform edge weights (0.8 < 1.0 = uniform)
    
    # Walk generation parameters
    num_walks = 1000  # 1000 walks for test
    
    # Training parameters
    epochs = 3  # Fewer epochs for test
    batch_size = 16
    learning_rate = 0.001
    verbose = True

    print("="*70)
    print("GRAPHVERSE EXPERIMENT")
    print("="*70)
    if use_percentages:
        print(f"Using percentage-based rules: {num_ascenders}% ascenders, {num_evens}% evens, {num_repeaters}% repeaters")
    else:
        print(f"Using fixed-count rules: {num_ascenders} ascenders, {num_evens} evens, {num_repeaters} repeaters")
    print("="*70)

    model, G, vocab, rule_instances, token_level_data, experiment_metadata = main(
        n, num_walks, context_window_size,
        num_ascenders, num_evens, num_repeaters, 
        repeater_min_steps, repeater_max_steps, 
        epochs, batch_size, learning_rate, verbose=verbose,
        use_percentages=use_percentages, edge_concentration=edge_concentration
    )
    
    # Print comprehensive experiment summary
    print("\n" + "="*70)
    print("EXPERIMENT ANALYSIS SUMMARY")
    print("="*70)
    
    summary = experiment_metadata.get_comprehensive_summary()
    
    # Graph composition
    graph_summary = summary['graph_metadata']
    print(f"Graph composition:")
    print(f"  Total nodes: {graph_summary['nodes']}")
    print(f"  Total edges: {graph_summary['edges']}")
    print(f"  Edge density: {graph_summary['edge_density']:.3f}")
    print(f"  Rule nodes: {graph_summary['rule_composition']['total_rule_nodes']} ({graph_summary['rule_composition']['total_rule_percentage']:.1f}%)")
    
    # Training corpus analysis
    corpus_summary = summary['corpus_metadata']
    print(f"\nTraining corpus:")
    print(f"  Total walks: {corpus_summary['basic_stats']['total_walks']}")
    print(f"  Unique sequences: {corpus_summary['basic_stats']['unique_sequences']}")
    print(f"  Sequence diversity: {corpus_summary['basic_stats']['sequence_diversity']:.3f}")
    
    if 'rule_exposure' in corpus_summary:
        print(f"  Rule exposure:")
        for rule_type, percent in corpus_summary['rule_exposure']['exposure_percentages'].items():
            print(f"    {rule_type}: {percent:.1f}%")
    
    # Analysis insights
    if 'analysis_metrics' in summary:
        analysis = summary['analysis_metrics']
        if 'rule_density_vs_exposure' in analysis:
            print(f"\nRule analysis insights:")
            for rule_type, data in analysis['rule_density_vs_exposure'].items():
                print(f"  {rule_type}: {data['exposure_amplification']:.1f}x exposure amplification")
    
    print("="*70)

    # Get the experiment folder name (latest)
    import glob
    experiment_folders = glob.glob("experiments/run_*")
    if experiment_folders:
        latest_folder = max(experiment_folders)
        
        # Paths to experiment outputs
        error_summary_path = f"{latest_folder}/evaluation/error_summary.json"
        kl_csv_path = f"{latest_folder}/evaluation/kl_divergence_timeseries.csv"

        # Original visualizations
        plot_error_summary(error_summary_path, output_path=f"{latest_folder}/evaluation/error_rates.png")
        plot_kl_divergence_timeseries(kl_csv_path, walk_idx=0, output_path=f"{latest_folder}/evaluation/kl_walk0.png")
        plot_aggregate_kl(kl_csv_path, output_path=f"{latest_folder}/evaluation/kl_aggregate.png")
        
        # Enhanced token-by-token visualizations
        plot_token_kl_heatmap(token_level_data, output_path=f"{latest_folder}/evaluation/token_kl_heatmap.png")
        plot_token_entropy_vs_kl(token_level_data, output_path=f"{latest_folder}/evaluation/entropy_vs_kl.png")
        plot_prediction_confidence_analysis(token_level_data, output_path=f"{latest_folder}/evaluation/prediction_analysis.png")
        
        print(f"All visualizations saved to: {latest_folder}/evaluation/")
    else:
        print("No experiment folders found")
