#!/usr/bin/env python3
"""
Large-Scale Experiment Runner

Runs the complete large-scale context boundary analysis experiment using
pre-trained models and the large-scale evaluation infrastructure.
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.large_scale_config import LARGE_SCALE_CONFIG
from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
from graphverse.llm.large_scale_evaluation import LargeScaleEvaluator
from graphverse.utils.multi_experiment_runner import MultiExperimentRunner
from graphverse.utils.memory_monitor import MemoryMonitor, MemoryOptimizer, monitor_large_scale_experiment
from graphverse.utils.checkpoint_manager import CheckpointManager, ProgressTracker
from graphverse.llm.evaluation import get_large_scale_trajectory_config
import torch


def load_graph_and_rules(graph_path="large_scale_graph"):
    """Load graph and rules from saved files."""
    print(f"Loading graph from {graph_path}...")
    
    # Load graph
    graph = Graph.load_graph(graph_path)
    
    # Load rule information
    with open(f"{graph_path}_rules.json", "r") as f:
        rule_info = json.load(f)
    
    # Recreate rule objects
    ascender_rule = AscenderRule(rule_info["ascender_nodes"])
    even_rule = EvenRule(rule_info["even_nodes"]) 
    repeater_rule = RepeaterRule(rule_info["repeater_nodes_dict"])
    
    rules = [ascender_rule, even_rule, repeater_rule]
    
    print(f"Graph loaded: {graph.n:,} vertices")
    print(f"Rules loaded: {len(ascender_rule.member_nodes)} ascenders, "
          f"{len(even_rule.member_nodes)} evens, {len(repeater_rule.member_nodes)} repeaters")
    
    return graph, rules, rule_info


def load_model_and_vocab(model_path, vocab_path, device='cpu'):
    """Load a trained model and vocabulary."""
    print(f"Loading model from {model_path}...")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Recreate model architecture
    from graphverse.llm.model import WalkTransformer
    model = WalkTransformer(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        max_length=model_config['max_length']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load vocabulary
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    print(f"Model loaded: context={model_config['max_length']}, vocab={len(vocab.token2idx)}")
    
    return model, vocab, model_config


def run_single_context_experiment(
    context_window, model_path, vocab_path, graph, rules, 
    config, output_base_dir, device='cpu', verbose=True
):
    """
    Run experiment for a single context window size.
    
    Args:
        context_window: Context window size
        model_path: Path to trained model
        vocab_path: Path to vocabulary
        graph: Graph object
        rules: List of rule objects  
        config: Large scale configuration
        output_base_dir: Base output directory
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Path to experiment results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT: CONTEXT WINDOW {context_window}")
        print(f"{'='*80}")
    
    # Load model and vocabulary
    model, vocab, model_config = load_model_and_vocab(model_path, vocab_path, device)
    
    # Create experiment folder
    exp_folder = os.path.join(output_base_dir, f"context_{context_window}")
    os.makedirs(exp_folder, exist_ok=True)
    
    # Set up trajectory sampling configuration
    trajectory_config = get_large_scale_trajectory_config(
        num_walks=config["num_walks"],
        sample_rate=config["trajectory_sampling"]["sample_rate"],
        stratified=config["trajectory_sampling"]["stratified"]
    )
    
    # Set up memory monitoring
    memory_monitor = MemoryMonitor(
        memory_limit_gb=config["batch_processing"]["memory_limit_gb"],
        alert_threshold=0.8,
        critical_threshold=0.95
    )
    
    if verbose:
        print(f"Trajectory sampling: {trajectory_config['description']}")
        print(f"Memory limit: {config['batch_processing']['memory_limit_gb']}GB")
    
    # Create large-scale evaluator with config
    evaluator = LargeScaleEvaluator(
        experiment_folder=exp_folder,
        checkpoint_frequency=config["evaluation"]["checkpoint_frequency"],
        experiment_config=config  # Pass full config for distribution analysis
    )
    
    # Start memory monitoring
    memory_monitor.start_monitoring(verbose=verbose)
    
    try:
        # Run large-scale evaluation
        start_time = time.time()
        
        result_folder = evaluator.evaluate_large_scale(
            model=model,
            graph=graph,
            vocab=vocab,
            num_walks=config["num_walks"],
            min_start_length=3,
            max_start_length=8,
            rules=rules,
            batch_size=config["batch_processing"]["walk_batch_size"],
            trajectory_sampling_config=trajectory_config,
            resume_from_checkpoint=True,
            verbose=verbose
        )
        
        experiment_time = time.time() - start_time
        
        # Stop memory monitoring
        memory_monitor.stop_monitoring()
        memory_summary = memory_monitor.get_memory_summary()
        
        # Save experiment metadata
        experiment_metadata = {
            "context_window": context_window,
            "model_config": model_config,
            "trajectory_config": trajectory_config,
            "experiment_time": experiment_time,
            "memory_summary": memory_summary,
            "num_walks": config["num_walks"],
            "batch_size": config["batch_processing"]["walk_batch_size"],
            "completion_time": time.time()
        }
        
        metadata_path = os.path.join(exp_folder, "experiment_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(experiment_metadata, f, indent=2, default=str)
        
        if verbose:
            print(f"\n✅ Context {context_window} completed successfully!")
            print(f"   Experiment time: {experiment_time/3600:.1f} hours")
            print(f"   Peak memory: {memory_summary.get('peak_memory_gb', 0):.1f}GB")
            print(f"   Results saved to: {result_folder}")
        
        return result_folder
        
    except Exception as e:
        memory_monitor.stop_monitoring()
        print(f"❌ Error in context {context_window} experiment: {e}")
        raise e


def run_all_experiments(
    graph_path="large_scale_graph",
    models_dir="large_scale_models", 
    output_dir="large_scale_results",
    context_windows=None,
    device='auto',
    verbose=True
):
    """
    Run experiments for all context window sizes.
    
    Args:
        graph_path: Path to graph files
        models_dir: Directory containing trained models
        output_dir: Output directory for results
        context_windows: List of context windows (None for config default)
        device: Device to run on ('auto', 'cpu', 'cuda')
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping context_window -> result_path
    """
    if context_windows is None:
        context_windows = LARGE_SCALE_CONFIG["context_windows"]
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print("=" * 80)
        print("LARGE-SCALE CONTEXT BOUNDARY EXPERIMENT")
        print("=" * 80)
        print(f"Context windows: {context_windows}")
        print(f"Target walks per context: {LARGE_SCALE_CONFIG['num_walks']:,}")
        print(f"Total walks: {len(context_windows) * LARGE_SCALE_CONFIG['num_walks']:,}")
        print(f"Device: {device}")
        print(f"Output directory: {output_dir}")
    
    # Load graph and rules
    graph, rules, rule_info = load_graph_and_rules(graph_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall experiment configuration
    overall_config = {
        "experiment_type": "large_scale_context_boundary_analysis",
        "context_windows": context_windows,
        "config": LARGE_SCALE_CONFIG,
        "graph_path": graph_path,
        "models_dir": models_dir,
        "rule_info": rule_info,
        "device": device,
        "start_time": time.time()
    }
    
    config_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(overall_config, f, indent=2, default=str)
    
    # Track overall progress
    checkpoint_manager = CheckpointManager(output_dir)
    progress_tracker = ProgressTracker(
        experiment_name="large_scale_context_analysis",
        total_items=len(context_windows) * LARGE_SCALE_CONFIG["num_walks"],
        checkpoint_manager=checkpoint_manager,
        checkpoint_frequency=LARGE_SCALE_CONFIG["num_walks"]  # Checkpoint after each context
    )
    
    results = {}
    total_start_time = time.time()
    
    for i, context_window in enumerate(context_windows):
        if verbose:
            print(f"\n{'='*60}")
            print(f"CONTEXT {i+1}/{len(context_windows)}: {context_window}")
            print(f"{'='*60}")
        
        try:
            # Find model and vocab files for this context
            model_path = os.path.join(models_dir, f"model_ctx_{context_window}.pt")
            vocab_path = os.path.join(models_dir, f"vocab_ctx_{context_window}.pkl")
            
            if not os.path.exists(model_path):
                print(f"❌ Model not found: {model_path}")
                continue
            
            if not os.path.exists(vocab_path):
                print(f"❌ Vocabulary not found: {vocab_path}")
                continue
            
            # Run experiment for this context
            result_folder = run_single_context_experiment(
                context_window=context_window,
                model_path=model_path,
                vocab_path=vocab_path,
                graph=graph,
                rules=rules,
                config=LARGE_SCALE_CONFIG,
                output_base_dir=output_dir,
                device=device,
                verbose=verbose
            )
            
            results[context_window] = result_folder
            
            # Update progress
            progress_tracker.update_progress(
                items_completed=LARGE_SCALE_CONFIG["num_walks"],
                context_window=context_window,
                completed_contexts=i+1
            )
            
            if verbose:
                elapsed = time.time() - total_start_time
                remaining = len(context_windows) - (i + 1)
                avg_time = elapsed / (i + 1)
                eta = remaining * avg_time
                
                print(f"✅ Context {context_window} completed")
                print(f"   Progress: {i+1}/{len(context_windows)} contexts")
                print(f"   Elapsed: {elapsed/3600:.1f}h, ETA: {eta/3600:.1f}h")
            
        except Exception as e:
            print(f"❌ Error in context {context_window}: {e}")
            import traceback
            traceback.print_exc()
            
            # Record error but continue with other contexts
            progress_tracker.record_error(f"Context {context_window}: {e}")
    
    total_time = time.time() - total_start_time
    
    # Save final results mapping
    results_mapping = {
        "results": results,
        "total_time": total_time,
        "completed_contexts": len(results),
        "failed_contexts": len(context_windows) - len(results),
        "completion_time": time.time()
    }
    
    results_path = os.path.join(output_dir, "results_mapping.json")
    with open(results_path, "w") as f:
        json.dump(results_mapping, f, indent=2)
    
    if verbose:
        print(f"\n{'='*80}")
        print("LARGE-SCALE EXPERIMENT COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {total_time/3600:.1f} hours")
        print(f"Completed contexts: {len(results)}/{len(context_windows)}")
        print(f"Results directory: {output_dir}")
        
        print("\nCompleted experiments:")
        for ctx, path in results.items():
            print(f"  Context {ctx}: {path}")
        
        if len(results) < len(context_windows):
            failed = set(context_windows) - set(results.keys())
            print(f"\nFailed contexts: {failed}")
    
    return results


def analyze_experiment_results(results_dir, verbose=True):
    """
    Analyze and summarize experiment results.
    
    Args:
        results_dir: Directory containing experiment results
        verbose: Whether to print analysis
        
    Returns:
        Analysis summary
    """
    if verbose:
        print(f"\n{'='*60}")
        print("EXPERIMENT ANALYSIS")
        print(f"{'='*60}")
    
    # Load results mapping
    results_path = os.path.join(results_dir, "results_mapping.json")
    if not os.path.exists(results_path):
        print(f"❌ Results mapping not found: {results_path}")
        return None
    
    with open(results_path, "r") as f:
        results_mapping = json.load(f)
    
    analysis = {
        "total_contexts": len(results_mapping["results"]),
        "total_time_hours": results_mapping["total_time"] / 3600,
        "context_summaries": {}
    }
    
    for context_window, result_path in results_mapping["results"].items():
        context_window = int(context_window)  # JSON keys are strings
        
        # Load experiment metadata
        metadata_path = os.path.join(result_path, "experiment_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Load final results
            final_results_path = os.path.join(result_path, "evaluation", "final_results.json")
            if os.path.exists(final_results_path):
                with open(final_results_path, "r") as f:
                    final_results = json.load(f)
                
                error_summary = final_results.get("aggregated_error_summary", {})
                distribution_summary = final_results.get("aggregated_distribution_metrics", {})
                
                context_summary = {
                    "context_window": context_window,
                    "experiment_time_hours": metadata.get("experiment_time", 0) / 3600,
                    "total_walks": final_results.get("total_walks", 0),
                    "peak_memory_gb": metadata.get("memory_summary", {}).get("peak_memory_gb", 0),
                    "repeater_error_rate": error_summary.get("repeater_error_rate", 0),
                    "ascender_error_rate": error_summary.get("ascender_error_rate", 0),
                    "even_error_rate": error_summary.get("even_error_rate", 0),
                    "avg_steps_per_walk": error_summary.get("avg_steps_per_walk", 0),
                    # Add distribution metrics
                    "avg_kl_from_graph": distribution_summary.get("kl_from_graph_mean", 0),
                    "avg_structural_awareness": distribution_summary.get("structural_awareness_mean", 0),
                    "avg_overall_quality": distribution_summary.get("overall_quality_mean", 0),
                    "avg_top1_agreement": distribution_summary.get("top1_agreement_with_graph_mean", 0)
                }
                
                analysis["context_summaries"][context_window] = context_summary
                
                if verbose:
                    print(f"Context {context_window}:")
                    print(f"  Walks: {context_summary['total_walks']:,}")
                    print(f"  Time: {context_summary['experiment_time_hours']:.1f}h")
                    print(f"  Memory: {context_summary['peak_memory_gb']:.1f}GB")
                    print(f"  Repeater errors: {context_summary['repeater_error_rate']:.2%}")
                    print(f"  Steps/walk: {context_summary['avg_steps_per_walk']:.1f}")
                    print(f"  KL from graph: {context_summary['avg_kl_from_graph']:.4f}")
                    print(f"  Structural awareness: {context_summary['avg_structural_awareness']:.4f}")
                    print(f"  Quality score: {context_summary['avg_overall_quality']:.4f}")
                    print(f"  Top-1 agreement: {context_summary['avg_top1_agreement']:.4f}")
    
    if verbose:
        print(f"\nOverall Summary:")
        print(f"  Total contexts: {analysis['total_contexts']}")
        print(f"  Total time: {analysis['total_time_hours']:.1f} hours")
        
        if analysis["context_summaries"]:
            total_walks = sum(cs["total_walks"] for cs in analysis["context_summaries"].values())
            avg_repeater_error = sum(cs["repeater_error_rate"] for cs in analysis["context_summaries"].values()) / len(analysis["context_summaries"])
            print(f"  Total walks: {total_walks:,}")
            print(f"  Avg repeater error: {avg_repeater_error:.2%}")
    
    # Generate all visualization types
    if verbose:
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*60)
    
    try:
        # Create distribution comparison plots
        create_distribution_comparison_plots(analysis, results_dir, verbose)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not generate distribution visualizations: {e}")
    
    try:
        # Create rule-specific error rate plots
        from graphverse.llm.evaluation_vis import plot_rule_specific_error_rates_by_context
        context_windows = sorted([int(k) for k in results_mapping["results"].keys()])
        plot_path = os.path.join(results_dir, "rule_specific_error_rates_by_context.png")
        plot_rule_specific_error_rates_by_context(
            results_mapping["results"], context_windows, plot_path, figsize=(16, 10)
        )
    except Exception as e:
        if verbose:
            print(f"Warning: Could not generate rule-specific error plots: {e}")
    
    try:
        # Create exemplar and walk-level aggregate plots
        from graphverse.llm.evaluation_vis import plot_distributional_metric_exemplars, plot_walk_level_distributional_aggregates
        import pickle
        
        # Collect token-level data from batch files across experiments
        all_token_data = []
        data_collected = 0
        max_tokens_per_experiment = 5000  # Limit data collection for memory
        
        for context_window, result_path in results_mapping["results"].items():
            if data_collected >= max_tokens_per_experiment * 3:  # Limit total collection
                break
                
            batch_dir = os.path.join(result_path, "batches")
            if not os.path.exists(batch_dir):
                continue
                
            # Load a sample of batch files
            batch_files = [f for f in os.listdir(batch_dir) if f.endswith('.pkl')][:3]  # First 3 batches
            
            for batch_file in batch_files:
                batch_path = os.path.join(batch_dir, batch_file)
                try:
                    with open(batch_path, "rb") as f:
                        batch_data = pickle.load(f)
                    
                    token_data = batch_data.get("token_data", [])
                    # Sample token data to manage memory
                    if len(token_data) > max_tokens_per_experiment:
                        import random
                        token_data = random.sample(token_data, max_tokens_per_experiment)
                    
                    # Filter for tokens with distribution comparison data
                    comparison_tokens = [t for t in token_data if 'core_distribution_comparison' in t]
                    if comparison_tokens:
                        all_token_data.extend(comparison_tokens)
                        data_collected += len(comparison_tokens)
                        
                    if data_collected >= max_tokens_per_experiment * 3:
                        break
                        
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not load batch file {batch_file}: {e}")
                    continue
        
        if all_token_data:
            if verbose:
                print(f"Collected {len(all_token_data)} tokens with distribution data")
            
            # Create exemplar plots
            exemplar_path = os.path.join(results_dir, "distributional_metric_exemplars.png")
            plot_distributional_metric_exemplars(
                all_token_data, exemplar_path, figsize=(18, 12), n_exemplars=6
            )
            
            # Create walk-level aggregate plots
            aggregate_path = os.path.join(results_dir, "walk_level_distributional_aggregates.png")
            plot_walk_level_distributional_aggregates(
                all_token_data, aggregate_path, figsize=(16, 10)
            )
        else:
            if verbose:
                print("Warning: No token-level distribution data found for exemplar plots")
                
    except Exception as e:
        if verbose:
            print(f"Warning: Could not generate exemplar plots: {e}")
    
    return analysis


def create_distribution_comparison_plots(analysis, results_dir, verbose=True):
    """Create distribution comparison plots across context windows."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not analysis["context_summaries"]:
        return
    
    # Extract data for plotting
    contexts = sorted(analysis["context_summaries"].keys())
    kl_from_graph = [analysis["context_summaries"][c]["avg_kl_from_graph"] for c in contexts]
    structural_awareness = [analysis["context_summaries"][c]["avg_structural_awareness"] for c in contexts]
    quality_scores = [analysis["context_summaries"][c]["avg_overall_quality"] for c in contexts]
    top1_agreement = [analysis["context_summaries"][c]["avg_top1_agreement"] for c in contexts]
    repeater_errors = [analysis["context_summaries"][c]["repeater_error_rate"] for c in contexts]
    
    # Create multi-panel plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Distribution Comparison Analysis Across Context Windows", fontsize=16, fontweight='bold')
    
    # Plot 1: KL Divergence from Graph Structure
    axes[0, 0].plot(contexts, kl_from_graph, 'o-', color='red', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Context Window Size")
    axes[0, 0].set_ylabel("Average KL Divergence")
    axes[0, 0].set_title("KL Divergence from Graph Structure")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log', base=2)
    
    # Plot 2: Structural Awareness
    axes[0, 1].plot(contexts, structural_awareness, 'o-', color='blue', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel("Context Window Size")
    axes[0, 1].set_ylabel("Structural Awareness Score")
    axes[0, 1].set_title("Model's Graph Structure Awareness")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: Overall Quality Score
    axes[0, 2].plot(contexts, quality_scores, 'o-', color='green', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel("Context Window Size")
    axes[0, 2].set_ylabel("Overall Quality Score")
    axes[0, 2].set_title("Overall Prediction Quality")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xscale('log', base=2)
    axes[0, 2].set_ylim([0, 1])
    
    # Plot 4: Top-1 Agreement with Graph
    axes[1, 0].plot(contexts, top1_agreement, 'o-', color='purple', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel("Context Window Size")
    axes[1, 0].set_ylabel("Top-1 Agreement Rate")
    axes[1, 0].set_title("Top-1 Agreement with Graph Structure")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log', base=2)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 5: Repeater Error vs KL Divergence
    axes[1, 1].scatter(kl_from_graph, repeater_errors, c=contexts, cmap='viridis', s=100, alpha=0.7)
    axes[1, 1].set_xlabel("Average KL Divergence from Graph")
    axes[1, 1].set_ylabel("Repeater Error Rate")
    axes[1, 1].set_title("Repeater Errors vs Distribution Divergence")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar for context windows
    scatter = axes[1, 1].scatter(kl_from_graph, repeater_errors, c=contexts, cmap='viridis', s=100, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Context Window Size')
    
    # Plot 6: Multiple metrics comparison
    x = np.arange(len(contexts))
    width = 0.2
    
    axes[1, 2].bar(x - width, structural_awareness, width, label='Structural Awareness', alpha=0.8)
    axes[1, 2].bar(x, quality_scores, width, label='Overall Quality', alpha=0.8)
    axes[1, 2].bar(x + width, top1_agreement, width, label='Top-1 Agreement', alpha=0.8)
    
    axes[1, 2].set_xlabel("Context Window Size")
    axes[1, 2].set_ylabel("Score")
    axes[1, 2].set_title("Distribution Quality Metrics Comparison")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels([str(c) for c in contexts])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, "distribution_comparison_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✅ Distribution comparison plots saved to: {plot_path}")
    
    # Create summary statistics plot
    create_distribution_summary_plot(analysis, results_dir, verbose)


def create_distribution_summary_plot(analysis, results_dir, verbose=True):
    """Create a summary plot of key distribution metrics."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    contexts = sorted(analysis["context_summaries"].keys())
    
    # Create correlation matrix of key metrics
    metrics_data = {
        'KL from Graph': [analysis["context_summaries"][c]["avg_kl_from_graph"] for c in contexts],
        'Structural Awareness': [analysis["context_summaries"][c]["avg_structural_awareness"] for c in contexts],
        'Quality Score': [analysis["context_summaries"][c]["avg_overall_quality"] for c in contexts],
        'Top-1 Agreement': [analysis["context_summaries"][c]["avg_top1_agreement"] for c in contexts],
        'Repeater Errors': [analysis["context_summaries"][c]["repeater_error_rate"] for c in contexts]
    }
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Distribution Comparison Summary", fontsize=14, fontweight='bold')
    
    # Metrics by context window heatmap
    metric_names = list(metrics_data.keys())
    data_matrix = np.array([metrics_data[name] for name in metric_names])
    
    im1 = ax1.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
    ax1.set_xticks(range(len(contexts)))
    ax1.set_xticklabels(contexts)
    ax1.set_yticks(range(len(metric_names)))
    ax1.set_yticklabels(metric_names)
    ax1.set_xlabel("Context Window Size")
    ax1.set_title("Metrics by Context Window")
    
    # Add text annotations
    for i in range(len(metric_names)):
        for j in range(len(contexts)):
            text = ax1.text(j, i, f'{data_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=ax1)
    
    # Correlation matrix
    correlation_matrix = np.corrcoef(data_matrix)
    im2 = ax2.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(metric_names)))
    ax2.set_xticklabels(metric_names, rotation=45, ha='right')
    ax2.set_yticks(range(len(metric_names)))
    ax2.set_yticklabels(metric_names)
    ax2.set_title("Metric Correlations")
    
    # Add correlation values
    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            text = ax2.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", 
                           color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                           fontsize=9)
    
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    
    # Save plot
    summary_plot_path = os.path.join(results_dir, "distribution_summary_analysis.png")
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"✅ Distribution summary plots saved to: {summary_plot_path}")


def main():
    """Main experiment runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run large-scale context boundary analysis experiment")
    parser.add_argument("--graph", "-g", default="large_scale_graph",
                       help="Path to graph files (default: large_scale_graph)")
    parser.add_argument("--models", "-m", default="large_scale_models",
                       help="Directory containing trained models (default: large_scale_models)")
    parser.add_argument("--output", "-o", default="large_scale_results",
                       help="Output directory for results (default: large_scale_results)")
    parser.add_argument("--contexts", nargs="+", type=int,
                       help="Context window sizes to run (default: from config)")
    parser.add_argument("--device", choices=['auto', 'cpu', 'cuda'], default='auto',
                       help="Device to run on (default: auto)")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze existing results instead of running experiment")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    context_windows = args.contexts if args.contexts else None
    
    try:
        if args.analyze:
            # Analyze existing results
            analysis = analyze_experiment_results(args.output, verbose)
            if analysis is None:
                sys.exit(1)
        else:
            # Run the experiment
            results = run_all_experiments(
                graph_path=args.graph,
                models_dir=args.models,
                output_dir=args.output,
                context_windows=context_windows,
                device=args.device,
                verbose=verbose
            )
            
            if not results:
                print("❌ No experiments completed successfully!")
                sys.exit(1)
            
            # Analyze results
            if verbose:
                analyze_experiment_results(args.output, verbose)
        
        if verbose:
            print(f"\n✅ Large-scale experiment pipeline completed!")
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()