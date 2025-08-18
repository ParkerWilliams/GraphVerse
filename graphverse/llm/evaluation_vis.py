import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_error_summary(error_summary_path, output_path=None):
    with open(error_summary_path, "r") as f:
        summary = json.load(f)
    labels = []
    values = []
    for k, v in summary.items():
        if k.endswith("_rate"):
            labels.append(k.replace("_error_rate", ""))
            values.append(v)
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Error Rate")
    plt.title("Model Error Rates")
    plt.ylim(0, 1)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()

def plot_kl_divergence_timeseries(kl_csv_path, walk_idx=0, output_path=None, zoom_steps=None):
    # Read KL series for all walks
    walk_kl = []
    with open(kl_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["walk_idx"])
            if idx == walk_idx:
                walk_kl.append((int(row["step_idx"]), float(row["kl_divergence"])))
    if not walk_kl:
        print(f"No KL data for walk {walk_idx}")
        return
    walk_kl.sort()
    steps, kls = zip(*walk_kl)
    if zoom_steps is not None:
        steps = steps[-zoom_steps:]
        kls = kls[-zoom_steps:]
    plt.figure(figsize=(10, 4))
    plt.plot(steps, kls, marker="o")
    plt.xlabel("Step")
    plt.ylabel("KL Divergence")
    plt.title(f"KL Divergence Time Series (Walk {walk_idx})")
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()

def plot_aggregate_kl(kl_csv_path, output_path=None):
    # Aggregate KL by step across all walks
    step_kl = {}
    with open(kl_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step_idx"])
            kl = float(row["kl_divergence"])
            step_kl.setdefault(step, []).append(kl)
    steps = sorted(step_kl.keys())
    means = [np.mean(step_kl[s]) for s in steps]
    stds = [np.std(step_kl[s]) for s in steps]
    plt.figure(figsize=(10, 4))
    plt.plot(steps, means, label="Mean KL")
    plt.fill_between(steps, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.3, label="Std Dev")
    plt.xlabel("Step")
    plt.ylabel("KL Divergence")
    plt.title("Aggregate KL Divergence Across Walks")
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def plot_distribution_comparison_dashboard(token_level_data, output_path=None, figsize=(20, 12)):
    """
    Create a comprehensive dashboard showing LLM distribution comparisons vs baselines.
    
    Args:
        token_level_data: List of token-level dictionaries with core_distribution_comparison
        output_path: Path to save the plot
        figsize: Figure size tuple
    """
    # Filter data to only include tokens with distribution comparisons
    comparison_data = []
    for token in token_level_data:
        if 'core_distribution_comparison' in token:
            comparison_data.append(token)
    
    if not comparison_data:
        print("No distribution comparison data found")
        return
    
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    fig.suptitle("LLM Distribution vs Baseline Comparisons", fontsize=16, fontweight='bold')
    
    # Extract metrics for plotting
    steps = [d['step_idx'] for d in comparison_data]
    
    # 1. KL Divergences from different baselines
    baselines = ['graph_structure', 'uniform_valid', 'exponential_fitted', 'uniform_full']
    colors = ['red', 'blue', 'green', 'orange']
    
    ax = axes[0, 0]
    for i, baseline in enumerate(baselines):
        kl_values = []
        for d in comparison_data:
            dist_comp = d['core_distribution_comparison']
            if baseline in dist_comp['distribution_distances']:
                kl_values.append(dist_comp['distribution_distances'][baseline]['kl_divergence'])
            else:
                kl_values.append(0)
        ax.plot(steps, kl_values, label=baseline.replace('_', ' ').title(), color=colors[i], alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence from Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cosine Similarity to baselines
    ax = axes[0, 1]
    for i, baseline in enumerate(baselines):
        cosine_values = []
        for d in comparison_data:
            dist_comp = d['core_distribution_comparison']
            if baseline in dist_comp['distribution_distances']:
                cosine_values.append(dist_comp['distribution_distances'][baseline]['cosine_similarity'])
            else:
                cosine_values.append(0)
        ax.plot(steps, cosine_values, label=baseline.replace('_', ' ').title(), color=colors[i], alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity to Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Valid Neighbor Focus
    ax = axes[0, 2]
    valid_masses = []
    for d in comparison_data:
        dist_comp = d['core_distribution_comparison']
        model_stats = dist_comp['model_distribution_stats']
        # Compute valid neighbor mass for model
        # This would need to be added to model_distribution_stats
        valid_masses.append(0.5)  # Placeholder
    ax.plot(steps, valid_masses, 'purple', alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Valid Neighbor Mass")
    ax.set_title("Model Focus on Valid Neighbors")
    ax.grid(True, alpha=0.3)
    
    # 4. Overall Prediction Quality
    ax = axes[0, 3]
    quality_scores = []
    for d in comparison_data:
        dist_comp = d['core_distribution_comparison']
        if 'prediction_quality_scores' in dist_comp:
            quality_scores.append(dist_comp['prediction_quality_scores'].get('overall_quality', 0))
        else:
            quality_scores.append(0)
    ax.plot(steps, quality_scores, 'darkgreen', alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Quality Score")
    ax.set_title("Overall Prediction Quality")
    ax.grid(True, alpha=0.3)
    
    # 5. Distribution Overlap Analysis
    ax = axes[1, 0]
    for i, baseline in enumerate(['graph_structure', 'uniform_valid']):
        overlap_values = []
        for d in comparison_data:
            dist_comp = d['core_distribution_comparison']
            if baseline in dist_comp['distribution_overlap_analysis']:
                overlap_values.append(dist_comp['distribution_overlap_analysis'][baseline]['overlap_coefficient'])
            else:
                overlap_values.append(0)
        ax.plot(steps, overlap_values, label=baseline.replace('_', ' ').title(), color=colors[i], alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Overlap Coefficient")
    ax.set_title("Distribution Overlap with Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Entropy Comparison
    ax = axes[1, 1]
    model_entropies = []
    graph_entropies = []
    for d in comparison_data:
        model_entropies.append(d['entropy'])
        dist_comp = d['core_distribution_comparison']
        if 'graph_structure' in dist_comp['baseline_distributions']:
            graph_entropies.append(dist_comp['baseline_distributions']['graph_structure']['entropy'])
        else:
            graph_entropies.append(0)
    
    ax.plot(steps, model_entropies, label='Model', color='red', alpha=0.7)
    ax.plot(steps, graph_entropies, label='Graph Structure', color='blue', alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy")
    ax.set_title("Entropy: Model vs Graph Structure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Structural Awareness
    ax = axes[1, 2]
    structural_awareness = []
    for d in comparison_data:
        dist_comp = d['core_distribution_comparison']
        if 'prediction_quality_scores' in dist_comp:
            structural_awareness.append(dist_comp['prediction_quality_scores'].get('structural_awareness', 0))
        else:
            structural_awareness.append(0)
    ax.plot(steps, structural_awareness, 'teal', alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Structural Awareness")
    ax.set_title("Model's Graph Structure Awareness")
    ax.grid(True, alpha=0.3)
    
    # 8. Top-k Agreement
    ax = axes[1, 3]
    top_1_agreement = []
    top_5_agreement = []
    for d in comparison_data:
        dist_comp = d['core_distribution_comparison']
        if 'graph_structure' in dist_comp['distribution_overlap_analysis']:
            overlap_analysis = dist_comp['distribution_overlap_analysis']['graph_structure']
            top_1_agreement.append(overlap_analysis['agreement_on_top_k'].get('top_1', 0))
            top_5_agreement.append(overlap_analysis['agreement_on_top_k'].get('top_5', 0))
        else:
            top_1_agreement.append(0)
            top_5_agreement.append(0)
    
    ax.plot(steps, top_1_agreement, label='Top-1 Agreement', color='red', alpha=0.7)
    ax.plot(steps, top_5_agreement, label='Top-5 Agreement', color='blue', alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Agreement Rate")
    ax.set_title("Model-Graph Top-k Agreement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. JS Divergence comparison
    ax = axes[2, 0]
    for i, baseline in enumerate(['graph_structure', 'uniform_valid']):
        js_values = []
        for d in comparison_data:
            dist_comp = d['core_distribution_comparison']
            if baseline in dist_comp['distribution_distances']:
                js_values.append(dist_comp['distribution_distances'][baseline]['js_divergence'])
            else:
                js_values.append(0)
        ax.plot(steps, js_values, label=baseline.replace('_', ' ').title(), color=colors[i], alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("JS Divergence")
    ax.set_title("Jensen-Shannon Divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 10. L1 and L2 Distance
    ax = axes[2, 1]
    l1_distances = []
    l2_distances = []
    for d in comparison_data:
        dist_comp = d['core_distribution_comparison']
        if 'graph_structure' in dist_comp['distribution_distances']:
            l1_distances.append(dist_comp['distribution_distances']['graph_structure']['l1_distance'])
            l2_distances.append(dist_comp['distribution_distances']['graph_structure']['l2_distance'])
        else:
            l1_distances.append(0)
            l2_distances.append(0)
    
    ax.plot(steps, l1_distances, label='L1 Distance', color='red', alpha=0.7)
    ax.plot(steps, l2_distances, label='L2 Distance', color='blue', alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.set_title("L1/L2 Distance from Graph Structure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 11. Concentration Quality
    ax = axes[2, 2]
    concentration_quality = []
    for d in comparison_data:
        dist_comp = d['core_distribution_comparison']
        if 'prediction_quality_scores' in dist_comp:
            concentration_quality.append(dist_comp['prediction_quality_scores'].get('concentration_quality', 0))
        else:
            concentration_quality.append(0)
    ax.plot(steps, concentration_quality, 'purple', alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Concentration Quality")
    ax.set_title("Model Concentration Quality")
    ax.grid(True, alpha=0.3)
    
    # 12. Summary Statistics
    ax = axes[2, 3]
    # Show average divergences as bar chart
    avg_kl_graph = np.mean([d['core_distribution_comparison']['distribution_distances']['graph_structure']['kl_divergence'] 
                           for d in comparison_data if 'graph_structure' in d['core_distribution_comparison']['distribution_distances']])
    avg_kl_uniform = np.mean([d['core_distribution_comparison']['distribution_distances']['uniform_valid']['kl_divergence'] 
                             for d in comparison_data if 'uniform_valid' in d['core_distribution_comparison']['distribution_distances']])
    avg_kl_exp = np.mean([d['core_distribution_comparison']['distribution_distances']['exponential_fitted']['kl_divergence'] 
                         for d in comparison_data if 'exponential_fitted' in d['core_distribution_comparison']['distribution_distances']])
    
    baseline_names = ['Graph\nStructure', 'Uniform\nValid', 'Exponential\nFitted']
    avg_kls = [avg_kl_graph, avg_kl_uniform, avg_kl_exp]
    colors_bar = ['red', 'blue', 'green']
    
    bars = ax.bar(baseline_names, avg_kls, color=colors_bar, alpha=0.7)
    ax.set_ylabel("Average KL Divergence")
    ax.set_title("Average KL Divergence by Baseline")
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_kls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Distribution comparison dashboard saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_baseline_comparison_summary(token_level_data, output_path=None, figsize=(15, 10)):
    """
    Create summary plots comparing model performance against different baselines.
    
    Args:
        token_level_data: List of token-level dictionaries with distribution comparisons
        output_path: Path to save the plot
        figsize: Figure size tuple
    """
    # Extract comparison data
    comparison_data = [t for t in token_level_data if 'core_distribution_comparison' in t]
    
    if not comparison_data:
        print("No distribution comparison data found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Model vs Baseline Distribution Analysis Summary", fontsize=14, fontweight='bold')
    
    baselines = ['graph_structure', 'uniform_valid', 'exponential_fitted']
    baseline_labels = ['Graph Structure', 'Uniform Valid', 'Exponential Fitted']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # 1. Average KL Divergence
    ax = axes[0, 0]
    avg_kls = []
    for baseline in baselines:
        kl_values = []
        for d in comparison_data:
            if baseline in d['core_distribution_comparison']['distribution_distances']:
                kl_values.append(d['core_distribution_comparison']['distribution_distances'][baseline]['kl_divergence'])
        avg_kls.append(np.mean(kl_values) if kl_values else 0)
    
    bars = ax.bar(baseline_labels, avg_kls, color=colors, alpha=0.7)
    ax.set_ylabel("Average KL Divergence")
    ax.set_title("KL Divergence from Baselines")
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, avg_kls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Average Cosine Similarity
    ax = axes[0, 1]
    avg_cosines = []
    for baseline in baselines:
        cosine_values = []
        for d in comparison_data:
            if baseline in d['core_distribution_comparison']['distribution_distances']:
                cosine_values.append(d['core_distribution_comparison']['distribution_distances'][baseline]['cosine_similarity'])
        avg_cosines.append(np.mean(cosine_values) if cosine_values else 0)
    
    bars = ax.bar(baseline_labels, avg_cosines, color=colors, alpha=0.7)
    ax.set_ylabel("Average Cosine Similarity")
    ax.set_title("Cosine Similarity to Baselines")
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, avg_cosines):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Distribution Overlap
    ax = axes[0, 2]
    avg_overlaps = []
    for baseline in baselines:
        overlap_values = []
        for d in comparison_data:
            if baseline in d['core_distribution_comparison']['distribution_overlap_analysis']:
                overlap_values.append(d['core_distribution_comparison']['distribution_overlap_analysis'][baseline]['overlap_coefficient'])
        avg_overlaps.append(np.mean(overlap_values) if overlap_values else 0)
    
    bars = ax.bar(baseline_labels, avg_overlaps, color=colors, alpha=0.7)
    ax.set_ylabel("Average Overlap Coefficient")
    ax.set_title("Distribution Overlap")
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, avg_overlaps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Top-k Agreement (using graph_structure baseline)
    ax = axes[1, 0]
    top_k_labels = ['Top-1', 'Top-3', 'Top-5', 'Top-10']
    top_k_agreements = []
    
    for k_label in top_k_labels:
        k_key = k_label.lower().replace('-', '_')
        agreement_values = []
        for d in comparison_data:
            if 'graph_structure' in d['core_distribution_comparison']['distribution_overlap_analysis']:
                agreement_data = d['core_distribution_comparison']['distribution_overlap_analysis']['graph_structure']['agreement_on_top_k']
                agreement_values.append(agreement_data.get(k_key, 0))
        top_k_agreements.append(np.mean(agreement_values) if agreement_values else 0)
    
    bars = ax.bar(top_k_labels, top_k_agreements, color='#9b59b6', alpha=0.7)
    ax.set_ylabel("Agreement Rate")
    ax.set_title("Top-k Agreement with Graph Structure")
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, top_k_agreements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 5. Quality Scores
    ax = axes[1, 1]
    quality_dimensions = ['structural_awareness', 'concentration_quality', 'distributional_fit', 'neighbor_prioritization']
    quality_labels = ['Structural\nAwareness', 'Concentration\nQuality', 'Distributional\nFit', 'Neighbor\nPrioritization']
    
    avg_qualities = []
    for dimension in quality_dimensions:
        quality_values = []
        for d in comparison_data:
            if 'prediction_quality_scores' in d['core_distribution_comparison']:
                quality_values.append(d['core_distribution_comparison']['prediction_quality_scores'].get(dimension, 0))
        avg_qualities.append(np.mean(quality_values) if quality_values else 0)
    
    bars = ax.bar(quality_labels, avg_qualities, color='#f39c12', alpha=0.7)
    ax.set_ylabel("Quality Score")
    ax.set_title("Prediction Quality Dimensions")
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, avg_qualities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Distance Metrics Heatmap
    ax = axes[1, 2]
    distance_metrics = ['kl_divergence', 'js_divergence', 'ks_distance', 'l1_distance', 'l2_distance']
    distance_data = np.zeros((len(baselines), len(distance_metrics)))
    
    for i, baseline in enumerate(baselines):
        for j, metric in enumerate(distance_metrics):
            metric_values = []
            for d in comparison_data:
                if baseline in d['core_distribution_comparison']['distribution_distances']:
                    metric_values.append(d['core_distribution_comparison']['distribution_distances'][baseline][metric])
            distance_data[i, j] = np.mean(metric_values) if metric_values else 0
    
    # Normalize for better visualization
    distance_data_norm = (distance_data - distance_data.min()) / (distance_data.max() - distance_data.min() + 1e-8)
    
    im = ax.imshow(distance_data_norm, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(distance_metrics)))
    ax.set_yticks(range(len(baseline_labels)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in distance_metrics], rotation=45, ha='right')
    ax.set_yticklabels(baseline_labels)
    ax.set_title("Distance Metrics Heatmap\n(Normalized)")
    
    # Add text annotations
    for i in range(len(baseline_labels)):
        for j in range(len(distance_metrics)):
            text = ax.text(j, i, f'{distance_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Baseline comparison summary saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_repeater_context_split(evaluation_results, rules, graph, context_windows, output_path=None, figsize=(15, 8)):
    """
    Plot repeater violations split by whether k is shorter or longer than context window.
    
    Args:
        evaluation_results: List of evaluation results from evaluate_model
        rules: List of rule objects
        graph: Graph object
        context_windows: List of context window sizes to analyze
        output_path: Path to save plot (optional)
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Collect data for each context window
    shorter_data = {}
    longer_data = {}
    
    for cw in context_windows:
        violation_rates = collect_repeater_violations_by_length(evaluation_results, rules, graph, context_window=cw)
        
        # Aggregate rates for shorter and longer than context
        shorter_rates = violation_rates['shorter_than_context']
        longer_rates = violation_rates['longer_than_context']
        
        # Calculate average violation rate for each category
        if shorter_rates:
            avg_shorter = np.mean(list(shorter_rates.values()))
        else:
            avg_shorter = 0.0
            
        if longer_rates:
            avg_longer = np.mean(list(longer_rates.values()))
        else:
            avg_longer = 0.0
            
        shorter_data[cw] = avg_shorter
        longer_data[cw] = avg_longer
    
    # Plot data
    x = list(shorter_data.keys())
    y_shorter = list(shorter_data.values())
    y_longer = list(longer_data.values())
    
    # Left plot: Violation rates comparison
    axes[0].plot(x, [y * 100 for y in y_shorter], 'o-', label='k ≤ context window', 
                 linewidth=2.5, markersize=8, color='#2E8B57')
    axes[0].plot(x, [y * 100 for y in y_longer], 's-', label='k > context window', 
                 linewidth=2.5, markersize=8, color='#DC143C')
    
    axes[0].set_xlabel('Context Window Size', fontsize=12)
    axes[0].set_ylabel('Average Violation Rate (%)', fontsize=12)
    axes[0].set_title('Repeater Violations by Context Window Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(max(y_shorter), max(y_longer)) * 120 if y_shorter or y_longer else 100)
    
    # Right plot: Ratio of violations
    ratio_data = []
    for cw in x:
        if longer_data[cw] > 0:
            ratio = shorter_data[cw] / longer_data[cw]
        else:
            ratio = 0
        ratio_data.append(ratio)
    
    bars = axes[1].bar(x, ratio_data, color='#4169E1', alpha=0.7, edgecolor='black')
    axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal violation rate')
    axes[1].set_xlabel('Context Window Size', fontsize=12)
    axes[1].set_ylabel('Ratio (Shorter/Longer)', fontsize=12)
    axes[1].set_title('Violation Rate Ratio: Shorter vs Longer than Context', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, ratio_data):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Repeater Rule Learning: Impact of Context Window', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes

def plot_repeater_context_analysis(data_dict, output_path=None, figsize=(15, 10)):
    """
    Plot how context window length affects repeater rule violation rates across models and repeater lengths.
    
    Args:
        data_dict: Dictionary with structure:
            {
                model_name: {
                    context_window: {
                        repeater_length_k: violation_rate (0.0-1.0)
                    }
                }
            }
        output_path: Path to save plot (optional)
        figsize: Figure size tuple
    """
    if not data_dict:
        print("No data provided for plotting")
        return
    
    # Extract unique repeater lengths and context windows
    all_k_values = set()
    all_context_windows = set()
    all_models = list(data_dict.keys())
    
    for model_data in data_dict.values():
        for context_window, k_data in model_data.items():
            all_context_windows.add(context_window)
            all_k_values.update(k_data.keys())
    
    all_k_values = sorted(all_k_values)
    all_context_windows = sorted(all_context_windows)
    
    # Calculate subplot layout
    n_k = len(all_k_values)
    cols = min(3, n_k)
    rows = (n_k + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_k == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Color map for models
    import matplotlib.cm as cm
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(all_models)))
    model_colors = dict(zip(all_models, colors))
    
    # Plot each repeater length k
    for i, k in enumerate(all_k_values):
        ax = axes[i]
        
        # Plot each model
        for model_name in all_models:
            model_data = data_dict[model_name]
            
            # Extract data for this k value
            x_vals = []
            y_vals = []
            
            for context_window in all_context_windows:
                if context_window in model_data and k in model_data[context_window]:
                    x_vals.append(context_window)
                    y_vals.append(model_data[context_window][k])
            
            if x_vals and y_vals:
                ax.plot(x_vals, y_vals, 'o-', 
                       color=model_colors[model_name], 
                       label=model_name, 
                       linewidth=2, 
                       markersize=6,
                       alpha=0.8)
        
        ax.set_xlabel("Context Window Length")
        ax.set_ylabel("Violation Rate")
        ax.set_title(f"Repeater Length k={k}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hide unused subplots
    for j in range(n_k, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Repeater Rule Violations vs Context Window Length", fontsize=16, y=0.98)
    plt.tight_layout(rect=(0, 0, 0.85, 0.96))
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes

def collect_repeater_violations_by_length(evaluation_results, rules, graph, context_window=None):
    """
    Analyze repeater rule violations broken down by repeater length k.
    
    Args:
        evaluation_results: List of evaluation results from evaluate_model
        rules: List of rule objects
        graph: Graph object
        context_window: Current context window size (optional)
        
    Returns:
        Dictionary with violation rates for all repeaters, and separated by context window comparison
    """
    # Find repeater rules and group by k value
    repeater_rules_by_k = defaultdict(list)
    repeater_rules_shorter_than_context = defaultdict(list)
    repeater_rules_longer_than_context = defaultdict(list)
    
    for rule in rules:
        if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            # Group repeater nodes by their k values
            for node, k in rule.members_nodes_dict.items():
                repeater_rules_by_k[k].append((rule, node))
                
                # Also categorize by context window if provided
                if context_window is not None:
                    if k <= context_window:
                        repeater_rules_shorter_than_context[k].append((rule, node))
                    else:
                        repeater_rules_longer_than_context[k].append((rule, node))
    
    # Count violations for each category
    violation_counts_all = {k: 0 for k in repeater_rules_by_k.keys()}
    violation_counts_shorter = {k: 0 for k in repeater_rules_shorter_than_context.keys()}
    violation_counts_longer = {k: 0 for k in repeater_rules_longer_than_context.keys()}
    total_walks = len(evaluation_results)
    
    for result in evaluation_results:
        walk = result.get("generated_walk", [])
        if not walk:
            continue
            
        # Check all repeaters
        for k, rule_node_pairs in repeater_rules_by_k.items():
            for rule, node in rule_node_pairs:
                if node in walk:
                    temp_rule_dict = {node: k}
                    from ..graph.rules import RepeaterRule
                    temp_rule = RepeaterRule(temp_rule_dict)
                    
                    if not temp_rule.is_satisfied_by(walk, graph):
                        violation_counts_all[k] += 1
                        
                        # Also count in appropriate context window category
                        if context_window is not None:
                            if k <= context_window:
                                violation_counts_shorter[k] += 1
                            else:
                                violation_counts_longer[k] += 1
                        break  # Only count one violation per walk per k
    
    # Calculate violation rates
    violation_rates = {
        'all': {},
        'shorter_than_context': {},
        'longer_than_context': {}
    }
    
    for k, count in violation_counts_all.items():
        violation_rates['all'][k] = count / total_walks if total_walks > 0 else 0.0
    
    for k, count in violation_counts_shorter.items():
        violation_rates['shorter_than_context'][k] = count / total_walks if total_walks > 0 else 0.0
        
    for k, count in violation_counts_longer.items():
        violation_rates['longer_than_context'][k] = count / total_walks if total_walks > 0 else 0.0
    
    return violation_rates

def save_multi_model_repeater_data(data_dict, output_path):
    """
    Save multi-model repeater analysis data to JSON file.
    
    Args:
        data_dict: Dictionary with model -> context_window -> k -> violation_rate structure
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f"Multi-model repeater data saved to: {output_path}")

def load_multi_model_repeater_data(input_path):
    """
    Load multi-model repeater analysis data from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Dictionary with model -> context_window -> k -> violation_rate structure
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys back to integers where appropriate
    converted_data = {}
    for model_name, model_data in data.items():
        converted_data[model_name] = {}
        for context_window_str, k_data in model_data.items():
            context_window = int(context_window_str)
            converted_data[model_name][context_window] = {}
            for k_str, violation_rate in k_data.items():
                k = int(k_str)
                converted_data[model_name][context_window][k] = violation_rate
    
    return converted_data


def plot_token_kl_heatmap(token_level_data, output_path=None, figsize=(15, 8), max_walks=10, max_steps=50):
    """
    Plot a heatmap of KL divergences across walks and token positions.
    
    Args:
        token_level_data: List of token-level dictionaries from evaluate_model
        output_path: Path to save plot (optional)
        figsize: Figure size tuple
        max_walks: Maximum number of walks to display
        max_steps: Maximum number of steps per walk to display
    """
    if not token_level_data:
        print("No token-level data provided")
        return
    
    # Organize data into a matrix
    walk_indices = sorted(set(item['walk_idx'] for item in token_level_data))[:max_walks]
    
    # Create matrix for each KL divergence type
    kl_types = ['negative_exponential', 'uniform', 'graph_neighbors']
    
    fig, axes = plt.subplots(len(kl_types), 1, figsize=figsize)
    if len(kl_types) == 1:
        axes = [axes]
    
    for kl_idx, kl_type in enumerate(kl_types):
        # Initialize matrix with NaN
        kl_matrix = np.full((len(walk_indices), max_steps), np.nan)
        
        for item in token_level_data:
            walk_idx = item['walk_idx']
            step_idx = item['step_idx']
            
            if walk_idx in walk_indices and step_idx < max_steps:
                row_idx = walk_indices.index(walk_idx)
                kl_value = item['kl_divergences'].get(kl_type, np.nan)
                kl_matrix[row_idx, step_idx] = kl_value
        
        # Plot heatmap
        im = axes[kl_idx].imshow(kl_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        axes[kl_idx].set_title(f'KL Divergence: {kl_type}')
        axes[kl_idx].set_xlabel('Token Position')
        axes[kl_idx].set_ylabel('Walk Index')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[kl_idx], label='KL Divergence')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Token KL heatmap saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


def plot_token_entropy_vs_kl(token_level_data, output_path=None, figsize=(12, 8), sample_size=1000):
    """
    Plot relationship between model entropy and KL divergence for individual tokens.
    
    Args:
        token_level_data: List of token-level dictionaries from evaluate_model
        output_path: Path to save plot (optional)
        figsize: Figure size tuple
        sample_size: Number of tokens to sample for plotting
    """
    if not token_level_data:
        print("No token-level data provided")
        return
    
    # Sample data if too large
    if len(token_level_data) > sample_size:
        import random
        token_data_sample = random.sample(token_level_data, sample_size)
    else:
        token_data_sample = token_level_data
    
    # Extract data
    entropies = [item['entropy'] for item in token_data_sample]
    kl_neg_exp = [item['kl_divergences']['negative_exponential'] for item in token_data_sample]
    kl_uniform = [item['kl_divergences']['uniform'] for item in token_data_sample]
    confidences = [item['prediction_confidence'] for item in token_data_sample]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Entropy vs KL (negative exponential)
    axes[0, 0].scatter(entropies, kl_neg_exp, alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Model Entropy')
    axes[0, 0].set_ylabel('KL Divergence (Negative Exponential)')
    axes[0, 0].set_title('Entropy vs KL Divergence')
    
    # Entropy vs KL (uniform)
    axes[0, 1].scatter(entropies, kl_uniform, alpha=0.6, s=20, color='orange')
    axes[0, 1].set_xlabel('Model Entropy')
    axes[0, 1].set_ylabel('KL Divergence (Uniform)')
    axes[0, 1].set_title('Entropy vs KL Divergence (Uniform)')
    
    # Confidence vs KL (negative exponential)
    axes[1, 0].scatter(confidences, kl_neg_exp, alpha=0.6, s=20, color='green')
    axes[1, 0].set_xlabel('Prediction Confidence')
    axes[1, 0].set_ylabel('KL Divergence (Negative Exponential)')
    axes[1, 0].set_title('Confidence vs KL Divergence')
    
    # Context length vs KL
    context_lengths = [item['context_length'] for item in token_data_sample]
    axes[1, 1].scatter(context_lengths, kl_neg_exp, alpha=0.6, s=20, color='red')
    axes[1, 1].set_xlabel('Context Length')
    axes[1, 1].set_ylabel('KL Divergence (Negative Exponential)')
    axes[1, 1].set_title('Context Length vs KL Divergence')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Entropy vs KL plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes


def plot_prediction_confidence_analysis(token_level_data, output_path=None, figsize=(15, 10)):
    """
    Analyze model prediction confidence patterns.
    
    Args:
        token_level_data: List of token-level dictionaries from evaluate_model
        output_path: Path to save plot (optional)
        figsize: Figure size tuple
    """
    if not token_level_data:
        print("No token-level data provided")
        return
    
    # Separate valid and invalid edge predictions
    valid_edge_data = [item for item in token_level_data if item['is_valid_edge']]
    invalid_edge_data = [item for item in token_level_data if not item['is_valid_edge']]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Confidence distributions
    valid_confidences = [item['prediction_confidence'] for item in valid_edge_data]
    invalid_confidences = [item['prediction_confidence'] for item in invalid_edge_data]
    
    axes[0, 0].hist(valid_confidences, bins=30, alpha=0.7, label='Valid Edges', color='green')
    axes[0, 0].hist(invalid_confidences, bins=30, alpha=0.7, label='Invalid Edges', color='red')
    axes[0, 0].set_xlabel('Prediction Confidence')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Confidence Distribution by Edge Validity')
    axes[0, 0].legend()
    
    # KL divergence distributions
    valid_kl = [item['kl_divergences']['negative_exponential'] for item in valid_edge_data]
    invalid_kl = [item['kl_divergences']['negative_exponential'] for item in invalid_edge_data]
    
    axes[0, 1].hist(valid_kl, bins=30, alpha=0.7, label='Valid Edges', color='green')
    axes[0, 1].hist(invalid_kl, bins=30, alpha=0.7, label='Invalid Edges', color='red')
    axes[0, 1].set_xlabel('KL Divergence')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('KL Divergence Distribution by Edge Validity')
    axes[0, 1].legend()
    
    # Entropy distributions
    valid_entropy = [item['entropy'] for item in valid_edge_data]
    invalid_entropy = [item['entropy'] for item in invalid_edge_data]
    
    axes[0, 2].hist(valid_entropy, bins=30, alpha=0.7, label='Valid Edges', color='green')
    axes[0, 2].hist(invalid_entropy, bins=30, alpha=0.7, label='Invalid Edges', color='red')
    axes[0, 2].set_xlabel('Model Entropy')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Entropy Distribution by Edge Validity')
    axes[0, 2].legend()
    
    # Position-based analysis
    step_positions = [item['step_idx'] for item in token_level_data]
    confidences = [item['prediction_confidence'] for item in token_level_data]
    kl_values = [item['kl_divergences']['negative_exponential'] for item in token_level_data]
    
    # Binned analysis
    max_pos = max(step_positions)
    bins = min(20, max_pos)
    bin_edges = np.linspace(0, max_pos, bins + 1)
    
    bin_confidences = []
    bin_kl = []
    bin_centers = []
    
    for i in range(bins):
        mask = (np.array(step_positions) >= bin_edges[i]) & (np.array(step_positions) < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_confidences.append(np.mean([confidences[j] for j in range(len(confidences)) if mask[j]]))
            bin_kl.append(np.mean([kl_values[j] for j in range(len(kl_values)) if mask[j]]))
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
    
    axes[1, 0].plot(bin_centers, bin_confidences, 'o-', color='blue')
    axes[1, 0].set_xlabel('Token Position')
    axes[1, 0].set_ylabel('Mean Prediction Confidence')
    axes[1, 0].set_title('Confidence vs Position')
    
    axes[1, 1].plot(bin_centers, bin_kl, 'o-', color='purple')
    axes[1, 1].set_xlabel('Token Position')
    axes[1, 1].set_ylabel('Mean KL Divergence')
    axes[1, 1].set_title('KL Divergence vs Position')
    
    # Top prediction accuracy
    correct_predictions = sum(1 for item in token_level_data if item['is_valid_edge'])
    total_predictions = len(token_level_data)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Create bar chart for summary statistics instead of text
    categories = ['Accuracy', 'Valid/Total']
    values = [accuracy, correct_predictions/total_predictions if total_predictions > 0 else 0]
    bars = axes[1, 2].bar(categories, values, color=['blue', 'green'], alpha=0.7)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_ylabel('Ratio')
    axes[1, 2].set_title('Prediction Summary')
    
    # Add legend with actual counts
    axes[1, 2].legend([f'Accuracy: {accuracy:.3f}', 
                       f'Valid: {correct_predictions}/{total_predictions}'],
                      loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Prediction confidence analysis saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes

def plot_density_vs_accuracy(density_study_results, output_path=None, figsize=(15, 10)):
    """
    Plot the relationship between repeater density and accuracy.
    
    Args:
        density_study_results: Results from run_repeater_density_study
        output_path: Path to save plot (optional)
        figsize: Figure size tuple
    """
    if not density_study_results:
        print("No density study results provided")
        return
    
    # Extract data for plotting
    plot_data = {}
    
    for config_name, results in density_study_results.items():
        if "error" in results:
            continue
            
        density_stats = results["density_stats"]
        error_summary = results["error_summary"]
        violation_rates = results["violation_rates_by_k"]
        
        # Overall accuracy
        overall_accuracy = 1.0 - error_summary["repeater_error_rate"]
        
        # Extract node-specific data
        for node, exposure_count in density_stats["repeater_exposure_counts"].items():
            if node not in plot_data:
                plot_data[node] = {
                    "exposures": [],
                    "accuracies": [],
                    "config_names": [],
                    "violation_rates_by_k": defaultdict(list)
                }
            
            plot_data[node]["exposures"].append(exposure_count)
            plot_data[node]["accuracies"].append(overall_accuracy)
            plot_data[node]["config_names"].append(config_name)
            
            # Store k-specific violation rates
            for k, rate in violation_rates.items():
                accuracy_k = 1.0 - rate
                plot_data[node]["violation_rates_by_k"][k].append(accuracy_k)
    
    if not plot_data:
        print("No valid data to plot")
        return
    
    # Determine subplot layout
    nodes = list(plot_data.keys())
    n_nodes = len(nodes)
    cols = min(3, n_nodes)
    rows = (n_nodes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_nodes == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Color map for different k values
    import matplotlib.cm as cm
    all_k_values = set()
    for node_data in plot_data.values():
        all_k_values.update(node_data["violation_rates_by_k"].keys())
    all_k_values = sorted(all_k_values)
    
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(all_k_values)))
    k_colors = dict(zip(all_k_values, colors))
    
    # Plot each node
    for i, node in enumerate(nodes):
        ax = axes[i]
        node_data = plot_data[node]
        
        # Plot overall accuracy
        ax.scatter(node_data["exposures"], node_data["accuracies"], 
                  color='black', s=60, alpha=0.7, label="Overall", zorder=3)
        
        # Add trend line for overall accuracy
        if len(node_data["exposures"]) > 1:
            z = np.polyfit(node_data["exposures"], node_data["accuracies"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(node_data["exposures"]), max(node_data["exposures"]), 100)
            ax.plot(x_trend, p(x_trend), 'k--', alpha=0.5)
        
        # Plot k-specific accuracies
        for k in all_k_values:
            if k in node_data["violation_rates_by_k"]:
                k_accuracies = node_data["violation_rates_by_k"][k]
                ax.scatter(node_data["exposures"], k_accuracies,
                          color=k_colors[k], s=40, alpha=0.6, 
                          label=f"k={k}", marker='s')
        
        ax.set_xlabel("Training Examples (Exposure Count)")
        ax.set_ylabel("Accuracy (1 - Violation Rate)")
        ax.set_title(f"Node {node}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hide unused subplots
    for j in range(n_nodes, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Repeater Training Density vs Model Accuracy", fontsize=16, y=0.98)
    plt.tight_layout(rect=(0, 0, 0.85, 0.96))
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Density vs accuracy plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, axes

def plot_density_correlation_summary(density_analysis, output_path=None, figsize=(12, 8)):
    """
    Plot summary of density-accuracy correlations across all repeater nodes.
    
    Args:
        density_analysis: Results from analyze_density_vs_accuracy
        output_path: Path to save plot (optional)
        figsize: Figure size tuple
    """
    correlations = density_analysis.get("density_accuracy_correlation", {})
    
    if not correlations:
        print("No correlation data available")
        return
    
    nodes = list(correlations.keys())
    corr_values = [correlations[node] for node in nodes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot of correlations
    bars = ax1.bar(range(len(nodes)), corr_values, alpha=0.7)
    ax1.set_xlabel("Repeater Node")
    ax1.set_ylabel("Density-Accuracy Correlation")
    ax1.set_title("Training Density vs Accuracy Correlation by Node")
    ax1.set_xticks(range(len(nodes)))
    ax1.set_xticklabels([f"Node {n}" for n in nodes], rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Color bars based on correlation strength
    for bar, corr in zip(bars, corr_values):
        if corr > 0.5:
            bar.set_color('green')
        elif corr > 0.2:
            bar.set_color('orange')  
        elif corr < -0.2:
            bar.set_color('red')
        else:
            bar.set_color('gray')
    
    # Histogram of correlation values
    ax2.hist(corr_values, bins=10, alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Correlation Coefficient")
    ax2.set_ylabel("Number of Nodes")
    ax2.set_title("Distribution of Density-Accuracy Correlations")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation summary plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, (ax1, ax2)


def plot_rule_specific_error_rates_by_context(experiment_results, context_windows, output_path=None, figsize=(15, 8)):
    """
    Plot error rates per rule per context window.
    
    Args:
        experiment_results: Dictionary mapping context_window -> experiment results
        context_windows: List of context window sizes
        output_path: Path to save the plot
        figsize: Figure size tuple
    """
    if not experiment_results:
        print("No experiment results provided")
        return
    
    # Rule types we track
    rule_types = ['repeater', 'ascender', 'even', 'broken_graph']
    rule_labels = ['Repeater Rules', 'Ascender Rules', 'Even Rules', 'Broken Graph']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    # Extract error rates for each rule type and context window
    error_data = {rule_type: [] for rule_type in rule_types}
    valid_contexts = []
    
    for context_window in context_windows:
        if context_window not in experiment_results:
            continue
            
        # Load final results for this context
        exp_result = experiment_results[context_window]
        final_results_path = os.path.join(exp_result, "evaluation", "final_results.json")
        
        if os.path.exists(final_results_path):
            with open(final_results_path, "r") as f:
                final_results = json.load(f)
            
            error_summary = final_results.get("aggregated_error_summary", {})
            valid_contexts.append(context_window)
            
            for rule_type in rule_types:
                error_key = f"{rule_type}_error_rate"
                error_rate = error_summary.get(error_key, 0.0)
                error_data[rule_type].append(error_rate * 100)  # Convert to percentage
    
    if not valid_contexts:
        print("No valid experiment results found")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Rule-Specific Error Rates by Context Window", fontsize=16, fontweight='bold')
    
    # Plot 1: Line plot showing error rates across context windows
    for i, rule_type in enumerate(rule_types):
        ax1.plot(valid_contexts, error_data[rule_type], 'o-', 
                 label=rule_labels[i], color=colors[i], 
                 linewidth=2.5, markersize=8, alpha=0.8)
    
    ax1.set_xlabel("Context Window Size", fontsize=12)
    ax1.set_ylabel("Error Rate (%)", fontsize=12)
    ax1.set_title("Error Rates Across Context Windows", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Plot 2: Heatmap of error rates
    error_matrix = np.array([error_data[rule_type] for rule_type in rule_types])
    
    im = ax2.imshow(error_matrix, cmap='Reds', aspect='auto')
    ax2.set_xticks(range(len(valid_contexts)))
    ax2.set_yticks(range(len(rule_types)))
    ax2.set_xticklabels(valid_contexts)
    ax2.set_yticklabels(rule_labels)
    ax2.set_xlabel("Context Window Size", fontsize=12)
    ax2.set_title("Error Rate Heatmap (%)", fontsize=14)
    
    # Add text annotations to heatmap
    for i in range(len(rule_types)):
        for j in range(len(valid_contexts)):
            text = ax2.text(j, i, f'{error_matrix[i, j]:.1f}%',
                           ha="center", va="center", 
                           color="white" if error_matrix[i, j] > 50 else "black",
                           fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Error Rate (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Rule-specific error rate plots saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    return fig, (ax1, ax2)


def plot_distributional_metric_exemplars(token_level_data, output_path=None, figsize=(18, 12), n_exemplars=6):
    """
    Plot exemplars of distributional metrics showing specific examples.
    
    Args:
        token_level_data: List of token-level dictionaries with distribution comparisons
        output_path: Path to save the plot
        figsize: Figure size tuple
        n_exemplars: Number of exemplar walks to show
    """
    # Filter data with distribution comparisons
    comparison_data = [t for t in token_level_data if 'core_distribution_comparison' in t]
    
    if not comparison_data:
        print("No distribution comparison data found")
        return
    
    # Group data by walk_idx
    walk_data = defaultdict(list)
    for token in comparison_data:
        walk_data[token['walk_idx']].append(token)
    
    # Select exemplar walks with diverse characteristics
    walk_indices = list(walk_data.keys())
    if len(walk_indices) < n_exemplars:
        n_exemplars = len(walk_indices)
    
    # Select exemplars based on different KL divergence ranges
    exemplar_walks = []
    kl_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, float('inf'))]
    
    for kl_min, kl_max in kl_ranges[:n_exemplars]:
        best_walk = None
        best_match_score = -1
        
        for walk_idx in walk_indices:
            walk_tokens = walk_data[walk_idx]
            # Calculate average KL from graph structure
            avg_kl = np.mean([t['core_distribution_comparison']['distribution_distances']['graph_structure']['kl_divergence'] 
                              for t in walk_tokens if 'graph_structure' in t['core_distribution_comparison']['distribution_distances']])
            
            if kl_min <= avg_kl < kl_max:
                # Score based on walk length and data completeness
                match_score = len(walk_tokens) + (1 if avg_kl > (kl_min + kl_max) / 2 else 0)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_walk = walk_idx
        
        if best_walk is not None:
            exemplar_walks.append(best_walk)
            walk_indices.remove(best_walk)
    
    # Fill remaining exemplars if needed
    while len(exemplar_walks) < n_exemplars and walk_indices:
        exemplar_walks.append(walk_indices.pop(0))
    
    if not exemplar_walks:
        print("No suitable exemplar walks found")
        return
    
    # Create subplots
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle("Exemplar Distributional Metrics Across Walks", fontsize=16, fontweight='bold')
    
    for idx, walk_idx in enumerate(exemplar_walks[:6]):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        walk_tokens = walk_data[walk_idx]
        steps = [t['step_idx'] for t in walk_tokens]
        
        # Plot multiple metrics
        kl_graph = [t['core_distribution_comparison']['distribution_distances']['graph_structure']['kl_divergence'] 
                    for t in walk_tokens if 'graph_structure' in t['core_distribution_comparison']['distribution_distances']]
        structural_awareness = [t['core_distribution_comparison']['prediction_quality_scores'].get('structural_awareness', 0) 
                               for t in walk_tokens if 'prediction_quality_scores' in t['core_distribution_comparison']]
        top1_agreement = [t['core_distribution_comparison']['distribution_overlap_analysis']['graph_structure']['agreement_on_top_k'].get('top_1', 0) 
                          for t in walk_tokens if 'graph_structure' in t['core_distribution_comparison']['distribution_overlap_analysis']]
        
        # Plot lines
        ax.plot(steps[:len(kl_graph)], kl_graph, 'r-', label='KL from Graph', linewidth=2, alpha=0.8)
        ax.plot(steps[:len(structural_awareness)], structural_awareness, 'b-', label='Structural Awareness', linewidth=2, alpha=0.8)
        ax.plot(steps[:len(top1_agreement)], top1_agreement, 'g-', label='Top-1 Agreement', linewidth=2, alpha=0.8)
        
        # Formatting
        ax.set_xlabel("Step")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"Walk {walk_idx} (avg KL: {np.mean(kl_graph):.3f})")
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Distributional metric exemplars saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    return fig, axes


def plot_walk_level_distributional_aggregates(token_level_data, output_path=None, figsize=(16, 10)):
    """
    Plot aggregated distributional metrics at the walk level.
    
    Args:
        token_level_data: List of token-level dictionaries with distribution comparisons
        output_path: Path to save the plot
        figsize: Figure size tuple
    """
    # Filter and group data by walk
    comparison_data = [t for t in token_level_data if 'core_distribution_comparison' in t]
    
    if not comparison_data:
        print("No distribution comparison data found")
        return
    
    # Group by walk and calculate walk-level aggregates
    walk_data = defaultdict(list)
    for token in comparison_data:
        walk_data[token['walk_idx']].append(token)
    
    walk_aggregates = []
    walk_indices = []
    
    for walk_idx, tokens in walk_data.items():
        if len(tokens) < 3:  # Skip walks with too few tokens
            continue
            
        # Calculate walk-level metrics
        kl_graph_values = [t['core_distribution_comparison']['distribution_distances']['graph_structure']['kl_divergence'] 
                           for t in tokens if 'graph_structure' in t['core_distribution_comparison']['distribution_distances']]
        kl_uniform_values = [t['core_distribution_comparison']['distribution_distances']['uniform_valid']['kl_divergence'] 
                             for t in tokens if 'uniform_valid' in t['core_distribution_comparison']['distribution_distances']]
        structural_awareness_values = [t['core_distribution_comparison']['prediction_quality_scores'].get('structural_awareness', 0) 
                                       for t in tokens if 'prediction_quality_scores' in t['core_distribution_comparison']]
        quality_values = [t['core_distribution_comparison']['prediction_quality_scores'].get('overall_quality', 0) 
                          for t in tokens if 'prediction_quality_scores' in t['core_distribution_comparison']]
        top1_agreement_values = [t['core_distribution_comparison']['distribution_overlap_analysis']['graph_structure']['agreement_on_top_k'].get('top_1', 0) 
                                 for t in tokens if 'graph_structure' in t['core_distribution_comparison']['distribution_overlap_analysis']]
        
        walk_aggregate = {
            'walk_idx': walk_idx,
            'walk_length': len(tokens),
            'kl_graph_mean': np.mean(kl_graph_values) if kl_graph_values else 0,
            'kl_graph_std': np.std(kl_graph_values) if kl_graph_values else 0,
            'kl_uniform_mean': np.mean(kl_uniform_values) if kl_uniform_values else 0,
            'structural_awareness_mean': np.mean(structural_awareness_values) if structural_awareness_values else 0,
            'quality_mean': np.mean(quality_values) if quality_values else 0,
            'top1_agreement_mean': np.mean(top1_agreement_values) if top1_agreement_values else 0,
            'kl_graph_final': kl_graph_values[-1] if kl_graph_values else 0,
            'quality_trend': np.polyfit(range(len(quality_values)), quality_values, 1)[0] if len(quality_values) > 2 else 0
        }
        
        walk_aggregates.append(walk_aggregate)
        walk_indices.append(walk_idx)
    
    if not walk_aggregates:
        print("No valid walk aggregates found")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Walk-Level Distributional Metric Aggregates", fontsize=16, fontweight='bold')
    
    # Extract arrays for plotting
    kl_graph_means = [w['kl_graph_mean'] for w in walk_aggregates]
    kl_graph_stds = [w['kl_graph_std'] for w in walk_aggregates]
    structural_awareness_means = [w['structural_awareness_mean'] for w in walk_aggregates]
    quality_means = [w['quality_mean'] for w in walk_aggregates]
    top1_agreement_means = [w['top1_agreement_mean'] for w in walk_aggregates]
    walk_lengths = [w['walk_length'] for w in walk_aggregates]
    quality_trends = [w['quality_trend'] for w in walk_aggregates]
    
    # Plot 1: Distribution of mean KL divergences from graph structure
    axes[0, 0].hist(kl_graph_means, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Mean KL Divergence from Graph')
    axes[0, 0].set_ylabel('Number of Walks')
    axes[0, 0].set_title('Distribution of Walk-Level\nKL Divergence Means')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(kl_graph_means), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(kl_graph_means):.3f}')
    axes[0, 0].legend()
    
    # Plot 2: Structural awareness vs quality
    scatter = axes[0, 1].scatter(structural_awareness_means, quality_means, 
                                 c=walk_lengths, cmap='viridis', alpha=0.6, s=40)
    axes[0, 1].set_xlabel('Mean Structural Awareness')
    axes[0, 1].set_ylabel('Mean Overall Quality')
    axes[0, 1].set_title('Structural Awareness vs Quality\n(colored by walk length)')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 1], label='Walk Length')
    
    # Plot 3: Walk length vs KL divergence
    axes[0, 2].scatter(walk_lengths, kl_graph_means, alpha=0.6, color='orange', s=40)
    axes[0, 2].set_xlabel('Walk Length')
    axes[0, 2].set_ylabel('Mean KL Divergence from Graph')
    axes[0, 2].set_title('Walk Length vs\nKL Divergence')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add trend line
    if len(walk_lengths) > 1:
        z = np.polyfit(walk_lengths, kl_graph_means, 1)
        p = np.poly1d(z)
        axes[0, 2].plot(sorted(walk_lengths), p(sorted(walk_lengths)), "r--", alpha=0.8)
    
    # Plot 4: Top-1 agreement distribution
    axes[1, 0].hist(top1_agreement_means, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Mean Top-1 Agreement with Graph')
    axes[1, 0].set_ylabel('Number of Walks')
    axes[1, 0].set_title('Distribution of Walk-Level\nTop-1 Agreement')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(np.mean(top1_agreement_means), color='green', linestyle='--', 
                       label=f'Mean: {np.mean(top1_agreement_means):.3f}')
    axes[1, 0].legend()
    
    # Plot 5: KL divergence variability
    axes[1, 1].scatter(kl_graph_means, kl_graph_stds, alpha=0.6, color='purple', s=40)
    axes[1, 1].set_xlabel('Mean KL Divergence')
    axes[1, 1].set_ylabel('KL Divergence Std Dev')
    axes[1, 1].set_title('KL Divergence:\nMean vs Variability')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Quality trend analysis
    axes[1, 2].hist(quality_trends, bins=20, alpha=0.7, color='teal', edgecolor='black')
    axes[1, 2].set_xlabel('Quality Trend Slope')
    axes[1, 2].set_ylabel('Number of Walks')
    axes[1, 2].set_title('Distribution of Quality\nTrend Slopes')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axvline(0, color='black', linestyle='--', alpha=0.5, label='No trend')
    axes[1, 2].axvline(np.mean(quality_trends), color='teal', linestyle='--', 
                       label=f'Mean: {np.mean(quality_trends):.4f}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Walk-level distributional aggregates saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    return fig, axes, walk_aggregates