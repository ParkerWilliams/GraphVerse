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

def collect_repeater_violations_by_length(evaluation_results, rules, graph):
    """
    Analyze repeater rule violations broken down by repeater length k.
    
    Args:
        evaluation_results: List of evaluation results from evaluate_model
        rules: List of rule objects
        graph: Graph object
        
    Returns:
        Dictionary mapping repeater_length_k -> violation_rate
    """
    # Find repeater rules and group by k value
    repeater_rules_by_k = defaultdict(list)
    
    for rule in rules:
        if hasattr(rule, 'is_repeater_rule') and rule.is_repeater_rule:
            # Group repeater nodes by their k values
            for node, k in rule.members_nodes_dict.items():
                repeater_rules_by_k[k].append((rule, node))
    
    # Count violations for each k value
    violation_counts = {k: 0 for k in repeater_rules_by_k.keys()}
    total_walks = len(evaluation_results)
    
    for result in evaluation_results:
        walk = result.get("generated_walk", [])
        if not walk:
            continue
            
        for k, rule_node_pairs in repeater_rules_by_k.items():
            for rule, node in rule_node_pairs:
                # Check if this specific node violates the rule
                if node in walk:
                    # Create a temporary rule with just this node
                    temp_rule_dict = {node: k}
                    from ..graph.rules import RepeaterRule
                    temp_rule = RepeaterRule(temp_rule_dict)
                    
                    if not temp_rule.is_satisfied_by(walk, graph):
                        violation_counts[k] += 1
                        break  # Only count one violation per walk per k
    
    # Calculate violation rates
    violation_rates = {}
    for k, count in violation_counts.items():
        violation_rates[k] = count / total_walks if total_walks > 0 else 0.0
    
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