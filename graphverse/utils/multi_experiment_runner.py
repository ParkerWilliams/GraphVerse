import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from .experiment_manager import save_config
from ..llm.evaluation import evaluate_model, get_large_scale_trajectory_config
from ..llm.large_scale_evaluation import LargeScaleEvaluator


class MultiExperimentRunner:
    """
    Runs multiple experiments across different context windows to analyze
    how rule violations (especially repeater rules) spike when rules exceed
    the context length.
    """
    
    def __init__(self, base_experiment_folder="context_window_experiments"):
        self.base_folder = base_experiment_folder
        self.experiment_results = []
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_multi_experiment_folder(self):
        """Create folder structure for multi-experiment run."""
        run_name = f"context_analysis_{self.run_timestamp}"
        path = os.path.join(self.base_folder, run_name)
        os.makedirs(os.path.join(path, "individual_experiments"), exist_ok=True)
        os.makedirs(os.path.join(path, "aggregated_results"), exist_ok=True)
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
        return path
    
    def run_context_window_experiments(
        self,
        model,
        graph,
        vocab,
        rules,
        context_windows: List[int],
        base_config: Dict[str, Any],
        num_walks_per_experiment: int = 100,
        verbose: bool = True
    ) -> str:
        """
        Run experiments across different context windows.
        
        Args:
            model: The language model to evaluate
            graph: Graph object
            vocab: Vocabulary object
            rules: List of rule objects
            context_windows: List of context window sizes to test
            base_config: Base configuration for experiments
            num_walks_per_experiment: Number of walks to generate per experiment
            verbose: Whether to print progress
            
        Returns:
            Path to the multi-experiment results folder
        """
        
        # Create main experiment folder
        main_folder = self.create_multi_experiment_folder()
        
        # Save overall configuration
        overall_config = {
            "experiment_type": "context_window_analysis",
            "context_windows": context_windows,
            "num_walks_per_experiment": num_walks_per_experiment,
            "base_config": base_config,
            "run_timestamp": self.run_timestamp,
            "total_experiments": len(context_windows)
        }
        save_config(overall_config, main_folder)
        
        if verbose:
            print(f"Starting context window analysis with {len(context_windows)} experiments")
            print(f"Context windows: {context_windows}")
            print(f"Results will be saved to: {main_folder}")
        
        # Run experiments for each context window
        for i, context_window in enumerate(context_windows):
            if verbose:
                print(f"\n--- Experiment {i+1}/{len(context_windows)}: Context Window = {context_window} ---")
            
            # Create individual experiment folder
            exp_folder = os.path.join(main_folder, "individual_experiments", f"ctx_{context_window}")
            os.makedirs(exp_folder, exist_ok=True)
            
            # Update config for this experiment
            exp_config = base_config.copy()
            exp_config["context_window"] = context_window
            exp_config["experiment_id"] = i
            save_config(exp_config, exp_folder)
            
            # Set model's context window if it has this parameter
            if hasattr(model, 'context_window'):
                model.context_window = context_window
            elif hasattr(model, 'max_length'):
                model.max_length = context_window
            
            # Run evaluation
            try:
                evaluation_results, error_summary, kl_series = evaluate_model(
                    model=model,
                    graph=graph,
                    vocab=vocab,
                    num_walks=num_walks_per_experiment,
                    min_start_length=base_config.get("min_start_length", 3),
                    max_start_length=base_config.get("max_start_length", 8),
                    rules=rules,
                    verbose=verbose
                )
                
                # Ensure results are not None
                if evaluation_results is None:
                    evaluation_results = []
                if error_summary is None:
                    error_summary = {"repeater_error_rate": 0.0, "ascender_error_rate": 0.0, 
                                   "even_error_rate": 0.0, "broken_graph_error_rate": 0.0, "total_steps": 0}
                if kl_series is None:
                    kl_series = []
                    
            except Exception as e:
                if verbose:
                    print(f"Error in evaluation for context window {context_window}: {e}")
                # Use default values
                evaluation_results = []
                error_summary = {"repeater_error_rate": 0.0, "ascender_error_rate": 0.0, 
                               "even_error_rate": 0.0, "broken_graph_error_rate": 0.0, "total_steps": 0}
                kl_series = []
            
            # Save individual experiment results
            self._save_individual_experiment_results(
                exp_folder, evaluation_results, error_summary, kl_series, context_window
            )
            
            # Extract rule-specific violations for analysis
            rule_violations = self._analyze_rule_violations(evaluation_results, rules, graph)
            
            # Store results for aggregation
            experiment_result = {
                "context_window": context_window,
                "error_summary": error_summary,
                "rule_violations": rule_violations,
                "kl_series_stats": self._compute_kl_stats(kl_series),
                "experiment_folder": exp_folder
            }
            self.experiment_results.append(experiment_result)
            
            if verbose:
                print(f"Repeater error rate: {error_summary['repeater_error_rate']:.4f}")
                print(f"Ascender error rate: {error_summary['ascender_error_rate']:.4f}")
                print(f"Even error rate: {error_summary['even_error_rate']:.4f}")
        
        # Save aggregated results
        self._save_aggregated_results(main_folder)
        
        if verbose:
            print(f"\n=== Context Window Analysis Complete ===")
            print(f"Results saved to: {main_folder}")
            print("Use plot_context_window_analysis() to visualize results")
        
        return main_folder
    
    def run_large_scale_context_window_experiments(
        self,
        model,
        graph,
        vocab,
        rules,
        context_windows: List[int],
        num_walks_per_experiment: int = 1000000,
        batch_size: int = 50000,
        trajectory_sample_rate: float = 0.02,
        use_large_scale_config: bool = True,
        verbose: bool = True
    ) -> str:
        """
        Run large-scale context window experiments using batch processing.
        
        Args:
            model: The language model to evaluate
            graph: Graph object
            vocab: Vocabulary object
            rules: List of rule objects
            context_windows: List of context window sizes to test
            num_walks_per_experiment: Number of walks to generate per experiment
            batch_size: Number of walks per batch for processing
            trajectory_sample_rate: Sampling rate for trajectory storage
            use_large_scale_config: Whether to use optimized large-scale configuration
            verbose: Whether to print progress
            
        Returns:
            Path to the multi-experiment results folder
        """
        # Create main experiment folder
        main_folder = self.create_multi_experiment_folder()
        
        # Save overall configuration
        overall_config = {
            "experiment_type": "large_scale_context_window_analysis",
            "context_windows": context_windows,
            "num_walks_per_experiment": num_walks_per_experiment,
            "batch_size": batch_size,
            "trajectory_sample_rate": trajectory_sample_rate,
            "use_large_scale_config": use_large_scale_config,
            "run_timestamp": self.run_timestamp,
            "total_experiments": len(context_windows),
            "total_walks": len(context_windows) * num_walks_per_experiment
        }
        save_config(overall_config, main_folder)
        
        if verbose:
            print(f"Starting large-scale context window analysis with {len(context_windows)} experiments")
            print(f"Context windows: {context_windows}")
            print(f"Walks per experiment: {num_walks_per_experiment:,}")
            print(f"Total walks: {len(context_windows) * num_walks_per_experiment:,}")
            print(f"Batch size: {batch_size:,}")
            print(f"Trajectory sampling rate: {trajectory_sample_rate:.1%}")
            print(f"Results will be saved to: {main_folder}")
        
        # Run experiments for each context window
        for i, context_window in enumerate(context_windows):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Experiment {i+1}/{len(context_windows)}: Context Window = {context_window}")
                print(f"{'='*80}")
            
            # Create individual experiment folder
            exp_folder = os.path.join(main_folder, "individual_experiments", f"ctx_{context_window}")
            
            # Set up trajectory sampling configuration
            if use_large_scale_config:
                trajectory_config = get_large_scale_trajectory_config(
                    num_walks_per_experiment, 
                    sample_rate=trajectory_sample_rate, 
                    stratified=True
                )
            else:
                trajectory_config = None
            
            # Update config for this experiment
            exp_config = {
                "context_window": context_window,
                "num_walks": num_walks_per_experiment,
                "batch_size": batch_size,
                "trajectory_config": trajectory_config,
                "experiment_id": i,
                "start_time": datetime.now().isoformat()
            }
            save_config(exp_config, exp_folder)
            
            # Set model's context window if it has this parameter
            if hasattr(model, 'context_window'):
                model.context_window = context_window
            elif hasattr(model, 'max_length'):
                model.max_length = context_window
            
            # Use LargeScaleEvaluator for high walk counts
            if num_walks_per_experiment >= 10000:
                evaluator = LargeScaleEvaluator(exp_folder)
                
                try:
                    evaluator.evaluate_large_scale(
                        model=model,
                        graph=graph,
                        vocab=vocab,
                        num_walks=num_walks_per_experiment,
                        min_start_length=3,
                        max_start_length=8,
                        rules=rules,
                        batch_size=batch_size,
                        trajectory_sampling_config=trajectory_config,
                        resume_from_checkpoint=True,
                        verbose=verbose
                    )
                    
                    # Load final results for aggregation
                    results_file = os.path.join(exp_folder, "evaluation", "final_results.json")
                    if os.path.exists(results_file):
                        with open(results_file, "r") as f:
                            final_results = json.load(f)
                        error_summary = final_results["aggregated_error_summary"]
                    else:
                        error_summary = {"repeater_error_rate": 0.0, "ascender_error_rate": 0.0, 
                                       "even_error_rate": 0.0, "broken_graph_error_rate": 0.0, "total_steps": 0}
                        
                except Exception as e:
                    if verbose:
                        print(f"Error in large-scale evaluation for context window {context_window}: {e}")
                    error_summary = {"repeater_error_rate": 0.0, "ascender_error_rate": 0.0, 
                                   "even_error_rate": 0.0, "broken_graph_error_rate": 0.0, "total_steps": 0}
            
            else:
                # Use traditional evaluation for smaller experiments
                try:
                    evaluation_results, error_summary, kl_series, token_data, progressive_analysis, exemplars, trajectories = evaluate_model(
                        model=model,
                        graph=graph,
                        vocab=vocab,
                        num_walks=num_walks_per_experiment,
                        min_start_length=3,
                        max_start_length=8,
                        rules=rules,
                        verbose=verbose,
                        trajectory_sampling_config=trajectory_config
                    )
                    
                    # Save traditional results
                    self._save_individual_experiment_results(
                        exp_folder, evaluation_results, error_summary, kl_series, context_window
                    )
                    
                except Exception as e:
                    if verbose:
                        print(f"Error in evaluation for context window {context_window}: {e}")
                    error_summary = {"repeater_error_rate": 0.0, "ascender_error_rate": 0.0, 
                                   "even_error_rate": 0.0, "broken_graph_error_rate": 0.0, "total_steps": 0}
            
            # Store results for aggregation (simplified for large-scale)
            experiment_result = {
                "context_window": context_window,
                "error_summary": error_summary,
                "num_walks": num_walks_per_experiment,
                "experiment_folder": exp_folder,
                "is_large_scale": num_walks_per_experiment >= 10000
            }
            self.experiment_results.append(experiment_result)
            
            if verbose:
                print(f"Context {context_window} completed:")
                print(f"  Repeater error rate: {error_summary['repeater_error_rate']:.4f}")
                print(f"  Ascender error rate: {error_summary['ascender_error_rate']:.4f}")
                print(f"  Even error rate: {error_summary['even_error_rate']:.4f}")
        
        # Save aggregated results
        self._save_large_scale_aggregated_results(main_folder)
        
        if verbose:
            print(f"\n{'='*80}")
            print("LARGE-SCALE CONTEXT WINDOW ANALYSIS COMPLETE")
            print(f"{'='*80}")
            print(f"Results saved to: {main_folder}")
            print("Use plot_context_window_analysis() to visualize results")
        
        return main_folder
    
    def _save_individual_experiment_results(
        self, 
        folder: str, 
        evaluation_results: List[Dict], 
        error_summary: Dict, 
        kl_series: List[List], 
        context_window: int
    ):
        """Save results for individual experiment."""
        
        # Save error summary
        with open(os.path.join(folder, "error_summary.json"), "w") as f:
            json.dump(error_summary, f, indent=2)
        
        # Save KL divergence series
        kl_path = os.path.join(folder, "kl_divergence_timeseries.csv")
        with open(kl_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["walk_idx", "step_idx", "kl_divergence", "context_window"])
            for walk_idx, walk_kl in enumerate(kl_series):
                if walk_kl is not None:
                    for step_idx, kl in enumerate(walk_kl):
                        if kl is not None:
                            writer.writerow([walk_idx, step_idx, kl, context_window])
        
        # Save evaluation results summary
        walk_lengths = [len(r["generated_walk"]) for r in evaluation_results if r.get("generated_walk")]
        avg_walk_length = np.mean(walk_lengths) if walk_lengths else 0.0
        
        results_summary = {
            "context_window": context_window,
            "total_walks": len(evaluation_results),
            "avg_walk_length": float(avg_walk_length),
            "walks_summary": [
                {
                    "start_walk": r.get("start_walk", []),
                    "generated_walk_length": len(r.get("generated_walk", [])),
                    "walk_id": i
                }
                for i, r in enumerate(evaluation_results)
                if r.get("generated_walk") is not None
            ]
        }
        
        with open(os.path.join(folder, "results_summary.json"), "w") as f:
            json.dump(results_summary, f, indent=2)
    
    def _analyze_rule_violations(self, evaluation_results: List[Dict], rules: List, graph) -> Dict:
        """Analyze rule violations in detail."""
        
        violations_by_rule = {
            "repeater": [],
            "ascender": [],
            "even": []
        }
        
        for result in evaluation_results:
            walk = result["generated_walk"]
            
            for rule in rules:
                rule_type = None
                if hasattr(rule, "is_repeater_rule") and rule.is_repeater_rule:
                    rule_type = "repeater"
                elif hasattr(rule, "is_ascender_rule") and rule.is_ascender_rule:
                    rule_type = "ascender"
                elif hasattr(rule, "is_even_rule") and rule.is_even_rule:
                    rule_type = "even"
                
                if rule_type:
                    is_satisfied = rule.is_satisfied_by(walk, graph)
                    violations_by_rule[rule_type].append({
                        "walk_length": len(walk),
                        "violated": not is_satisfied,
                        "walk": walk
                    })
        
        return violations_by_rule
    
    def _compute_kl_stats(self, kl_series: List[List]) -> Dict:
        """Compute statistics for KL divergence series."""
        if not kl_series:
            return {
                "mean_kl": 0.0,
                "std_kl": 0.0,
                "max_kl": 0.0,
                "min_kl": 0.0,
                "num_steps_total": 0
            }
        
        # Filter out None values and empty lists
        valid_kl_series = [walk_kl for walk_kl in kl_series if walk_kl is not None and len(walk_kl) > 0]
        
        if not valid_kl_series:
            return {
                "mean_kl": 0.0,
                "std_kl": 0.0,
                "max_kl": 0.0,
                "min_kl": 0.0,
                "num_steps_total": 0
            }
        
        all_kl_values = [kl for walk_kl in valid_kl_series for kl in walk_kl if kl is not None]
        
        if not all_kl_values:
            return {
                "mean_kl": 0.0,
                "std_kl": 0.0,
                "max_kl": 0.0,
                "min_kl": 0.0,
                "num_steps_total": 0
            }
        
        return {
            "mean_kl": float(np.mean(all_kl_values)),
            "std_kl": float(np.std(all_kl_values)),
            "max_kl": float(np.max(all_kl_values)),
            "min_kl": float(np.min(all_kl_values)),
            "num_steps_total": len(all_kl_values)
        }
    
    def _save_aggregated_results(self, main_folder: str):
        """Save aggregated results across all context windows."""
        
        # Prepare aggregated data
        aggregated_data = {
            "context_windows": [r["context_window"] for r in self.experiment_results],
            "repeater_error_rates": [r["error_summary"]["repeater_error_rate"] for r in self.experiment_results],
            "ascender_error_rates": [r["error_summary"]["ascender_error_rate"] for r in self.experiment_results],
            "even_error_rates": [r["error_summary"]["even_error_rate"] for r in self.experiment_results],
            "broken_graph_error_rates": [r["error_summary"]["broken_graph_error_rate"] for r in self.experiment_results],
            "total_steps": [r["error_summary"]["total_steps"] for r in self.experiment_results],
            "kl_stats": [r["kl_series_stats"] for r in self.experiment_results]
        }
        
        # Save as JSON
        aggregated_path = os.path.join(main_folder, "aggregated_results", "aggregated_results.json")
        with open(aggregated_path, "w") as f:
            json.dump(aggregated_data, f, indent=2)
        
        # Save as CSV for easy plotting
        csv_path = os.path.join(main_folder, "aggregated_results", "error_rates_by_context.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "context_window", "repeater_error_rate", "ascender_error_rate", 
                "even_error_rate", "broken_graph_error_rate", "total_steps",
                "mean_kl", "std_kl", "max_kl"
            ])
            
            for result in self.experiment_results:
                kl_stats = result["kl_series_stats"]
                writer.writerow([
                    result["context_window"],
                    result["error_summary"]["repeater_error_rate"],
                    result["error_summary"]["ascender_error_rate"],
                    result["error_summary"]["even_error_rate"],
                    result["error_summary"]["broken_graph_error_rate"],
                    result["error_summary"]["total_steps"],
                    kl_stats.get("mean_kl", 0),
                    kl_stats.get("std_kl", 0),
                    kl_stats.get("max_kl", 0)
                ])
    
    def _save_large_scale_aggregated_results(self, main_folder: str):
        """Save aggregated results for large-scale experiments."""
        
        # Prepare aggregated data (similar to original but handles large-scale)
        aggregated_data = {
            "context_windows": [r["context_window"] for r in self.experiment_results],
            "repeater_error_rates": [r["error_summary"]["repeater_error_rate"] for r in self.experiment_results],
            "ascender_error_rates": [r["error_summary"]["ascender_error_rate"] for r in self.experiment_results],
            "even_error_rates": [r["error_summary"]["even_error_rate"] for r in self.experiment_results],
            "broken_graph_error_rates": [r["error_summary"]["broken_graph_error_rate"] for r in self.experiment_results],
            "total_steps": [r["error_summary"]["total_steps"] for r in self.experiment_results],
            "num_walks_per_experiment": [r["num_walks"] for r in self.experiment_results],
            "is_large_scale": [r["is_large_scale"] for r in self.experiment_results]
        }
        
        # Save as JSON
        aggregated_path = os.path.join(main_folder, "aggregated_results", "aggregated_results.json")
        with open(aggregated_path, "w") as f:
            json.dump(aggregated_data, f, indent=2)
        
        # Save as CSV for easy plotting
        csv_path = os.path.join(main_folder, "aggregated_results", "error_rates_by_context.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "context_window", "repeater_error_rate", "ascender_error_rate", 
                "even_error_rate", "broken_graph_error_rate", "total_steps",
                "num_walks", "is_large_scale"
            ])
            
            for result in self.experiment_results:
                writer.writerow([
                    result["context_window"],
                    result["error_summary"]["repeater_error_rate"],
                    result["error_summary"]["ascender_error_rate"],
                    result["error_summary"]["even_error_rate"],
                    result["error_summary"]["broken_graph_error_rate"],
                    result["error_summary"]["total_steps"],
                    result["num_walks"],
                    result["is_large_scale"]
                ])
    
    def plot_context_window_analysis(self, results_folder: str, save_plots: bool = True):
        """
        Plot the results of context window analysis.
        
        Args:
            results_folder: Path to the multi-experiment results folder
            save_plots: Whether to save plots to disk
        """
        import matplotlib.pyplot as plt
        
        # Load aggregated results
        aggregated_path = os.path.join(results_folder, "aggregated_results", "aggregated_results.json")
        with open(aggregated_path, "r") as f:
            data = json.load(f)
        
        context_windows = data["context_windows"]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Context Window Analysis: Rule Violation Rates", fontsize=16)
        
        # Plot 1: All error rates
        axes[0, 0].plot(context_windows, data["repeater_error_rates"], 'ro-', label="Repeater", linewidth=2, markersize=6)
        axes[0, 0].plot(context_windows, data["ascender_error_rates"], 'bs-', label="Ascender", linewidth=2, markersize=6)
        axes[0, 0].plot(context_windows, data["even_error_rates"], 'g^-', label="Even", linewidth=2, markersize=6)
        axes[0, 0].set_xlabel("Context Window Size")
        axes[0, 0].set_ylabel("Error Rate")
        axes[0, 0].set_title("Rule Violation Rates vs Context Window")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Focus on repeater rule (the main interest)
        axes[0, 1].plot(context_windows, data["repeater_error_rates"], 'ro-', linewidth=3, markersize=8)
        axes[0, 1].set_xlabel("Context Window Size")
        axes[0, 1].set_ylabel("Repeater Rule Error Rate")
        axes[0, 1].set_title("Repeater Rule Violations vs Context Window")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: KL divergence statistics
        mean_kl = [stats.get("mean_kl", 0.0) if stats else 0.0 for stats in data["kl_stats"]]
        max_kl = [stats.get("max_kl", 0.0) if stats else 0.0 for stats in data["kl_stats"]]
        axes[1, 0].plot(context_windows, mean_kl, 'mo-', label="Mean KL", linewidth=2, markersize=6)
        axes[1, 0].plot(context_windows, max_kl, 'co-', label="Max KL", linewidth=2, markersize=6)
        axes[1, 0].set_xlabel("Context Window Size")
        axes[1, 0].set_ylabel("KL Divergence")
        axes[1, 0].set_title("KL Divergence vs Context Window")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Total steps (model activity)
        axes[1, 1].plot(context_windows, data["total_steps"], 'ko-', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel("Context Window Size")
        axes[1, 1].set_ylabel("Total Generation Steps")
        axes[1, 1].set_title("Model Activity vs Context Window")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plots_folder = os.path.join(results_folder, "plots")
            plt.savefig(os.path.join(plots_folder, "context_window_analysis.png"), dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {os.path.join(plots_folder, 'context_window_analysis.png')}")
        
        plt.show()
        
        return fig, axes


def run_context_window_study(
    model,
    graph, 
    vocab,
    rules,
    context_windows: Optional[List[int]] = None,
    base_config: Optional[Dict] = None,
    num_walks: int = 100
) -> str:
    """
    Convenience function to run a complete context window study.
    
    Args:
        model: Language model to evaluate
        graph: Graph object
        vocab: Vocabulary object  
        rules: List of rule objects
        context_windows: List of context window sizes (default: [16, 32, 64, 128, 256, 512])
        base_config: Base configuration dict
        num_walks: Number of walks per experiment
        
    Returns:
        Path to results folder
    """
    
    if context_windows is None:
        context_windows = [16, 32, 64, 128, 256, 512]
    
    if base_config is None:
        base_config = {
            "min_start_length": 3,
            "max_start_length": 8,
            "model_type": str(type(model).__name__),
            "graph_size": graph.n
        }
    
    runner = MultiExperimentRunner()
    results_folder = runner.run_context_window_experiments(
        model=model,
        graph=graph,
        vocab=vocab,
        rules=rules,
        context_windows=context_windows,
        base_config=base_config,
        num_walks_per_experiment=num_walks,
        verbose=True
    )
    
    return results_folder

def run_multi_model_repeater_study(
    models_dict,
    graph,
    vocab,
    rules,
    context_windows=None,
    num_walks=100,
    base_config=None,
    output_folder="multi_model_repeater_study"
):
    """
    Run repeater analysis across multiple models and context windows.
    
    Args:
        models_dict: Dictionary mapping model_name -> model_object
        graph: Graph object
        vocab: Vocabulary object
        rules: List of rule objects
        context_windows: List of context window sizes
        num_walks: Number of walks per experiment
        base_config: Base configuration dict
        output_folder: Folder to save results
        
    Returns:
        Dictionary with model -> context_window -> k -> violation_rate structure
    """
    from ..llm.evaluation import evaluate_model
    from ..llm.evaluation_vis import collect_repeater_violations_by_length, save_multi_model_repeater_data
    import os
    from datetime import datetime
    
    if context_windows is None:
        context_windows = [16, 32, 64, 128, 256, 512]
    
    if base_config is None:
        base_config = {
            "min_start_length": 3,
            "max_start_length": 8,
            "graph_size": graph.n
        }
    
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(output_folder, f"multi_model_study_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)
    
    # Store results for all models
    all_results = {}
    
    print(f"Running multi-model repeater study with {len(models_dict)} models")
    print(f"Context windows: {context_windows}")
    print(f"Results will be saved to: {results_folder}")
    
    for model_name, model in models_dict.items():
        print(f"\n=== Evaluating model: {model_name} ===")
        model_results = {}
        
        for context_window in context_windows:
            print(f"  Context window: {context_window}")
            
            # Set model context window if possible
            if hasattr(model, 'context_window'):
                model.context_window = context_window
            elif hasattr(model, 'max_length'):
                model.max_length = context_window
            
            # Run evaluation
            try:
                evaluation_results, error_summary, kl_series = evaluate_model(
                    model=model,
                    graph=graph,
                    vocab=vocab,
                    num_walks=num_walks,
                    min_start_length=base_config.get("min_start_length", 3),
                    max_start_length=base_config.get("max_start_length", 8),
                    rules=rules,
                    verbose=False
                )
                
                # Collect repeater violations by length
                violation_rates = collect_repeater_violations_by_length(
                    evaluation_results, rules, graph
                )
                
                model_results[context_window] = violation_rates
                
                print(f"    Overall repeater error rate: {error_summary['repeater_error_rate']:.4f}")
                for k, rate in violation_rates.items():
                    print(f"    Repeater k={k} violation rate: {rate:.4f}")
                    
            except Exception as e:
                print(f"    Error evaluating model {model_name} with context {context_window}: {e}")
                model_results[context_window] = {}
        
        all_results[model_name] = model_results
    
    # Save results
    output_path = os.path.join(results_folder, "multi_model_repeater_data.json")
    save_multi_model_repeater_data(all_results, output_path)
    
    # Generate plot
    from ..llm.evaluation_vis import plot_repeater_context_analysis
    plot_path = os.path.join(results_folder, "repeater_context_analysis.png")
    plot_repeater_context_analysis(all_results, output_path=plot_path)
    
    print(f"\n=== Multi-model repeater study complete ===")
    print(f"Data saved to: {output_path}")
    print(f"Plot saved to: {plot_path}")
    
    return all_results

def run_repeater_density_study(
    graph,
    vocab_base,
    rules,
    density_configurations,
    model_config=None,
    training_config=None,
    eval_config=None,
    output_folder="repeater_density_study"
):
    """
    Study how repeater node density in training data affects model accuracy.
    
    Args:
        graph: Graph object
        vocab_base: Base vocabulary (can be from minimal training data)
        rules: List of rule objects  
        density_configurations: List of dicts, each containing:
            {
                "name": "high_density_node_5",
                "densities": {node_id: additional_walk_count},
                "description": "High exposure for node 5"
            }
        model_config: Dict with model parameters
        training_config: Dict with training parameters  
        eval_config: Dict with evaluation parameters
        output_folder: Folder to save results
        
    Returns:
        Dictionary with density_config -> accuracy_metrics structure
    """
    from ..data.preparation import prepare_density_controlled_training_data
    from ..llm.training import train_model
    from ..llm.evaluation import evaluate_model
    from ..llm.evaluation_vis import collect_repeater_violations_by_length
    import os
    from datetime import datetime
    
    # Default configurations
    if model_config is None:
        model_config = {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1
        }
    
    if training_config is None:
        training_config = {
            "batch_size": 32,
            "num_epochs": 10,
            "learning_rate": 0.001,
            "num_walks": 1000,
            "min_length": 3,
            "max_length": 8
        }
    
    if eval_config is None:
        eval_config = {
            "num_walks": 200,
            "min_start_length": 3,
            "max_start_length": 8
        }
    
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(output_folder, f"density_study_{timestamp}")
    os.makedirs(results_folder, exist_ok=True)
    
    # Store results for all density configurations
    all_results = {}
    
    print(f"Running repeater density study with {len(density_configurations)} configurations")
    print(f"Results will be saved to: {results_folder}")
    
    for config in density_configurations:
        config_name = config["name"]
        densities = config["densities"]
        description = config.get("description", "")
        
        print(f"\n=== Configuration: {config_name} ===")
        print(f"Description: {description}")
        print(f"Density targets: {densities}")
        
        try:
            # Prepare training data with controlled densities
            training_data, vocab, density_stats = prepare_density_controlled_training_data(
                graph=graph,
                num_walks=training_config["num_walks"],
                min_length=training_config["min_length"],
                max_length=training_config["max_length"],
                rules=rules,
                repeater_densities=densities,
                verbose=True
            )
            
            print(f"Training data prepared: {density_stats['total_walks']} walks")
            print(f"Repeater exposures: {density_stats['repeater_exposure_counts']}")
            
            # Train model
            model = train_model(
                training_data=training_data,
                vocab=vocab,
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                dropout=model_config["dropout"],
                batch_size=training_config["batch_size"],
                num_epochs=training_config["num_epochs"],
                learning_rate=training_config["learning_rate"],
                verbose=True
            )
            
            print("Model training completed")
            
            # Evaluate model
            evaluation_results, error_summary, kl_series = evaluate_model(
                model=model,
                graph=graph,
                vocab=vocab,
                num_walks=eval_config["num_walks"],
                min_start_length=eval_config["min_start_length"],
                max_start_length=eval_config["max_start_length"],
                rules=rules,
                verbose=True
            )
            
            # Collect detailed repeater violations by length
            violation_rates_by_k = collect_repeater_violations_by_length(
                evaluation_results, rules, graph
            )
            
            # Store results
            config_results = {
                "config_name": config_name,
                "description": description,
                "density_targets": densities,
                "density_stats": density_stats,
                "error_summary": error_summary,
                "violation_rates_by_k": violation_rates_by_k,
                "evaluation_results": evaluation_results
            }
            
            all_results[config_name] = config_results
            
            print(f"Overall repeater error rate: {error_summary['repeater_error_rate']:.4f}")
            for k, rate in violation_rates_by_k.items():
                print(f"Repeater k={k} violation rate: {rate:.4f}")
                
        except Exception as e:
            print(f"Error in configuration {config_name}: {e}")
            all_results[config_name] = {
                "config_name": config_name,
                "error": str(e),
                "density_targets": densities
            }
    
    # Save results
    import json
    output_path = os.path.join(results_folder, "density_study_results.json")
    
    # Prepare serializable results
    serializable_results = {}
    for config_name, results in all_results.items():
        if "error" not in results:
            serializable_results[config_name] = {
                "config_name": results["config_name"],
                "description": results["description"],
                "density_targets": results["density_targets"],
                "density_stats": results["density_stats"],
                "error_summary": results["error_summary"],
                "violation_rates_by_k": results["violation_rates_by_k"]
                # Skip evaluation_results as it's too large for JSON
            }
        else:
            serializable_results[config_name] = results
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n=== Repeater density study complete ===")
    print(f"Results saved to: {output_path}")
    
    return all_results

def analyze_density_vs_accuracy(density_study_results):
    """
    Analyze the relationship between repeater density and accuracy.
    
    Args:
        density_study_results: Results from run_repeater_density_study
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "density_accuracy_correlation": {},
        "node_specific_analysis": {},
        "summary_stats": {}
    }
    
    # Extract data for analysis
    for config_name, results in density_study_results.items():
        if "error" in results:
            continue
            
        density_stats = results["density_stats"]
        error_summary = results["error_summary"]
        violation_rates = results["violation_rates_by_k"]
        
        # Overall repeater accuracy
        overall_accuracy = 1.0 - error_summary["repeater_error_rate"]
        
        # Node-specific analysis
        for node, exposure_count in density_stats["repeater_exposure_counts"].items():
            if node not in analysis["node_specific_analysis"]:
                analysis["node_specific_analysis"][node] = {
                    "exposures": [],
                    "accuracies": [],
                    "violation_rates_by_k": {}
                }
            
            analysis["node_specific_analysis"][node]["exposures"].append(exposure_count)
            analysis["node_specific_analysis"][node]["accuracies"].append(overall_accuracy)
            
            # Store k-specific violation rates
            for k, rate in violation_rates.items():
                if k not in analysis["node_specific_analysis"][node]["violation_rates_by_k"]:
                    analysis["node_specific_analysis"][node]["violation_rates_by_k"][k] = []
                analysis["node_specific_analysis"][node]["violation_rates_by_k"][k].append(rate)
    
    # Calculate correlations
    import numpy as np
    for node, data in analysis["node_specific_analysis"].items():
        if len(data["exposures"]) > 1:
            correlation = np.corrcoef(data["exposures"], data["accuracies"])[0, 1]
            analysis["density_accuracy_correlation"][node] = correlation
    
    return analysis