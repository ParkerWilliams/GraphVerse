"""
Large-scale evaluation utilities for GraphVerse experiments.
Designed for handling 1M+ walks with memory efficiency and batch processing.
"""

import os
import json
import pickle
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator
import numpy as np
from tqdm import tqdm

from .evaluation import evaluate_model, get_large_scale_trajectory_config, estimate_trajectory_memory_usage
from ..utils.experiment_manager import save_trajectory_metadata, save_config
from ..analysis.metadata import EvaluationTrajectoryMetadata


class LargeScaleEvaluator:
    """
    Handles evaluation of models on large numbers of walks with memory management,
    batch processing, and checkpoint recovery.
    """
    
    def __init__(self, experiment_folder: str, checkpoint_frequency: int = 100000):
        """
        Initialize large-scale evaluator.
        
        Args:
            experiment_folder: Folder to save results and checkpoints
            checkpoint_frequency: How often to save checkpoints (number of walks)
        """
        self.experiment_folder = experiment_folder
        self.checkpoint_frequency = checkpoint_frequency
        
        # Create necessary directories
        os.makedirs(os.path.join(experiment_folder, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(experiment_folder, "evaluation"), exist_ok=True)
        os.makedirs(os.path.join(experiment_folder, "batches"), exist_ok=True)
        
        # State tracking
        self.completed_walks = 0
        self.start_time = None
        self.batch_results = []
        self.aggregated_trajectories = []
        
    def evaluate_large_scale(
        self,
        model,
        graph,
        vocab,
        num_walks: int,
        min_start_length: int,
        max_start_length: int,
        rules,
        batch_size: int = 50000,
        trajectory_sampling_config: Optional[Dict] = None,
        resume_from_checkpoint: bool = True,
        verbose: bool = True
    ) -> str:
        """
        Evaluate model on large number of walks using batch processing.
        
        Args:
            model: Language model to evaluate
            graph: Graph object
            vocab: Vocabulary object
            num_walks: Total number of walks to generate
            min_start_length: Minimum starting walk length
            max_start_length: Maximum starting walk length
            rules: List of rule objects
            batch_size: Number of walks per batch
            trajectory_sampling_config: Configuration for trajectory sampling
            resume_from_checkpoint: Whether to resume from existing checkpoint
            verbose: Whether to print progress
            
        Returns:
            Path to results folder
        """
        self.start_time = time.time()
        
        # Set up trajectory sampling if not provided
        if trajectory_sampling_config is None:
            trajectory_sampling_config = get_large_scale_trajectory_config(
                num_walks, sample_rate=0.02, stratified=True
            )
        
        # Save configuration
        config = {
            "num_walks": num_walks,
            "batch_size": batch_size,
            "min_start_length": min_start_length,
            "max_start_length": max_start_length,
            "trajectory_sampling": trajectory_sampling_config,
            "experiment_start_time": datetime.now().isoformat(),
            "graph_size": graph.n,
            "num_rules": len(rules)
        }
        save_config(config, self.experiment_folder)
        
        if verbose:
            print(f"Starting large-scale evaluation: {num_walks:,} walks")
            print(f"Batch size: {batch_size:,}")
            print(f"Trajectory sampling: {trajectory_sampling_config.get('description', 'Default')}")
            
            # Memory estimation
            memory_est = estimate_trajectory_memory_usage(
                num_walks, trajectory_sampling_config, vocab_size=len(vocab.token2idx)
            )
            print(f"Estimated memory usage: {memory_est['total_mb']:.1f} MB")
        
        # Resume from checkpoint if exists
        if resume_from_checkpoint:
            checkpoint_file = os.path.join(self.experiment_folder, "checkpoints", "latest_checkpoint.pkl")
            if os.path.exists(checkpoint_file):
                if verbose:
                    print("Resuming from checkpoint...")
                self.load_checkpoint(checkpoint_file)
        
        # Process walks in batches
        remaining_walks = num_walks - self.completed_walks
        
        if remaining_walks > 0:
            for batch_start in tqdm(
                range(self.completed_walks, num_walks, batch_size),
                desc="Processing batches",
                initial=self.completed_walks // batch_size,
                total=(num_walks + batch_size - 1) // batch_size
            ):
                batch_end = min(batch_start + batch_size, num_walks)
                batch_walks = batch_end - batch_start
                
                if verbose:
                    print(f"\nProcessing batch {batch_start//batch_size + 1}: walks {batch_start+1}-{batch_end}")
                
                # Process this batch
                batch_results = self.process_batch(
                    model, graph, vocab, batch_walks, min_start_length, max_start_length,
                    rules, trajectory_sampling_config, batch_start, verbose
                )
                
                # Store batch results
                self.batch_results.append(batch_results)
                self.completed_walks = batch_end
                
                # Save checkpoint
                if self.completed_walks % self.checkpoint_frequency == 0 or batch_end == num_walks:
                    self.save_checkpoint(batch_start)
                    
                    if verbose:
                        elapsed = time.time() - self.start_time
                        rate = self.completed_walks / elapsed
                        remaining = (num_walks - self.completed_walks) / rate if rate > 0 else 0
                        print(f"Checkpoint: {self.completed_walks:,}/{num_walks:,} walks completed")
                        print(f"Rate: {rate:.1f} walks/sec, ETA: {remaining/60:.1f} minutes")
        
        # Finalize results
        final_results = self.finalize_results(verbose)
        
        if verbose:
            total_time = time.time() - self.start_time
            print(f"\nLarge-scale evaluation completed in {total_time/60:.1f} minutes")
            print(f"Average rate: {num_walks/total_time:.1f} walks/sec")
        
        return self.experiment_folder
    
    def process_batch(
        self,
        model,
        graph,
        vocab,
        num_walks: int,
        min_start_length: int,
        max_start_length: int,
        rules,
        trajectory_sampling_config: Dict,
        batch_offset: int,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        Process a single batch of walks.
        
        Args:
            model: Language model
            graph: Graph object
            vocab: Vocabulary object
            num_walks: Number of walks in this batch
            min_start_length: Minimum starting walk length
            max_start_length: Maximum starting walk length
            rules: List of rule objects
            trajectory_sampling_config: Trajectory sampling configuration
            batch_offset: Starting walk index for this batch
            verbose: Whether to print progress
            
        Returns:
            Dictionary with batch results
        """
        batch_start_time = time.time()
        
        # Run evaluation on this batch
        evaluation_results, error_summary, kl_series, token_data, progressive_analysis, exemplars, trajectories = evaluate_model(
            model=model,
            graph=graph,
            vocab=vocab,
            num_walks=num_walks,
            min_start_length=min_start_length,
            max_start_length=max_start_length,
            rules=rules,
            verbose=False,  # Suppress detailed output for batches
            track_token_details=True,
            trajectory_sampling_config=trajectory_sampling_config
        )
        
        # Adjust walk indices to account for batch offset
        for result in evaluation_results:
            result["global_walk_idx"] = result.get("walk_idx", 0) + batch_offset
        
        if token_data:
            for token in token_data:
                token["global_walk_idx"] = token.get("walk_idx", 0) + batch_offset
        
        if trajectories and trajectories.walk_trajectories:
            for traj in trajectories.walk_trajectories:
                traj.walk_idx += batch_offset
        
        # Store trajectories for aggregation
        if trajectories:
            self.aggregated_trajectories.extend(trajectories.walk_trajectories)
        
        batch_time = time.time() - batch_start_time
        
        # Save batch data to disk for memory efficiency
        batch_file = os.path.join(self.experiment_folder, "batches", f"batch_{batch_offset//50000}.pkl")
        batch_data = {
            "batch_offset": batch_offset,
            "num_walks": num_walks,
            "evaluation_results": evaluation_results,
            "error_summary": error_summary,
            "kl_series": kl_series,
            "token_data": token_data,
            "progressive_analysis": progressive_analysis,
            "exemplars": exemplars,
            "batch_time": batch_time,
            "batch_rate": num_walks / batch_time if batch_time > 0 else 0
        }
        
        with open(batch_file, "wb") as f:
            pickle.dump(batch_data, f)
        
        if verbose:
            print(f"Batch completed in {batch_time:.1f}s ({num_walks/batch_time:.1f} walks/sec)")
        
        return {
            "batch_offset": batch_offset,
            "num_walks": num_walks,
            "error_summary": error_summary,
            "batch_file": batch_file,
            "batch_time": batch_time,
            "batch_rate": num_walks / batch_time if batch_time > 0 else 0
        }
    
    def save_checkpoint(self, current_batch_start: int):
        """Save checkpoint with current progress."""
        checkpoint_file = os.path.join(self.experiment_folder, "checkpoints", f"checkpoint_{current_batch_start}.pkl")
        latest_file = os.path.join(self.experiment_folder, "checkpoints", "latest_checkpoint.pkl")
        
        checkpoint_data = {
            "completed_walks": self.completed_walks,
            "batch_results": self.batch_results,
            "aggregated_trajectories_count": len(self.aggregated_trajectories),
            "start_time": self.start_time,
            "checkpoint_time": time.time(),
            "current_batch_start": current_batch_start
        }
        
        # Save timestamped checkpoint
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        # Update latest checkpoint
        with open(latest_file, "wb") as f:
            pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self, checkpoint_file: str):
        """Load checkpoint and resume from saved state."""
        with open(checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        self.completed_walks = checkpoint_data["completed_walks"]
        self.batch_results = checkpoint_data["batch_results"]
        self.start_time = checkpoint_data["start_time"]
        
        # Reload aggregated trajectories from batch files
        self.aggregated_trajectories = []
        for batch_result in self.batch_results:
            batch_file = batch_result["batch_file"]
            if os.path.exists(batch_file):
                with open(batch_file, "rb") as f:
                    batch_data = pickle.load(f)
                    # Extract trajectories if they exist
                    # Note: trajectories are stored separately for memory efficiency
        
        print(f"Resumed from checkpoint: {self.completed_walks:,} walks completed")
    
    def finalize_results(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Aggregate results from all batches and create final summary.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with final aggregated results
        """
        if verbose:
            print("Finalizing results...")
        
        # Aggregate error summaries
        total_walks = sum(batch["num_walks"] for batch in self.batch_results)
        aggregated_errors = {
            "repeater_error_rate": 0,
            "ascender_error_rate": 0,
            "even_error_rate": 0,
            "broken_graph_error_rate": 0,
            "total_steps": 0,
            "avg_steps_per_walk": 0
        }
        
        total_repeater_errors = 0
        total_ascender_errors = 0
        total_even_errors = 0
        total_broken_errors = 0
        total_steps = 0
        
        for batch in self.batch_results:
            error_summary = batch["error_summary"]
            batch_walks = batch["num_walks"]
            
            total_repeater_errors += error_summary["repeater_error_rate"] * batch_walks
            total_ascender_errors += error_summary["ascender_error_rate"] * batch_walks
            total_even_errors += error_summary["even_error_rate"] * batch_walks
            total_broken_errors += error_summary["broken_graph_error_rate"] * batch_walks
            total_steps += error_summary["total_steps"]
        
        # Calculate overall error rates
        aggregated_errors["repeater_error_rate"] = total_repeater_errors / total_walks if total_walks > 0 else 0
        aggregated_errors["ascender_error_rate"] = total_ascender_errors / total_walks if total_walks > 0 else 0
        aggregated_errors["even_error_rate"] = total_even_errors / total_walks if total_walks > 0 else 0
        aggregated_errors["broken_graph_error_rate"] = total_broken_errors / total_walks if total_walks > 0 else 0
        aggregated_errors["total_steps"] = total_steps
        aggregated_errors["avg_steps_per_walk"] = total_steps / total_walks if total_walks > 0 else 0
        
        # Create final trajectory metadata
        final_trajectories = EvaluationTrajectoryMetadata(self.aggregated_trajectories) if self.aggregated_trajectories else None
        
        # Save final results
        final_results = {
            "total_walks": total_walks,
            "aggregated_error_summary": aggregated_errors,
            "batch_count": len(self.batch_results),
            "total_evaluation_time": time.time() - self.start_time if self.start_time else 0,
            "average_batch_rate": np.mean([batch["batch_rate"] for batch in self.batch_results]),
            "batch_summary": self.batch_results
        }
        
        # Save aggregated results
        results_file = os.path.join(self.experiment_folder, "evaluation", "final_results.json")
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2, default=lambda x: float(x) if np.isscalar(x) else str(x))
        
        # Save trajectory metadata
        if final_trajectories:
            save_trajectory_metadata(final_trajectories, self.experiment_folder)
        
        if verbose:
            print(f"Final results saved:")
            print(f"  Total walks: {total_walks:,}")
            print(f"  Repeater error rate: {aggregated_errors['repeater_error_rate']:.2%}")
            print(f"  Average rate: {final_results['average_batch_rate']:.1f} walks/sec")
            print(f"  Full trajectories stored: {len(self.aggregated_trajectories):,}")
        
        return final_results


def run_large_scale_context_experiment(
    model,
    graph,
    vocab,
    rules,
    context_windows: List[int],
    num_walks_per_context: int = 1000000,
    base_experiment_folder: str = "large_scale_context_experiments",
    trajectory_sample_rate: float = 0.02,
    batch_size: int = 50000
) -> Dict[str, str]:
    """
    Run large-scale context window experiments across multiple context sizes.
    
    Args:
        model: Language model to evaluate
        graph: Graph object
        vocab: Vocabulary object
        rules: List of rule objects
        context_windows: List of context window sizes to test
        num_walks_per_context: Number of walks per context window experiment
        base_experiment_folder: Base folder for all experiments
        trajectory_sample_rate: Sampling rate for full trajectory storage
        batch_size: Number of walks per batch
        
    Returns:
        Dictionary mapping context_window -> experiment_folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_folder = os.path.join(base_experiment_folder, f"large_scale_run_{timestamp}")
    os.makedirs(main_folder, exist_ok=True)
    
    # Save overall configuration
    overall_config = {
        "experiment_type": "large_scale_context_analysis",
        "context_windows": context_windows,
        "num_walks_per_context": num_walks_per_context,
        "trajectory_sample_rate": trajectory_sample_rate,
        "batch_size": batch_size,
        "total_walks": len(context_windows) * num_walks_per_context,
        "start_time": timestamp
    }
    save_config(overall_config, main_folder)
    
    results = {}
    
    print(f"Starting large-scale context window analysis")
    print(f"Context windows: {context_windows}")
    print(f"Walks per context: {num_walks_per_context:,}")
    print(f"Total walks: {len(context_windows) * num_walks_per_context:,}")
    
    for i, context_window in enumerate(context_windows):
        print(f"\n{'='*60}")
        print(f"Context Window {i+1}/{len(context_windows)}: {context_window}")
        print(f"{'='*60}")
        
        # Create experiment folder for this context window
        exp_folder = os.path.join(main_folder, f"context_{context_window}")
        
        # Set model context window if possible
        if hasattr(model, 'context_window'):
            model.context_window = context_window
        elif hasattr(model, 'max_length'):
            model.max_length = context_window
        
        # Configure trajectory sampling
        trajectory_config = get_large_scale_trajectory_config(
            num_walks_per_context, sample_rate=trajectory_sample_rate, stratified=True
        )
        
        # Run large-scale evaluation
        evaluator = LargeScaleEvaluator(exp_folder)
        
        try:
            result_folder = evaluator.evaluate_large_scale(
                model=model,
                graph=graph,
                vocab=vocab,
                num_walks=num_walks_per_context,
                min_start_length=3,
                max_start_length=8,
                rules=rules,
                batch_size=batch_size,
                trajectory_sampling_config=trajectory_config,
                verbose=True
            )
            
            results[context_window] = result_folder
            print(f"Context {context_window} completed successfully")
            
        except Exception as e:
            print(f"Error in context {context_window}: {e}")
            results[context_window] = f"ERROR: {e}"
    
    # Save final mapping
    results_file = os.path.join(main_folder, "experiment_results_mapping.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nLarge-scale context window analysis completed")
    print(f"Results saved to: {main_folder}")
    
    return results