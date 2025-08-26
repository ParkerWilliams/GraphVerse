"""
Medium-scale experiment configuration for GraphVerse.
Designed for 1K vertex graphs, 100K walks per context window, and faster iteration.
Perfect for development, pilot studies, and resource-constrained environments.
"""

import numpy as np
from typing import List, Dict, Any

# Core experiment parameters
MEDIUM_SCALE_CONFIG = {
    # Graph structure
    "n": 1000,  # 1K vertices (vs 10K large-scale)
    "min_edge_density": 0.4,  # Target edge density
    "edge_concentration": 0.8,  # Dirichlet concentration for edge weights
    "exponential_scale": 1.2,  # Scale parameter for edge weight distribution (trackable experiment parameter)
    # Note: Edge weights use exponential distribution (scale=1.2) for less peaked but distinguishable probabilities
    
    # Walk generation
    "num_walks": 100000,  # 100K walks per context (vs 1M large-scale)
    "walk_length_multiplier": 2,  # walks = 2x context window
    
    # Rule configuration (percentage-based)
    "use_percentages": True,
    "rule_percentages": {
        "ascenders": 10.0,    # 10% = 100 nodes
        "evens": 15.0,        # 15% = 150 nodes  
        "repeaters": 15.0,    # 15% = 150 nodes
    },
    
    # Context window experiments  
    "context_windows": [4, 8, 16, 32],
    
    # Repeater analysis - 4-bucket boundary testing
    # For context w: xs=0.6w, s=0.9w, l=1.1w, xl=1.4w
    # Context 4: XS:2, S:4, L:4, XL:6 → [2, 4, 6]
    # Context 8: XS:5, S:7, L:9, XL:11 → [5, 7, 9, 11] 
    # Context 16: XS:10, S:14, L:18, XL:22 → [10, 14, 18, 22]
    # Context 32: XS:19, S:29, L:35, XL:45 → [19, 29, 35, 45]
    "repeater_k_values": [2, 4, 5, 6, 7, 9, 10, 11, 14, 18, 19, 22, 29, 35, 45],
    "use_4_bucket_design": True,
    
    # Memory management (more generous sampling for medium scale)
    "trajectory_sampling": {
        "enabled": True,
        "sample_rate": 0.10,  # Store full trajectories for 10% of walks (10K walks)
        "stratified": True,   # Ensure sampling across termination reasons
        "min_samples_per_outcome": 500,  # Minimum samples per termination type
    },
    
    # Batch processing (smaller batches for medium scale)
    "batch_processing": {
        "enabled": True,
        "walk_batch_size": 10000,    # Process 10K walks per batch (vs 50K large-scale)
        "save_frequency": 5000,      # Save intermediate results every 5K walks
        "memory_limit_gb": 8,        # Memory limit for processing (vs 32GB large-scale)
    },
    
    # Training parameters (smaller models for faster training)
    "training": {
        "epochs": 8,           # Fewer epochs for faster iteration (vs 10 large-scale)
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_size": 256,    # Smaller model (vs 512 large-scale)
        "num_layers": 4,       # Fewer layers (vs 6 large-scale)
        "num_heads": 8,
        "dropout": 0.1,
    },
    
    # Evaluation settings
    "evaluation": {
        "track_token_details": True,
        "verbose_frequency": 5000,   # Progress updates every 5K walks
        "checkpoint_frequency": 25000,  # Checkpoints every 25K walks (vs 100K large-scale)
    },
    
    # Storage optimization
    "storage": {
        "compress_trajectories": True,
        "use_hdf5": True,
        "save_full_distributions": "sampled",  # Store full distributions for sampled walks
        "keep_intermediate_files": False,
    },
    
    # Analysis focus
    "analysis_targets": {
        "repeater_context_crossing": True,  # Primary focus
        "uncertainty_trajectories": True,
        "baseline_comparisons": True,
        "phase_transitions": True,
        "statistical_significance": True,
    },
    
    # Distribution comparison configuration
    "distribution_analysis": {
        "enabled": True,
        "baseline_distributions": ["graph_structure", "uniform_valid", "exponential_fitted", "uniform_full"],
        "distance_metrics": ["kl_divergence", "js_divergence", "ks_distance", "l1_distance", "l2_distance", "cosine_similarity"],
        "store_full_comparisons": "sampled",  # "all", "sampled", "none"
        "quality_assessment": True,
        "overlap_analysis": True,
        "progressive_tracking": True,
        "visualization": {
            "enabled": True,
            "generate_dashboards": True,
            "generate_summary_plots": True,
            "plot_format": "png",
            "plot_dpi": 150
        }
    }
}

def get_repeater_config_for_context(context_window: int) -> Dict[str, Any]:
    """
    Generate repeater configuration using 4-bucket boundary testing design.
    
    Creates 4 evenly-sized buckets of repeater lengths:
    - XS (Extra Small): k = 0.6w (well within context)
    - S (Small): k = 0.9w (at context boundary) 
    - L (Large): k = 1.1w (just beyond boundary)
    - XL (Extra Large): k = 1.4w (well beyond boundary)
    
    Args:
        context_window: Size of context window
        
    Returns:
        Dictionary with 4-bucket repeater configuration
    """
    w = context_window
    
    # Calculate k-values for 4 buckets (rounded to integers, min 2)
    k_xs = max(2, round(0.6 * w))  # Extra Small - well within context
    k_s = max(2, round(0.9 * w))   # Small - at context boundary  
    k_l = max(2, round(1.1 * w))   # Large - just beyond boundary
    k_xl = max(2, round(1.4 * w))  # Extra Large - well beyond boundary
    
    # Ensure unique values
    k_values = [k_xs, k_s, k_l, k_xl]
    k_values = sorted(list(dict.fromkeys(k_values)))  # Remove duplicates, keep sorted
    
    # Categorize into learnable vs challenging
    learnable_k = [k for k in k_values if k <= w]      # Should work
    challenging_k = [k for k in k_values if k > w]     # Should fail
    
    return {
        "k_xs": k_xs,
        "k_s": k_s, 
        "k_l": k_l,
        "k_xl": k_xl,
        "all_k_values": k_values,
        "learnable_k_values": learnable_k,
        "challenging_k_values": challenging_k,
        "context_boundary": w,
        "repeater_min_steps": min(k_values),
        "repeater_max_steps": max(k_values),
        "bucket_design": f"XS:{k_xs}, S:{k_s}, L:{k_l}, XL:{k_xl} (boundary at {w})"
    }

def estimate_memory_requirements(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate memory requirements for the medium-scale experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with memory estimates in GB
    """
    n = config["n"]
    num_walks = config["num_walks"]
    sampling_rate = config["trajectory_sampling"]["sample_rate"]
    
    estimates = {
        # Graph  matrix (much smaller for 1K vertices)
        "graph_": (n * n * 4) / (1024**3),  # float32
        
        # Training data (assuming avg walk length = 15 for smaller contexts)
        "training_data": (num_walks * 15 * 4) / (1024**3),  # int32
        
        # Full trajectory storage (sampled)
        "trajectory_full": (num_walks * sampling_rate * n * 8 * 4) / (1024**3),  # vocab_size * avg_steps * float32
        
        # Summary metrics for all walks
        "trajectory_summary": (num_walks * 40 * 4) / (1024**3),  # ~40 metrics per walk
        
        # Model parameters (smaller models)
        "model_parameters": 0.2,  # ~200MB for medium transformer
        
        # Working memory during evaluation
        "evaluation_working": 1.0,  # Estimated working memory
    }
    
    estimates["total_estimated"] = sum(estimates.values())
    estimates["recommended_system_ram"] = estimates["total_estimated"] * 2  # 2x safety factor
    
    return estimates

def create_context_experiment_plan(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create experiment plan for each context window.
    
    Args:
        base_config: Base configuration
        
    Returns:
        List of experiment configurations
    """
    experiments = []
    
    for context_window in base_config["context_windows"]:
        exp_config = base_config.copy()
        exp_config["context_window_size"] = context_window
        exp_config["walk_lengths"] = {
            "min_walk_length": context_window * 3,  # 3w minimum
            "max_walk_length": context_window * 4   # 4w maximum  
        }
        
        # Add repeater configuration for this context window
        repeater_config = get_repeater_config_for_context(context_window)
        exp_config.update(repeater_config)
        
        # Experiment-specific settings
        exp_config["experiment_name"] = f"medium_ctx_{context_window}"
        exp_config["focus_analysis"] = f"repeater_boundary_at_{context_window}"
        
        experiments.append(exp_config)
    
    return experiments

def validate_medium_scale_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and return warnings/recommendations for medium scale.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        "valid": True,
        "warnings": [],
        "recommendations": [],
        "estimated_runtime_hours": 0
    }
    
    # Memory validation (much lower requirements)
    memory_est = estimate_memory_requirements(config)
    if memory_est["total_estimated"] > 16:  # More than 16GB
        validation["warnings"].append(f"High memory requirement: {memory_est['total_estimated']:.1f}GB")
        validation["recommendations"].append("Consider reducing trajectory sampling rate")
    
    # Runtime estimation (much faster)
    num_walks = config["num_walks"]
    num_contexts = len(config["context_windows"])
    walks_per_second = 100  # Higher rate for smaller graphs
    
    total_walks = num_walks * num_contexts
    validation["estimated_runtime_hours"] = total_walks / (walks_per_second * 3600)
    
    if validation["estimated_runtime_hours"] > 12:  # More than 12 hours
        validation["warnings"].append(f"Long estimated runtime: {validation['estimated_runtime_hours']:.1f} hours")
        validation["recommendations"].append("Consider reducing walk count or context windows")
    
    # Storage validation
    if not config["storage"]["compress_trajectories"]:
        validation["recommendations"].append("Enable trajectory compression for efficiency")
    
    # Rule distribution validation for smaller graphs
    n = config["n"]
    min_rule_nodes = 5  # Minimum nodes per rule type
    
    for rule_type, percentage in config["rule_percentages"].items():
        num_nodes = int(n * percentage / 100)
        if num_nodes < min_rule_nodes:
            validation["warnings"].append(f"Low {rule_type} node count: {num_nodes} nodes")
            validation["recommendations"].append(f"Consider increasing {rule_type} percentage for small graphs")
    
    return validation

def get_medium_scale_trajectory_config(num_walks: int, sample_rate: float = 0.10, stratified: bool = True) -> Dict[str, Any]:
    """
    Generate trajectory sampling configuration optimized for medium-scale experiments.
    
    Args:
        num_walks: Total number of walks to be generated
        sample_rate: Fraction of walks to store full trajectory data for
        stratified: Whether to use stratified sampling by termination outcome
        
    Returns:
        Dictionary with trajectory sampling configuration
    """
    # Calculate minimum samples to ensure representation of all outcomes
    min_samples_per_outcome = max(200, int(num_walks * sample_rate / 6))  # Distribute across ~6 outcomes
    
    config = {
        "enabled": True,
        "sample_rate": sample_rate,
        "stratified": stratified,
        "min_samples_per_outcome": min_samples_per_outcome,
        "store_full_distributions": True,  # Can afford full distributions at medium scale
        "memory_efficient": True,
        "description": f"Medium-scale sampling: {sample_rate:.0%} of {num_walks:,} walks = {int(num_walks * sample_rate):,} full trajectories"
    }
    
    return config

# Example usage and validation
if __name__ == "__main__":
    print("Medium-Scale GraphVerse Experiment Configuration")
    print("=" * 60)
    
    # Show memory estimates
    memory_est = estimate_memory_requirements(MEDIUM_SCALE_CONFIG)
    print("\nMemory Requirements Estimate:")
    for component, size_gb in memory_est.items():
        print(f"  {component:.<30} {size_gb:>8.3f} GB")
    
    # Show validation
    validation = validate_medium_scale_config(MEDIUM_SCALE_CONFIG)
    print(f"\nEstimated Runtime: {validation['estimated_runtime_hours']:.1f} hours")
    
    if validation["warnings"]:
        print("\nWarnings:")
        for warning in validation["warnings"]:
            print(f"  ⚠ {warning}")
    
    if validation["recommendations"]:
        print("\nRecommendations:")
        for rec in validation["recommendations"]:
            print(f"  💡 {rec}")
    
    # Show experiment plan
    experiments = create_context_experiment_plan(MEDIUM_SCALE_CONFIG)
    print(f"\nPlanned Experiments: {len(experiments)} context windows")
    print("4-Bucket Context Window Boundary Testing Design:")
    for exp in experiments:
        ctx = exp["context_window_size"]
        repeater_config = get_repeater_config_for_context(ctx)
        walk_min = ctx * 3
        walk_max = ctx * 4
        print(f"  Context {ctx:>2}: {repeater_config['bucket_design']}, walks={walk_min}-{walk_max}")
        print(f"    Expected: Learnable {repeater_config['learnable_k_values']} vs Challenging {repeater_config['challenging_k_values']}")
    
    # Compare to large scale
    print(f"\nMedium vs Large Scale Comparison:")
    print(f"  Vertices: {MEDIUM_SCALE_CONFIG['n']:,} vs 10,000 (10x smaller)")
    print(f"  Walks/context: {MEDIUM_SCALE_CONFIG['num_walks']:,} vs 1,000,000 (10x smaller)")
    print(f"  Memory: {memory_est['total_estimated']:.1f}GB vs ~32GB (8x smaller)")
    print(f"  Runtime: {validation['estimated_runtime_hours']:.1f}h vs ~168h (28x faster)")
    print(f"  Total walks: {len(experiments) * MEDIUM_SCALE_CONFIG['num_walks']:,} vs 6,000,000")