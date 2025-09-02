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
    "context_windows": [16],  # Single context window of 16
    
    # Repeater analysis - specific k-values for context 16
    # k=8: 0.5 * context (well within)
    # k=14: 0.875 * context (near boundary)
    # k=18: 1.125 * context (just beyond)
    # k=24: 1.5 * context (well beyond)
    "repeater_k_values": [8, 14, 18, 24],
    "use_4_bucket_design": False,  # Using custom k-values
    
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
        "baseline_distributions": [
            "graph_structure", "uniform_valid", "exponential_fitted", "uniform_full",
            "rule_aware_oracle", "optimal_path_oracle", "repeater_oracle", "ascender_oracle", "even_oracle"
        ],
        "distance_metrics": ["kl_divergence", "js_divergence", "ks_distance", "l1_distance", "l2_distance", "cosine_similarity",
                         "cross_entropy", "mutual_information", "information_gain"],
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
    },
    
    # Comprehensive entropy analysis configuration (medium-scale optimized)
    "entropy_analysis": {
        "enabled": True,
        "comprehensive_metrics": True,  # Enable all entropy metrics
        "entropy_metrics": [
            "model_entropy",           # Basic model uncertainty
            "cross_entropy",           # H(p, q) between model and baselines  
            "mutual_information",      # I(X; Y) between predictions and targets
            "conditional_entropy",     # H(Y|X) uncertainty given context
            "entropy_rate",           # Rate of entropy change over time
            "relative_entropy",       # KL divergence D(p||q)
            "joint_entropy",          # H(X, Y) combined uncertainty
            "information_gain"        # Reduction in uncertainty
        ],
        "temporal_analysis": {
            "enabled": True,
            "track_entropy_dynamics": True,
            "window_size": 8,         # Smaller window for medium scale
            "trend_analysis": True,   # Analyze entropy trends over walk
            "phase_detection": True   # Detect entropy phase transitions
        },
        "correlation_analysis": {
            "enabled": True,
            "cross_metric_correlations": True,
            "baseline_correlations": True,
            "temporal_correlations": True,
            "significance_testing": True
        },
        "visualization": {
            "enabled": True,
            "comprehensive_dashboard": True,     # 4x3 entropy dashboard
            "correlation_heatmaps": True,        # Detailed correlation analysis
            "information_gain_waterfall": True,  # IG comparative analysis
            "temporal_entropy_plots": True,      # Entropy over time
            "distribution_comparisons": True,    # Entropy distributions
            "publication_quality": False,        # Standard quality for medium scale
            "save_formats": ["png"],             # PNG only for faster iteration
            "plot_dpi": 150                      # Standard DPI
        },
        "statistical_testing": {
            "enabled": True,
            "significance_tests": ["t_test", "mann_whitney"],  # Fewer tests for speed
            "multiple_comparisons_correction": "bonferroni",
            "confidence_level": 0.95,
            "effect_size_calculation": True
        }
    },
    
    # Rule violation temporal analysis configuration (medium-scale optimized)
    "violation_temporal_analysis": {
        "enabled": True,
        "lookback_window": 20,                    # Shorter window for faster analysis
        "max_cases_per_rule": 8,                  # Fewer samples for speed
        "min_violation_confidence": 0.6,          # Lower threshold for more cases
        "include_near_violations": False,         # Disable for speed
        
        # Violation case selection strategy (streamlined)
        "case_selection": {
            "ensure_rule_type_diversity": True,   # Balance across repeater, ascender, even
            "prefer_high_confidence": True,       # Prioritize confident violations
            "context_window_diversity": False,    # Disable for speed
            "entropy_pattern_diversity": False,   # Disable for speed
            "minimum_entropy_variance": 0.05     # Lower threshold
        },
        
        # Analysis focus areas (subset for speed)
        "analysis_targets": {
            "oracle_divergence_patterns": True,   # Core analysis
            "entropy_collapse_dynamics": True,    # Core analysis
            "context_boundary_effects": False,    # Disable for speed
            "rule_specific_signatures": True,     # Core analysis
            "predictive_indicators": False,       # Disable for speed
            "baseline_comparison_evolution": True  # Core analysis
        },
        
        # Visualization configuration (optimized)
        "visualization": {
            "entropy_timeline_plots": True,       # 3x3 comprehensive entropy over time
            "individual_case_studies": True,      # 2x3 detailed individual violation cases
            "violation_type_comparison": True,    # 2x2 comparative analysis across rule types
            "context_boundary_analysis": False,   # Disable for speed
            "publication_quality": False,         # Standard quality for speed
            "save_formats": ["png"],              # PNG only for speed
            "plot_dpi": 150,                      # Standard DPI
            "figsize_timeline": [16, 10],         # Smaller plots
            "figsize_case_studies": [18, 10],     # Smaller plots
            "figsize_comparison": [14, 10]        # Smaller plots
        },
        
        # Advanced analysis settings (minimal for speed)
        "advanced_analysis": {
            "entropy_rate_tracking": True,        # Core analysis
            "baseline_correlation_analysis": False, # Disable for speed
            "violation_prediction_modeling": False, # Disable for speed
            "phase_transition_detection": False,   # Disable for speed
            "information_cascade_analysis": False, # Disable for speed
            "uncertainty_decomposition": False    # Disable for speed
        }
    }
}

def get_repeater_config_for_context(context_window: int) -> Dict[str, Any]:
    """
    Generate repeater configuration for specific context window.
    
    For context_window=16, uses specific k-values: [8, 14, 18, 24]
    - k=8: 0.5 * context (well within)
    - k=14: 0.875 * context (near boundary)
    - k=18: 1.125 * context (just beyond)
    - k=24: 1.5 * context (well beyond)
    
    Args:
        context_window: Size of context window
        
    Returns:
        Dictionary with repeater configuration
    """
    w = context_window
    
    # Use specific k-values for context window 16
    if w == 16:
        k_values = [8, 14, 18, 24]
        k_xs = 8   # 0.5 * context (well within)
        k_s = 14   # 0.875 * context (near boundary)
        k_l = 18   # 1.125 * context (just beyond)
        k_xl = 24  # 1.5 * context (well beyond)
    else:
        # Default 4-bucket design for other context windows
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
        
        # Training data (assuming avg walk length = 32 for context 16)
        "training_data": (num_walks * 32 * 4) / (1024**3),  # int32
        
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
            "min_walk_length": context_window * 2,  # 2x context window
            "max_walk_length": context_window * 2   # 2x context window  
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