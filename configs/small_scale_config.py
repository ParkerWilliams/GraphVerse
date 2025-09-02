"""
Small-scale experiment configuration for GraphVerse.
Designed for 100 vertex graphs, 10K walks per context window, for quick testing.
Perfect for rapid development, debugging, and violation entropy analysis testing.
"""

import numpy as np
from typing import List, Dict, Any

# Core experiment parameters
SMALL_SCALE_CONFIG = {
    # Graph structure
    "n": 100,  # 100 vertices for quick testing
    "min_edge_density": 0.4,  # Target edge density
    "edge_concentration": 0.8,  # Dirichlet concentration for edge weights
    "exponential_scale": 1.2,  # Scale parameter for edge weight distribution (trackable experiment parameter)
    # Note: Edge weights use exponential distribution (scale=1.2) for less peaked but distinguishable probabilities
    
    # Walk generation
    "num_walks": 10000,  # 10K walks per context for quick testing
    "walk_length_multiplier": 2,  # walks = 2x context window
    
    # Rule configuration (percentage-based)
    "use_percentages": True,
    "rule_percentages": {
        "ascenders": 10.0,    # 10% = 10 nodes
        "evens": 15.0,        # 15% = 15 nodes  
        "repeaters": 15.0,    # 15% = 15 nodes
    },
    
    # Context window experiments - single context of 16
    "context_windows": [16],
    
    # Repeater analysis - specific k-values for context 16
    "repeater_k_values": [8, 14, 18, 24],
    "use_4_bucket_design": False,  # Simplified for small scale
    
    # Memory management (high sampling rate for small scale)
    "trajectory_sampling": {
        "enabled": True,
        "sample_rate": 0.50,  # Store full trajectories for 50% of walks (5K walks)
        "stratified": True,   # Ensure sampling across termination reasons
        "min_samples_per_outcome": 100,  # Minimum samples per termination type
    },
    
    # Batch processing (small batches)
    "batch_processing": {
        "enabled": True,
        "walk_batch_size": 2000,    # Process 2K walks per batch
        "save_frequency": 1000,     # Save intermediate results every 1K walks
        "memory_limit_gb": 4,       # Memory limit for processing
    },
    
    # Parallel processing configuration
    "parallelization": {
        "enabled": True,               # Enable parallel processing
        "strategy": "auto",           # "auto", "cpu_only", "gpu_preferred", "sequential"
        "cpu_workers": None,          # Number of CPU workers (None = auto-detect)
        "gpu_batch_size": 256,        # Smaller GPU batch size
        "force_sequential_threshold": 50,  # Use sequential for very small workloads
        "memory_per_worker_gb": 1,    # Estimated memory per worker
        "fallback_enabled": True,     # Fall back to sequential if parallel fails
        "progress_update_interval": 500,  # Progress updates every 500 walks
    },
    
    # Training parameters (small model for fast training)
    "training": {
        "epochs": 5,           # Fewer epochs for speed
        "batch_size": 16,      # Smaller batches
        "learning_rate": 0.001,
        "hidden_size": 128,    # Small model
        "num_layers": 3,       # Few layers
        "num_heads": 4,        # Fewer heads
        "dropout": 0.1,
    },
    
    # Evaluation settings
    "evaluation": {
        "track_token_details": True,
        "verbose_frequency": 1000,   # Progress updates every 1K walks
        "checkpoint_frequency": 5000,  # Checkpoints every 5K walks
    },
    
    # Storage optimization
    "storage": {
        "compress_trajectories": True,
        "use_hdf5": True,
        "save_full_distributions": "all",  # Store all distributions for small scale
        "keep_intermediate_files": True,   # Keep files for analysis
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
        "store_full_comparisons": "all",  # Store all comparisons for small scale
        "quality_assessment": True,
        "overlap_analysis": True,
        "progressive_tracking": True,
        "visualization": {
            "enabled": True,
            "generate_dashboards": True,
            "generate_summary_plots": True,
            "publication_quality": True,
            "plot_formats": ["png", "pdf"],
            "plot_dpi": 300,
            "save_individual_plots": True,
            "include_statistics": True,
            "error_bars": True
        }
    },
    
    # Comprehensive entropy analysis configuration
    "entropy_analysis": {
        "enabled": True,
        "comprehensive_metrics": True,
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
            "window_size": 5,         # Smaller rolling window
            "trend_analysis": True,
            "phase_detection": True
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
            "comprehensive_dashboard": True,
            "correlation_heatmaps": True,
            "information_gain_waterfall": True,
            "temporal_entropy_plots": True,
            "distribution_comparisons": True,
            "publication_quality": True,
            "save_formats": ["png", "pdf"],
            "plot_dpi": 300
        },
        "statistical_testing": {
            "enabled": True,
            "significance_tests": ["t_test", "mann_whitney", "ks_test"],
            "multiple_comparisons_correction": "bonferroni",
            "confidence_level": 0.95,
            "effect_size_calculation": True
        }
    },
    
    # Rule violation temporal analysis configuration  
    "violation_temporal_analysis": {
        "enabled": True,
        "lookback_window": 15,                    # Tokens before violation to analyze
        "max_cases_per_rule": 10,                 # Representative samples per rule type
        "min_violation_confidence": 0.6,          # Lower threshold for small scale
        "include_near_violations": True,          # Include high-probability violations that didn't occur
        
        # Violation case selection strategy
        "case_selection": {
            "ensure_rule_type_diversity": True,   # Balance across repeater, ascender, even
            "prefer_high_confidence": True,       # Prioritize confident violations
            "context_window_diversity": False,    # Single context window
            "entropy_pattern_diversity": True,    # Include different entropy signatures
            "minimum_entropy_variance": 0.05     # Lower threshold for small scale
        },
        
        # Analysis focus areas
        "analysis_targets": {
            "oracle_divergence_patterns": True,   # How model diverges from oracles over time
            "entropy_collapse_dynamics": True,    # Model uncertainty patterns before violations
            "context_boundary_effects": True,     # Impact of context window boundaries
            "rule_specific_signatures": True,     # Unique patterns per rule type
            "predictive_indicators": True,        # Early warning signals
            "baseline_comparison_evolution": True  # How all baselines evolve before violations
        },
        
        # Visualization configuration
        "visualization": {
            "entropy_timeline_plots": True,       # 3x3 comprehensive entropy over time
            "individual_case_studies": True,      # 2x3 detailed individual violation cases
            "violation_type_comparison": True,    # 2x2 comparative analysis across rule types
            "context_boundary_analysis": False,   # Skip for single context
            "publication_quality": True,          # High-resolution publication plots
            "save_formats": ["png", "pdf"],       # Multiple output formats
            "plot_dpi": 300,                      # Publication DPI
            "figsize_timeline": [15, 10],         # Smaller size for single context
            "figsize_case_studies": [16, 10],     # Adjusted size
            "figsize_comparison": [12, 10]        # Adjusted size
        },
        
        # Advanced analysis settings
        "advanced_analysis": {
            "entropy_rate_tracking": True,        # Track rate of entropy change
            "baseline_correlation_analysis": True, # Correlations between baseline divergences
            "violation_prediction_modeling": True, # Attempt to predict violations from entropy
            "phase_transition_detection": True,   # Detect phase changes in entropy patterns
            "information_cascade_analysis": True, # Track information flow before violations
            "uncertainty_decomposition": True     # Break down uncertainty sources
        }
    }
}

def get_repeater_config_for_context(context_window: int) -> Dict[str, Any]:
    """
    Generate repeater configuration that spans the context window boundary.
    
    Args:
        context_window: Size of context window
        
    Returns:
        Dictionary with repeater min/max steps
    """
    # Use specific k-values for context window 16
    if context_window == 16:
        k_values = [8, 14, 18, 24]
        return {
            "repeater_min_steps": min(k_values) - 1,  # 7
            "repeater_max_steps": max(k_values) + 1,  # 25
            "learnable_range": [k for k in k_values if k <= context_window],  # [8, 14]
            "challenging_range": [k for k in k_values if k > context_window],  # [18, 24]
            "context_boundary": context_window,
            "repeater_k_values": k_values
        }
    
    # Default for other context windows
    return {
        "repeater_min_steps": max(2, context_window // 4),  # Start well below context
        "repeater_max_steps": context_window * 2,           # Extend well beyond context
        "learnable_range": list(range(2, context_window + 1)),
        "challenging_range": list(range(context_window + 1, context_window * 2 + 1)),
        "context_boundary": context_window
    }

def estimate_memory_requirements(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate memory requirements for the small-scale experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with memory estimates in GB
    """
    n = config["n"]
    num_walks = config["num_walks"]
    sampling_rate = config["trajectory_sampling"]["sample_rate"]
    
    estimates = {
        # Graph adjacency matrix (dense)
        "graph_adjacency": (n * n * 4) / (1024**3),  # float32
        
        # Training data (assuming avg walk length = 16)
        "training_data": (num_walks * 32 * 4) / (1024**3),  # int32 (32 = 2x context window 16)
        
        # Full trajectory storage (sampled)
        "trajectory_full": (num_walks * sampling_rate * n * 8 * 4) / (1024**3),  # vocab_size * avg_steps * float32
        
        # Summary metrics for all walks
        "trajectory_summary": (num_walks * 50 * 4) / (1024**3),  # ~50 metrics per walk
        
        # Model parameters (estimated)
        "model_parameters": 0.1,  # ~100MB for small transformer
        
        # Working memory during evaluation
        "evaluation_working": 0.5,  # Estimated working memory
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
            "min_walk_length": context_window * 2,
            "max_walk_length": context_window * 2
        }
        
        # Add repeater configuration for this context window
        repeater_config = get_repeater_config_for_context(context_window)
        exp_config.update(repeater_config)
        
        # Experiment-specific settings
        exp_config["experiment_name"] = f"context_{context_window}"
        exp_config["focus_analysis"] = f"repeater_learning_boundary_at_{context_window}"
        
        experiments.append(exp_config)
    
    return experiments

def validate_small_scale_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and return warnings/recommendations.
    
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
    
    # Memory validation
    memory_est = estimate_memory_requirements(config)
    if memory_est["total_estimated"] > 8:  # More than 8GB
        validation["warnings"].append(f"High memory requirement: {memory_est['total_estimated']:.1f}GB")
        validation["recommendations"].append("Consider reducing trajectory sampling rate")
    
    # Runtime estimation
    num_walks = config["num_walks"]
    num_contexts = len(config["context_windows"])
    walks_per_second = 150  # Optimistic estimate for small graphs
    
    total_walks = num_walks * num_contexts
    validation["estimated_runtime_hours"] = total_walks / (walks_per_second * 3600)
    
    if validation["estimated_runtime_hours"] > 2:  # More than 2 hours
        validation["warnings"].append(f"Long estimated runtime: {validation['estimated_runtime_hours']:.1f} hours")
        validation["recommendations"].append("Consider reducing num_walks for faster iteration")
    
    return validation

# Example usage and validation
if __name__ == "__main__":
    print("Small-Scale GraphVerse Experiment Configuration")
    print("=" * 60)
    
    # Show memory estimates
    memory_est = estimate_memory_requirements(SMALL_SCALE_CONFIG)
    print("\\nMemory Requirements Estimate:")
    for component, size_gb in memory_est.items():
        print(f"  {component:.<30} {size_gb:>8.2f} GB")
    
    # Show validation
    validation = validate_small_scale_config(SMALL_SCALE_CONFIG)
    print(f"\\nEstimated Runtime: {validation['estimated_runtime_hours']:.1f} hours")
    
    if validation["warnings"]:
        print("\\nWarnings:")
        for warning in validation["warnings"]:
            print(f"  ⚠ {warning}")
    
    if validation["recommendations"]:
        print("\\nRecommendations:")
        for rec in validation["recommendations"]:
            print(f"  💡 {rec}")
    
    # Show experiment plan
    experiments = create_context_experiment_plan(SMALL_SCALE_CONFIG)
    print(f"\\nPlanned Experiments: {len(experiments)} context window")
    for exp in experiments:
        ctx = exp["context_window_size"]
        learnable = len(exp["learnable_range"])
        challenging = len(exp["challenging_range"])
        print(f"  Context {ctx:>2}: {learnable} learnable + {challenging} challenging repeater lengths")