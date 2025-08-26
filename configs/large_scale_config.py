"""
Large-scale experiment configuration for GraphVerse.
Designed for 10K vertex graphs, 1M walks, and multi-context window analysis.
"""

import numpy as np
from typing import List, Dict, Any

# Core experiment parameters
LARGE_SCALE_CONFIG = {
    # Graph structure
    "n": 10000,  # 10K vertices
    "min_edge_density": 0.4,  # Target edge density
    "edge_concentration": 0.8,  # Dirichlet concentration for edge weights
    "exponential_scale": 1.2,  # Scale parameter for edge weight distribution (trackable experiment parameter)
    # Note: Edge weights use exponential distribution (scale=1.2) for less peaked but distinguishable probabilities
    
    # Walk generation
    "num_walks": 1000000,  # 1M walks total
    "walk_length_multiplier": 2,  # walks = 2x context window
    
    # Rule configuration (percentage-based for large graphs)
    "use_percentages": True,
    "rule_percentages": {
        "ascenders": 10.0,    # 10% = 1000 nodes
        "evens": 15.0,        # 15% = 1500 nodes  
        "repeaters": 15.0,    # 15% = 1500 nodes (100 per k-value)
    },
    
    # Context window experiments
    "context_windows": [8, 16, 32, 64, 128, 256],
    
    # Repeater analysis - spans context boundaries
    "repeater_k_values": [2, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 320],
    
    # Memory management
    "trajectory_sampling": {
        "enabled": True,
        "sample_rate": 0.02,  # Store full trajectories for 2% of walks (20K walks)
        "stratified": True,   # Ensure sampling across termination reasons
        "min_samples_per_outcome": 1000,  # Minimum samples per termination type
    },
    
    # Batch processing  
    "batch_processing": {
        "enabled": True,
        "walk_batch_size": 50000,    # Process 50K walks per batch
        "save_frequency": 10000,     # Save intermediate results every 10K walks
        "memory_limit_gb": 32,       # Memory limit for processing
    },
    
    # Parallel processing configuration
    "parallelization": {
        "enabled": True,               # Enable parallel processing
        "strategy": "auto",           # "auto", "cpu_only", "gpu_preferred", "sequential"
        "cpu_workers": None,          # Number of CPU workers (None = auto-detect)
        "gpu_batch_size": 1024,       # Batch size for GPU processing
        "force_sequential_threshold": 100,  # Use sequential for small workloads
        "memory_per_worker_gb": 2,    # Estimated memory per worker
        "fallback_enabled": True,     # Fall back to sequential if parallel fails
        "progress_update_interval": 1000,  # Progress updates every N walks
    },
    
    # Training parameters
    "training": {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_size": 512,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
    },
    
    # Evaluation settings
    "evaluation": {
        "track_token_details": True,
        "verbose_frequency": 10000,  # Progress updates every 10K walks
        "checkpoint_frequency": 100000,  # Checkpoints every 100K walks
    },
    
    # Storage optimization
    "storage": {
        "compress_trajectories": True,
        "use_hdf5": True,
        "save_full_distributions": "sampled",  # "all", "sampled", or "none"
        "keep_intermediate_files": False,  # Clean up during processing
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
    Generate repeater configuration that spans the context window boundary.
    
    Args:
        context_window: Size of context window
        
    Returns:
        Dictionary with repeater min/max steps
    """
    # Create distribution around context window
    # Learnable: k <= context_window
    # Challenging: k > context_window  
    
    return {
        "repeater_min_steps": max(2, context_window // 4),  # Start well below context
        "repeater_max_steps": context_window * 2,           # Extend well beyond context
        "learnable_range": list(range(2, context_window + 1)),
        "challenging_range": list(range(context_window + 1, context_window * 2 + 1)),
        "context_boundary": context_window
    }

def estimate_memory_requirements(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate memory requirements for the large-scale experiment.
    
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
        
        # Training data (assuming avg walk length = 20)
        "training_data": (num_walks * 20 * 4) / (1024**3),  # int32
        
        # Full trajectory storage (sampled)
        "trajectory_full": (num_walks * sampling_rate * n * 10 * 4) / (1024**3),  # vocab_size * avg_steps * float32
        
        # Summary metrics for all walks
        "trajectory_summary": (num_walks * 50 * 4) / (1024**3),  # ~50 metrics per walk
        
        # Model parameters (estimated)
        "model_parameters": 0.5,  # ~500MB for large transformer
        
        # Working memory during evaluation
        "evaluation_working": 2.0,  # Estimated working memory
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

def validate_large_scale_config(config: Dict[str, Any]) -> Dict[str, Any]:
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
    if memory_est["total_estimated"] > 64:  # More than 64GB
        validation["warnings"].append(f"High memory requirement: {memory_est['total_estimated']:.1f}GB")
        validation["recommendations"].append("Consider reducing trajectory sampling rate")
    
    # Parallelization validation and performance estimation
    parallelization = config.get("parallelization", {})
    parallel_enabled = parallelization.get("enabled", False)
    
    # Runtime estimation with parallelization
    num_walks = config["num_walks"]
    num_contexts = len(config["context_windows"])
    
    if parallel_enabled and parallelization.get("strategy") != "sequential":
        # Detect hardware for realistic performance estimates
        try:
            from ..utils.hardware_detection import detect_hardware_capabilities
            hw_info = detect_hardware_capabilities()
            
            # Estimate performance based on detected hardware
            if hw_info.recommended_strategy.startswith("gpu_"):
                walks_per_second = 500  # Optimistic GPU estimate
                validation["recommendations"].append(f"GPU acceleration detected: {hw_info.gpu_type}")
            elif hw_info.cpu_cores >= 8:
                speedup_factor = min(hw_info.cpu_cores - 2, 8)  # Realistic speedup limit
                walks_per_second = 60 * speedup_factor / 8  # Scale from baseline
                validation["recommendations"].append(f"CPU parallel processing with {hw_info.cpu_cores} cores")
            else:
                walks_per_second = 80  # Modest improvement for smaller systems
                
        except ImportError:
            walks_per_second = 120  # Assume modest parallel improvement
            validation["recommendations"].append("Install parallel processing dependencies for better performance")
    else:
        walks_per_second = 60  # Conservative sequential estimate
        if num_walks * num_contexts > 100000:
            validation["recommendations"].append("Consider enabling parallel processing for large workloads")
    
    total_walks = num_walks * num_contexts
    validation["estimated_runtime_hours"] = total_walks / (walks_per_second * 3600)
    
    if validation["estimated_runtime_hours"] > 48:  # More than 2 days
        validation["warnings"].append(f"Long estimated runtime: {validation['estimated_runtime_hours']:.1f} hours")
        if not parallel_enabled:
            validation["recommendations"].append("Enable parallel processing to reduce runtime")
    
    # Parallelization-specific validations
    if parallel_enabled:
        cpu_workers = parallelization.get("cpu_workers")
        if cpu_workers and cpu_workers > 16:
            validation["warnings"].append(f"High worker count ({cpu_workers}) may cause overhead")
            
        memory_per_worker = parallelization.get("memory_per_worker_gb", 2)
        if cpu_workers and cpu_workers * memory_per_worker > memory_est["total_estimated"]:
            validation["warnings"].append("Worker memory requirements may exceed available memory")
    
    # Storage validation
    if not config["storage"]["compress_trajectories"]:
        validation["recommendations"].append("Enable trajectory compression for large-scale experiments")
    
    return validation

# Example usage and validation
if __name__ == "__main__":
    print("Large-Scale GraphVerse Experiment Configuration")
    print("=" * 60)
    
    # Show memory estimates
    memory_est = estimate_memory_requirements(LARGE_SCALE_CONFIG)
    print("\nMemory Requirements Estimate:")
    for component, size_gb in memory_est.items():
        print(f"  {component:.<30} {size_gb:>8.2f} GB")
    
    # Show validation
    validation = validate_large_scale_config(LARGE_SCALE_CONFIG)
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
    experiments = create_context_experiment_plan(LARGE_SCALE_CONFIG)
    print(f"\nPlanned Experiments: {len(experiments)} context windows")
    for exp in experiments:
        ctx = exp["context_window_size"]
        learnable = len(exp["learnable_range"])
        challenging = len(exp["challenging_range"])
        print(f"  Context {ctx:>3}: {learnable} learnable + {challenging} challenging repeater lengths")