#!/usr/bin/env python3
"""
Large-Scale GraphVerse Experiment Demo

This script demonstrates the complete large-scale experiment infrastructure
for analyzing repeater performance across context window boundaries with
1M walks and trajectory uncertainty tracking.

Features demonstrated:
- Large-scale configuration management
- Memory-efficient trajectory sampling  
- Batch processing framework
- Real-time memory monitoring
- Progress checkpointing and recovery
- Comprehensive uncertainty analysis
"""

import os
import sys
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.large_scale_config import LARGE_SCALE_CONFIG, validate_large_scale_config, estimate_memory_requirements
from graphverse.llm.evaluation import get_large_scale_trajectory_config
from graphverse.llm.large_scale_evaluation import LargeScaleEvaluator
from graphverse.utils.multi_experiment_runner import MultiExperimentRunner
from graphverse.utils.memory_monitor import MemoryMonitor, MemoryOptimizer, monitor_large_scale_experiment
from graphverse.utils.checkpoint_manager import CheckpointManager, ProgressTracker


def main():
    """
    Main demonstration of large-scale experiment capabilities.
    """
    print("=" * 80)
    print("GRAPHVERSE LARGE-SCALE EXPERIMENT INFRASTRUCTURE DEMO")
    print("=" * 80)
    
    # 1. Configuration Validation
    print("\n1. CONFIGURATION VALIDATION")
    print("-" * 40)
    
    validation = validate_large_scale_config(LARGE_SCALE_CONFIG)
    memory_est = estimate_memory_requirements(LARGE_SCALE_CONFIG)
    
    print(f"Configuration valid: {validation['valid']}")
    print(f"Estimated memory: {memory_est['total_estimated']:.1f} GB")
    print(f"Estimated runtime: {validation['estimated_runtime_hours']:.1f} hours")
    
    if validation['warnings']:
        print("\nWarnings:")
        for warning in validation['warnings']:
            print(f"  ⚠ {warning}")
    
    if validation['recommendations']:
        print("\nRecommendations:")
        for rec in validation['recommendations']:
            print(f"  💡 {rec}")
    
    # 2. Memory Analysis
    print("\n2. MEMORY REQUIREMENTS ANALYSIS")
    print("-" * 40)
    
    num_walks = LARGE_SCALE_CONFIG["num_walks"]
    memory_limit = 32  # GB
    
    # Get trajectory sampling recommendation
    sample_recommendations = MemoryOptimizer.recommend_sample_rate(
        num_walks=num_walks,
        memory_limit_gb=memory_limit
    )
    
    print(f"For {num_walks:,} walks with {memory_limit}GB memory limit:")
    print(f"Recommended trajectory sampling rate: {sample_recommendations['best_rate']:.1%}")
    
    # Show memory breakdown
    trajectory_config = get_large_scale_trajectory_config(
        num_walks, 
        sample_rate=sample_recommendations['best_rate']
    )
    
    print(f"Trajectory config: {trajectory_config['description']}")
    
    # Batch size recommendation
    batch_rec = MemoryOptimizer.suggest_batch_size(
        num_walks=num_walks,
        memory_limit_gb=memory_limit,
        sample_rate=sample_recommendations['best_rate']
    )
    
    print(f"Recommended batch size: {batch_rec['recommended_batch_size']:,}")
    print(f"Number of batches: {batch_rec['num_batches']}")
    print(f"Peak memory per batch: {batch_rec['estimated_peak_memory_gb']:.1f}GB")
    
    # 3. Context Window Analysis Setup
    print("\n3. CONTEXT WINDOW EXPERIMENT SETUP")
    print("-" * 40)
    
    context_windows = LARGE_SCALE_CONFIG["context_windows"]
    print(f"Context windows to analyze: {context_windows}")
    
    # Show repeater analysis focus
    print("\nRepeater analysis strategy:")
    for ctx in context_windows[:3]:  # Show first few
        from configs.large_scale_config import get_repeater_config_for_context
        rep_config = get_repeater_config_for_context(ctx)
        print(f"  Context {ctx}: {len(rep_config['learnable_range'])} learnable, "
              f"{len(rep_config['challenging_range'])} challenging repeater lengths")
    
    # 4. Checkpointing System Demo
    print("\n4. CHECKPOINTING AND PROGRESS TRACKING")
    print("-" * 40)
    
    # Create demo experiment folder
    demo_folder = "demo_large_scale_experiment"
    os.makedirs(demo_folder, exist_ok=True)
    
    # Set up checkpoint manager
    checkpoint_manager = CheckpointManager(demo_folder, max_checkpoints=5)
    
    # Create progress tracker
    tracker = ProgressTracker(
        experiment_name="large_scale_demo",
        total_items=num_walks,
        checkpoint_manager=checkpoint_manager,
        checkpoint_frequency=batch_rec['recommended_batch_size']
    )
    
    print(f"Checkpoint system initialized:")
    print(f"  Experiment folder: {demo_folder}")
    print(f"  Checkpoint frequency: {batch_rec['recommended_batch_size']:,} walks")
    
    # Simulate some progress
    print("\nSimulating experiment progress...")
    for i in range(3):
        progress_amount = batch_rec['recommended_batch_size']
        tracker.update_progress(
            items_completed=progress_amount,
            batch_number=i,
            current_context_window=context_windows[0] if i < len(context_windows) else None
        )
    
    progress_summary = tracker.get_progress_summary()
    print(f"Progress simulation completed:")
    print(f"  Items processed: {progress_summary['completed_items']:,}")
    print(f"  Progress: {progress_summary['progress_percentage']:.1f}%")
    print(f"  Checkpoints created: {progress_summary['checkpoint_count']}")
    
    # 5. Memory Monitoring Demo
    print("\n5. MEMORY MONITORING CAPABILITIES")
    print("-" * 40)
    
    # Create memory monitor
    monitor = MemoryMonitor(
        memory_limit_gb=memory_limit,
        alert_threshold=0.7,
        critical_threshold=0.9
    )
    
    current_memory = monitor.get_memory_info()
    print(f"Current memory usage: {current_memory['used_memory']/1024**3:.1f} GB")
    print(f"System memory: {current_memory['system_percent']:.1f}% used")
    
    # Show memory monitoring setup
    print(f"Memory monitoring configured:")
    print(f"  Limit: {memory_limit} GB")
    print(f"  Alert threshold: 70%")
    print(f"  Critical threshold: 90%")
    
    # 6. Large-Scale Evaluator Preview
    print("\n6. LARGE-SCALE EVALUATOR FRAMEWORK")
    print("-" * 40)
    
    # Show what a large-scale experiment would look like
    print(f"Large-scale experiment would process:")
    print(f"  Total walks: {num_walks:,}")
    print(f"  Context windows: {len(context_windows)}")
    print(f"  Total experiment walks: {num_walks * len(context_windows):,}")
    print(f"  Trajectory samples stored: {int(num_walks * sample_recommendations['best_rate']):,} per context")
    
    # Estimate full experiment requirements
    total_experiment_walks = num_walks * len(context_windows)
    total_runtime_hours = validation['estimated_runtime_hours'] * len(context_windows)
    
    print(f"\nFull experiment estimates:")
    print(f"  Total runtime: {total_runtime_hours:.1f} hours ({total_runtime_hours/24:.1f} days)")
    print(f"  Total walks: {total_experiment_walks:,}")
    print(f"  Average rate needed: {total_experiment_walks/total_runtime_hours:.0f} walks/hour")
    
    # 7. Integration Summary
    print("\n7. INTEGRATION SUMMARY")
    print("-" * 40)
    
    print("Large-scale infrastructure ready for:")
    print("✓ Memory-efficient trajectory sampling")
    print("✓ Batch processing with checkpointing")
    print("✓ Real-time memory monitoring")
    print("✓ Progress tracking and recovery")
    print("✓ Context window boundary analysis")
    print("✓ Repeater performance characterization")
    print("✓ Comprehensive uncertainty tracking")
    
    print(f"\nTo run actual large-scale experiment:")
    print(f"  1. Prepare graph with {LARGE_SCALE_CONFIG['n']:,} vertices")
    print(f"  2. Configure rules with {LARGE_SCALE_CONFIG['rule_percentages']} distribution")
    print(f"  3. Use MultiExperimentRunner.run_large_scale_context_window_experiments()")
    print(f"  4. Monitor with MemoryMonitor and ProgressTracker")
    print(f"  5. Analyze trajectory metadata for repeater context boundary effects")
    
    # Cleanup demo folder
    import shutil
    shutil.rmtree(demo_folder, ignore_errors=True)
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE - LARGE-SCALE INFRASTRUCTURE READY")
    print("=" * 80)


if __name__ == "__main__":
    main()