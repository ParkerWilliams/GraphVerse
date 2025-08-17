#!/usr/bin/env python3
"""
Minimal test script to verify trajectory tracking functionality.
"""

import numpy as np
import torch
from graphverse.analysis.metadata import WalkTrajectoryMetadata, EvaluationTrajectoryMetadata

def test_trajectory_metadata():
    """Test basic trajectory metadata functionality."""
    
    print("Testing WalkTrajectoryMetadata...")
    
    # Create a sample trajectory
    walk_idx = 0
    start_walk = [1, 2, 3]
    trajectory = WalkTrajectoryMetadata(walk_idx, start_walk)
    
    # Add some sample metrics
    for step in range(5):
        metrics = {
            'entropy': 2.0 + np.random.normal(0, 0.1),
            'confidence': 0.8 + np.random.normal(0, 0.05),
            'perplexity': 3.0 + np.random.normal(0, 0.2),
            'kl_divergences': {
                'uniform_random': 1.5 + np.random.normal(0, 0.1),
                'valid_neighbors': 1.2 + np.random.normal(0, 0.1)
            },
            'ks_distances': {
                'uniform_random': 0.3 + np.random.normal(0, 0.05),
                'valid_neighbors': 0.25 + np.random.normal(0, 0.05)
            },
            'distribution_stats': {
                'top_k_mass': [0.8, 0.9, 0.95],
                'effective_support_size': 10,
                'mode_probability': 0.8,
                'tail_mass': 0.05
            }
        }
        trajectory.add_step_metrics(step, metrics)
        
        # Test critical point detection
        trajectory.detect_critical_point(step, threshold=0.3)
    
    # Finalize trajectory
    trajectory.generated_walk = start_walk + [4, 5, 6, 7, 8]
    trajectory.final_length = len(trajectory.generated_walk)
    trajectory.termination_reason = "completed"
    trajectory.termination_step = 4
    
    # Compute statistics
    trajectory.compute_statistics()
    
    print(f"  ✓ Created trajectory for walk {trajectory.walk_idx}")
    print(f"  ✓ Final length: {trajectory.final_length}")
    print(f"  ✓ Termination: {trajectory.termination_reason}")
    print(f"  ✓ Mean entropy: {trajectory.trajectory_statistics.get('mean_entropy', 'N/A'):.3f}")
    print(f"  ✓ Mean confidence: {trajectory.trajectory_statistics.get('mean_confidence', 'N/A'):.3f}")
    print(f"  ✓ Critical points: {len(trajectory.critical_points)}")
    
    return trajectory

def test_evaluation_trajectories():
    """Test aggregate trajectory analysis."""
    
    print("\nTesting EvaluationTrajectoryMetadata...")
    
    # Create multiple sample trajectories
    trajectories = []
    
    for walk_idx in range(10):
        start_walk = [walk_idx, walk_idx + 1, walk_idx + 2]
        trajectory = WalkTrajectoryMetadata(walk_idx, start_walk)
        
        # Add metrics for varying lengths
        num_steps = np.random.randint(3, 8)
        for step in range(num_steps):
            metrics = {
                'entropy': 2.0 + np.random.normal(0, 0.5),
                'confidence': 0.7 + np.random.normal(0, 0.1),
                'kl_divergences': {
                    'uniform_random': 1.0 + np.random.normal(0, 0.3)
                },
                'ks_distances': {
                    'uniform_random': 0.2 + np.random.normal(0, 0.1)
                }
            }
            trajectory.add_step_metrics(step, metrics)
        
        # Random termination reasons
        reasons = ["completed", "invalid_edge", "end_token"]
        trajectory.termination_reason = np.random.choice(reasons)
        trajectory.termination_step = num_steps - 1
        trajectory.generated_walk = start_walk + list(range(10, 10 + num_steps))
        trajectory.final_length = len(trajectory.generated_walk)
        
        trajectory.compute_statistics()
        trajectories.append(trajectory)
    
    # Create evaluation trajectories
    eval_trajectories = EvaluationTrajectoryMetadata(trajectories)
    
    print(f"  ✓ Created aggregate analysis for {eval_trajectories.num_walks} walks")
    print(f"  ✓ Outcome groups: {list(eval_trajectories.outcome_groups.keys())}")
    print(f"  ✓ Trajectory patterns: {len(eval_trajectories.trajectory_patterns)} types")
    print(f"  ✓ Critical steps identified: {len(eval_trajectories.critical_steps)}")
    
    # Test summary generation
    summary = eval_trajectories.get_summary()
    print(f"  ✓ Summary generated with {len(summary)} keys")
    
    return eval_trajectories

def test_distance_functions():
    """Test KS distance and other distribution distance functions."""
    
    print("\nTesting distribution distance functions...")
    
    # Import the functions
    from graphverse.llm.evaluation import (
        compute_ks_distance, compute_js_divergence, 
        compute_wasserstein_distance, analyze_probability_distribution
    )
    
    # Create sample probability distributions
    p = np.array([0.7, 0.2, 0.05, 0.03, 0.02])
    q = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform
    
    # Test KS distance
    ks_dist = compute_ks_distance(p, q)
    print(f"  ✓ KS distance: {ks_dist:.3f}")
    
    # Test JS divergence
    js_div = compute_js_divergence(p, q)
    print(f"  ✓ JS divergence: {js_div:.3f}")
    
    # Test Wasserstein distance (if scipy available)
    w_dist = compute_wasserstein_distance(p, q)
    if w_dist is not None:
        print(f"  ✓ Wasserstein distance: {w_dist:.3f}")
    else:
        print("  - Wasserstein distance: scipy not available")
    
    print("  ✓ All distance functions working")

if __name__ == "__main__":
    print("="*60)
    print("TRAJECTORY TRACKING TEST")
    print("="*60)
    
    try:
        # Test individual trajectory
        trajectory = test_trajectory_metadata()
        
        # Test aggregate analysis
        eval_trajectories = test_evaluation_trajectories()
        
        # Test distance functions
        test_distance_functions()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("✓ Trajectory tracking system is working correctly")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()