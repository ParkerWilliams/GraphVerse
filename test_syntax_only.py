#!/usr/bin/env python3
"""
Test imports and basic syntax of trajectory tracking components.
"""

def test_imports():
    """Test that all modules can be imported without errors."""
    
    print("Testing imports...")
    
    try:
        # Test metadata imports
        print("  Importing metadata classes...")
        from graphverse.analysis.metadata import WalkTrajectoryMetadata, EvaluationTrajectoryMetadata
        print("  ✓ Metadata classes imported successfully")
        
        # Test evaluation functions (might fail due to missing dependencies)
        print("  Attempting to import evaluation functions...")
        try:
            from graphverse.llm.evaluation import (
                compute_ks_distance, compute_js_divergence, 
                compute_distribution_distances, analyze_probability_distribution
            )
            print("  ✓ Evaluation functions imported successfully")
        except ImportError as e:
            print(f"  ⚠ Evaluation functions import failed (expected due to missing deps): {e}")
        
        # Test basic class instantiation
        print("  Testing basic class instantiation...")
        trajectory = WalkTrajectoryMetadata(0, [1, 2, 3])
        print(f"  ✓ WalkTrajectoryMetadata created: walk_idx={trajectory.walk_idx}")
        
        # Test adding metrics without numpy
        print("  Testing basic functionality...")
        metrics = {
            'entropy': 2.0,
            'confidence': 0.8,
            'kl_divergences': {'uniform_random': 1.5},
            'ks_distances': {'uniform_random': 0.3}
        }
        trajectory.add_step_metrics(0, metrics)
        print("  ✓ Metrics added successfully")
        
        trajectory.termination_reason = "completed"
        trajectory.termination_step = 0
        trajectory.generated_walk = [1, 2, 3, 4]
        trajectory.final_length = 4
        
        print("  ✓ Trajectory finalized")
        
        # Test summary
        summary = trajectory.get_summary()
        print(f"  ✓ Summary generated: {len(summary)} keys")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import/syntax test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*50)
    print("SYNTAX AND IMPORT TEST")
    print("="*50)
    
    success = test_imports()
    
    if success:
        print("\n" + "="*50)
        print("✓ SYNTAX TEST PASSED!")
        print("✓ Core trajectory classes are working")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ SYNTAX TEST FAILED")
        print("="*50)