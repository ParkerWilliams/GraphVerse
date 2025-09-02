#!/usr/bin/env python3
"""
Medium Scale Experiment Runner - 1000 nodes, 100K walks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from configs.medium_scale_config import MEDIUM_SCALE_CONFIG
from scripts.main import main

def run_medium_scale_experiment():
    """Run medium scale experiment with 1000 nodes and 100K walks."""
    config = MEDIUM_SCALE_CONFIG
    
    print("🚀 Starting Medium Scale GraphVerse Experiment")
    print("=" * 60)
    print(f"Nodes: {config['n']:,}")
    print(f"Walks per context: {config['num_walks']:,}")
    print(f"Context windows: {config['context_windows']}")
    print(f"Total walks: {len(config['context_windows']) * config['num_walks']:,}")
    
    # Run experiment for single context window
    for context_window in [16]:  # Focus on context window 16
        print(f"\n🎯 Running context window {context_window}")
        print("-" * 40)
        
        # Calculate rule counts based on percentages
        # Use more conservative percentages for medium scale to avoid conflicts
        n = config['n']
        num_ascenders = int(n * 0.05)  # 5% = 50 nodes
        num_evens = int(n * 0.10)      # 10% = 100 nodes
        num_repeaters = int(n * 0.10)  # 10% = 100 nodes
        
        # Get repeater k-values for this context
        k_values = [k for k in config['repeater_k_values'] 
                   if k <= context_window * 1.5]  # Reasonable range for this context
        repeater_min_steps = min(k_values) if k_values else 2
        repeater_max_steps = max(k_values) if k_values else context_window
        
        print(f"  Rules: {num_ascenders} ascenders, {num_evens} evens, {num_repeaters} repeaters")
        print(f"  Repeater steps: {repeater_min_steps}-{repeater_max_steps}")
        
        # Adjust hyperparameters based on context window size
        if context_window == 16:
            # Use optimized hyperparameters for enhanced model
            lr = 0.001  # More reasonable learning rate with warmup
            eps = 15    # Sufficient epochs with better architecture
            bs = 64     # Moderate batch size (effective 128 with gradient accumulation)
        else:
            lr = 0.001
            eps = 10
            bs = config['training']['batch_size']
        
        # Run the experiment
        main(
            n=n,
            num_walks=config['num_walks'],
            context_window_size=context_window,
            num_ascenders=num_ascenders,
            num_evens=num_evens, 
            num_repeaters=num_repeaters,
            repeater_min_steps=repeater_min_steps,
            repeater_max_steps=repeater_max_steps,
            epochs=eps,
            batch_size=bs,
            learning_rate=lr,
            verbose=True,
            use_percentages=False,  # Use fixed counts, not percentages
            edge_concentration=config.get('edge_concentration', 0.8)  # Will be used as exponential_scale
        )
        
        print(f"✅ Completed context window {context_window}")
    
    print("\n" + "=" * 60)
    print("🎉 Medium Scale Experiment Complete!")
    print("📊 Generating post-rule-encounter filtered visualizations...")
    
    # Generate filtered visualizations
    try:
        from test_walk_visualization import test_aggregated_violation_analysis
        test_aggregated_violation_analysis()
        print("✅ All visualizations generated!")
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
        print("You can run visualizations manually with:")
        print("python -c \"from test_walk_visualization import test_aggregated_violation_analysis; test_aggregated_violation_analysis()\"")

if __name__ == "__main__":
    run_medium_scale_experiment()