#!/usr/bin/env python3
"""
Monitor the retraining process and show key metrics.
This can be run while retrain_fixed_model.py is executing.
"""

import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

def check_graph_edges(graph_path):
    """Check the number of edges in a saved graph."""
    if os.path.exists(graph_path):
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        edges = np.sum(graph.adjacency > 0) // 2
        density = edges / (graph.n * (graph.n - 1) // 2)
        return graph.n, edges, density
    return None, None, None

def monitor_training():
    """Monitor the retraining process."""
    
    print("="*70)
    print("RETRAINING MONITOR")
    print("="*70)
    print()
    print("Monitoring for new retrained models...")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    experiments_dir = Path("experiments")
    seen_dirs = set()
    
    while True:
        try:
            # Look for new experiment directories
            fixed_dirs = sorted([d for d in experiments_dir.glob("fixed_run_*") if d.is_dir()])
            small_test_dirs = sorted([d for d in experiments_dir.glob("small_test_*") if d.is_dir()])
            
            all_dirs = fixed_dirs + small_test_dirs
            
            for exp_dir in all_dirs:
                if exp_dir not in seen_dirs:
                    seen_dirs.add(exp_dir)
                    
                    print(f"\n{'='*70}")
                    print(f"NEW EXPERIMENT: {exp_dir.name}")
                    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("-"*70)
                    
                    # Check graph properties
                    graph_path = exp_dir / "data" / "graph.pkl"
                    nodes, edges, density = check_graph_edges(graph_path)
                    
                    if nodes:
                        print(f"Graph Statistics:")
                        print(f"  Nodes: {nodes}")
                        print(f"  Edges: {edges:,}")
                        print(f"  Density: {density:.4f}")
                    
                    # Check for config
                    config_path = exp_dir / "config.json"
                    if config_path.exists():
                        import json
                        with open(config_path) as f:
                            config = json.load(f)
                        print(f"\nTraining Configuration:")
                        print(f"  Walks: {config.get('num_walks', 'N/A'):,}")
                        print(f"  Epochs: {config.get('epochs', 'N/A')}")
                        print(f"  Batch size: {config.get('batch_size', 'N/A')}")
                        print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
                        print(f"  Context window: {config.get('context_window_size', 'N/A')}")
                        print(f"  Walk length: {config.get('min_walk_length', 'N/A')}-{config.get('max_walk_length', 'N/A')}")
                    
                    # Check if model exists
                    model_path = exp_dir / "model.pth"
                    if model_path.exists():
                        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
                        print(f"\n✅ Model saved: {model_size:.2f} MB")
                    
                    # Check for evaluation results
                    eval_path = exp_dir / "evaluation_results.json"
                    if eval_path.exists():
                        import json
                        with open(eval_path) as f:
                            eval_results = json.load(f)
                        print(f"\nEvaluation Results:")
                        if 'error_summary' in eval_results:
                            summary = eval_results['error_summary']
                            print(f"  Broken graph rate: {summary.get('broken_graph_rate', 0):.2%}")
                            print(f"  Repeater error: {summary.get('repeater_error_rate', 0):.2%}")
                            print(f"  Ascender error: {summary.get('ascender_error_rate', 0):.2%}")
                            print(f"  Even error: {summary.get('even_error_rate', 0):.2%}")
                    
                    print("="*70)
            
            # Sleep before checking again
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"\nError during monitoring: {e}")
            time.sleep(5)

def main():
    """Main monitoring function."""
    monitor_training()

if __name__ == "__main__":
    main()