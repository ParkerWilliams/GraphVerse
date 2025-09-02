#!/usr/bin/env python3
"""
Quick test of the retraining setup before running the full training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from retrain_fixed_model import create_dense_graph, setup_rules, generate_training_data

def test_setup():
    """Test that we can generate training data successfully."""
    print("Testing retraining setup...")
    
    # Small test configuration
    test_config = {
        'n': 100,  # Small graph for testing
        'num_walks': 100,  # Just 100 walks for test
        'context_window_size': 16,
        'min_walk_length': 32,
        'max_walk_length': 32,
        'num_ascenders': 5,
        'num_evens': 10,
        'num_repeaters': 10,
        'repeater_k_values': [8, 14, 18, 24],
        'edge_density': 0.4,
        'seed': 42
    }
    
    # Create graph
    print("\n1. Creating test graph...")
    graph = create_dense_graph(
        n=test_config['n'],
        target_density=test_config['edge_density'],
        seed=test_config['seed']
    )
    
    # Set up rules
    print("\n2. Setting up rules...")
    rules, rule_nodes = setup_rules(graph, test_config)
    
    # Test walk generation
    print("\n3. Testing walk generation...")
    from graphverse.graph.walk import generate_valid_walk
    
    successful_walks = 0
    failed_walks = 0
    
    for i in range(10):
        walk = generate_valid_walk(
            graph, i, 
            test_config['min_walk_length'], 
            test_config['max_walk_length'], 
            rules,
            max_attempts=10,
            verbose=False
        )
        
        if walk and len(walk) >= test_config['min_walk_length']:
            successful_walks += 1
            # Verify all edges exist
            all_edges_exist = True
            for j in range(1, len(walk)):
                if not graph.has_edge(walk[j-1], walk[j]):
                    all_edges_exist = False
                    print(f"  Walk {i}: Missing edge {walk[j-1]} -> {walk[j]}")
                    break
            
            if all_edges_exist:
                print(f"  Walk {i}: Success, length={len(walk)}")
        else:
            failed_walks += 1
            print(f"  Walk {i}: Failed to generate")
    
    print(f"\n  Results: {successful_walks}/10 successful walks")
    
    if successful_walks >= 8:
        print("\n✅ Setup test PASSED: Walk generation working well")
        return True
    else:
        print("\n❌ Setup test FAILED: Walk generation struggling")
        print("   May need to increase graph density or adjust parameters")
        return False

if __name__ == "__main__":
    success = test_setup()
    exit(0 if success else 1)