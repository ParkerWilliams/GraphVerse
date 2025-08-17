#!/usr/bin/env python3
"""
Test script for the repeater density study functionality.
This script tests the core components without running a full study.
"""

import sys
import traceback

def test_imports():
    """Test all necessary imports."""
    print("Testing imports...")
    
    try:
        from graphverse.graph.base import SimpleGraph
        from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
        from graphverse.data.preparation import prepare_density_controlled_training_data
        from graphverse.utils.multi_experiment_runner import run_repeater_density_study
        from graphverse.llm.evaluation_vis import plot_density_vs_accuracy
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_graph_creation():
    """Test creating a simple graph with rules."""
    print("\nTesting basic graph and rule creation...")
    
    try:
        from graphverse.graph.base import SimpleGraph
        from graphverse.graph.rules import RepeaterRule, AscenderRule
        
        # Create a small graph
        graph = SimpleGraph(10)
        
        # Add some edges
        for i in range(9):
            graph.add_edge(i, i + 1)
        
        # Create some rules
        repeater_rule = RepeaterRule({5: 3, 7: 2})  # Node 5 repeats every 3, node 7 every 2
        ascender_rule = AscenderRule({8})  # Node 8 is an ascender
        
        rules = [repeater_rule, ascender_rule]
        
        print(f"✓ Created graph with {graph.n} nodes")
        print(f"✓ Created {len(rules)} rules")
        print(f"  - Repeater nodes: {repeater_rule.member_nodes}")
        print(f"  - Ascender nodes: {ascender_rule.member_nodes}")
        
        return graph, rules
        
    except Exception as e:
        print(f"✗ Graph creation failed: {e}")
        traceback.print_exc()
        return None, None

def test_density_controlled_data_generation():
    """Test generating training data with controlled density."""
    print("\nTesting density-controlled data generation...")
    
    try:
        graph, rules = test_basic_graph_creation()
        if graph is None:
            return False
            
        from graphverse.data.preparation import prepare_density_controlled_training_data
        
        # Test with some density control
        density_config = {5: 10, 7: 5}  # More examples for nodes 5 and 7
        
        training_data, vocab, density_stats = prepare_density_controlled_training_data(
            graph=graph,
            num_walks=50,
            min_length=3,
            max_length=6,
            rules=rules,
            repeater_densities=density_config,
            verbose=True
        )
        
        print(f"✓ Generated training data: {training_data.shape}")
        print(f"✓ Vocabulary size: {len(vocab)}")
        print(f"✓ Density stats: {density_stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        traceback.print_exc()
        return False

def test_analysis_functions():
    """Test analysis helper functions."""
    print("\nTesting analysis functions...")
    
    try:
        from graphverse.llm.evaluation_vis import collect_repeater_violations_by_length
        from graphverse.utils.multi_experiment_runner import analyze_density_vs_accuracy
        
        # Create mock data for testing
        mock_results = {
            "config1": {
                "density_stats": {"repeater_exposure_counts": {5: 20, 7: 15}},
                "error_summary": {"repeater_error_rate": 0.3},
                "violation_rates_by_k": {3: 0.2, 2: 0.4}
            },
            "config2": {
                "density_stats": {"repeater_exposure_counts": {5: 30, 7: 25}},
                "error_summary": {"repeater_error_rate": 0.2},
                "violation_rates_by_k": {3: 0.1, 2: 0.3}
            }
        }
        
        analysis = analyze_density_vs_accuracy(mock_results)
        print(f"✓ Analysis completed: {len(analysis)} sections")
        print(f"  - Node analysis: {list(analysis['node_specific_analysis'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("GRAPHVERSE DENSITY STUDY TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_density_controlled_data_generation,
        test_analysis_functions
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The density study functionality is ready to use.")
        print("\nNext steps:")
        print("1. Use 'python' to run your scripts")
        print("2. Create your graph and rules")
        print("3. Define density configurations")
        print("4. Run the full density study")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())