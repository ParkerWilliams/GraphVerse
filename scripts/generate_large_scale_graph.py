#!/usr/bin/env python3
"""
Large-Scale Graph Generation Script

Generates a 10K vertex graph optimized for large-scale repeater context boundary analysis.
"""

import os
import sys
import time
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.large_scale_config import LARGE_SCALE_CONFIG
from configs.medium_scale_config import MEDIUM_SCALE_CONFIG
from graphverse.graph.graph_generation import generate_random_graph
from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
import numpy as np


def create_large_scale_rules(config):
    """
    Create rule objects for large-scale experiment.
    
    Args:
        config: Large scale configuration dictionary
        
    Returns:
        List of rule objects
    """
    n = config["n"]
    rule_percentages = config["rule_percentages"]
    
    # Calculate number of nodes for each rule type
    num_ascenders = int(n * rule_percentages["ascenders"] / 100)
    num_evens = int(n * rule_percentages["evens"] / 100) 
    num_repeaters = int(n * rule_percentages["repeaters"] / 100)
    
    print(f"Rule distribution for {n:,} vertices:")
    print(f"  Ascender nodes: {num_ascenders:,} ({rule_percentages['ascenders']}%)")
    print(f"  Even rule nodes: {num_evens:,} ({rule_percentages['evens']}%)")
    print(f"  Repeater nodes: {num_repeaters:,} ({rule_percentages['repeaters']}%)")
    print(f"  Regular nodes: {n - num_ascenders - num_evens - num_repeaters:,}")
    
    # Create rules with random node assignment
    ascender_nodes = np.random.choice(n, size=num_ascenders, replace=False)
    
    # Exclude ascender nodes from even rule selection
    remaining_nodes = np.setdiff1d(np.arange(n), ascender_nodes)
    even_nodes = np.random.choice(remaining_nodes, size=num_evens, replace=False)
    
    # Exclude both ascender and even nodes from repeater selection
    remaining_nodes = np.setdiff1d(remaining_nodes, even_nodes)
    repeater_nodes = np.random.choice(remaining_nodes, size=num_repeaters, replace=False)
    
    # Create rule objects
    ascender_rule = AscenderRule(ascender_nodes.tolist())
    even_rule = EvenRule(even_nodes.tolist())
    
    # For repeater rule, assign k values that span context boundaries
    repeater_k_values = config["repeater_k_values"]
    repeater_dict = {}
    
    # Distribute repeater nodes across different k values
    nodes_per_k = max(1, num_repeaters // len(repeater_k_values))
    
    for i, k in enumerate(repeater_k_values):
        start_idx = i * nodes_per_k
        end_idx = min((i + 1) * nodes_per_k, num_repeaters)
        
        if start_idx < num_repeaters:
            nodes_for_k = repeater_nodes[start_idx:end_idx]
            for node in nodes_for_k:
                repeater_dict[node] = k
    
    # Assign remaining nodes to random k values
    assigned_nodes = len(repeater_dict)
    if assigned_nodes < num_repeaters:
        remaining_repeater_nodes = repeater_nodes[assigned_nodes:]
        for node in remaining_repeater_nodes:
            k = np.random.choice(repeater_k_values)
            repeater_dict[node] = k
    
    repeater_rule = RepeaterRule(repeater_dict)
    
    print(f"\nRepeater k-value distribution:")
    k_counts = {}
    for k in repeater_dict.values():
        k_counts[k] = k_counts.get(k, 0) + 1
    
    for k in sorted(k_counts.keys()):
        print(f"  k={k}: {k_counts[k]} nodes")
    
    return [ascender_rule, even_rule, repeater_rule]


def generate_and_save_graph(config, output_path="large_scale_graph", verbose=True):
    """
    Generate and save the large-scale graph.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save graph (without extension)
        verbose: Whether to print detailed progress
        
    Returns:
        Generated graph object
    """
    if verbose:
        print("=" * 80)
        print("LARGE-SCALE GRAPH GENERATION")
        print("=" * 80)
        print(f"Target: {config['n']:,} vertices")
        print(f"Edge density: {config['min_edge_density']:.1%}")
        print(f"Edge concentration: {config['edge_concentration']}")
    
    # Create rules
    start_time = time.time()
    rules = create_large_scale_rules(config)
    
    if verbose:
        print(f"\nRule creation completed in {time.time() - start_time:.1f} seconds")
        print("\nGenerating graph structure...")
    
    # Generate graph
    graph_start = time.time()
    graph = generate_random_graph(
        n=config["n"],
        rules=rules,
        min_edge_density=config["min_edge_density"],
        edge_concentration=config["edge_concentration"],
        verbose=verbose,
        save_walks=False  # Don't generate walks during graph creation
    )
    
    graph_time = time.time() - graph_start
    
    if verbose:
        print(f"\nGraph generation completed in {graph_time:.1f} seconds")
        print("Saving graph...")
    
    # Save graph
    save_start = time.time()
    graph.save_graph(output_path)
    save_time = time.time() - save_start
    
    if verbose:
        print(f"Graph saved in {save_time:.1f} seconds")
        print(f"Files created:")
        print(f"  {output_path}.npy (adjacency matrix)")
        print(f"  {output_path}_attrs.json (node attributes)")
    
    # Save rule information separately for easy loading
    rule_info = {
        "ascender_nodes": rules[0].member_nodes,
        "even_nodes": rules[1].member_nodes,
        "repeater_nodes_dict": rules[2].members_nodes_dict,
        "repeater_k_values": config["repeater_k_values"],
        "rule_percentages": config["rule_percentages"],
        "total_nodes": config["n"]
    }
    
    import json
    with open(f"{output_path}_rules.json", "w") as f:
        json.dump(rule_info, f, indent=2)
    
    if verbose:
        print(f"  {output_path}_rules.json (rule assignments)")
        
        total_time = time.time() - start_time
        print(f"\nTotal generation time: {total_time:.1f} seconds")
        print("Graph generation complete!")
    
    return graph, rules


def validate_graph(graph, rules, config, verbose=True):
    """
    Validate the generated graph meets requirements.
    
    Args:
        graph: Generated graph object
        rules: List of rule objects
        config: Configuration dictionary
        verbose: Whether to print validation results
        
    Returns:
        True if validation passes, False otherwise
    """
    if verbose:
        print("\n" + "=" * 40)
        print("GRAPH VALIDATION")
        print("=" * 40)
    
    validation_passed = True
    
    # Check graph size
    if graph.n != config["n"]:
        print(f"❌ Graph size mismatch: expected {config['n']}, got {graph.n}")
        validation_passed = False
    elif verbose:
        print(f"✅ Graph size: {graph.n:,} vertices")
    
    # Check edge density
    from graphverse.graph.graph_generation import calculate_edge_density
    actual_density = calculate_edge_density(graph, verbose=False)
    
    if actual_density < config["min_edge_density"]:
        print(f"❌ Edge density too low: {actual_density:.3f} < {config['min_edge_density']:.3f}")
        validation_passed = False
    elif verbose:
        print(f"✅ Edge density: {actual_density:.3f} (target: {config['min_edge_density']:.3f})")
    
    # Check rule assignments
    ascender_rule, even_rule, repeater_rule = rules
    
    expected_ascenders = int(config["n"] * config["rule_percentages"]["ascenders"] / 100)
    expected_evens = int(config["n"] * config["rule_percentages"]["evens"] / 100)
    expected_repeaters = int(config["n"] * config["rule_percentages"]["repeaters"] / 100)
    
    if len(ascender_rule.member_nodes) != expected_ascenders:
        print(f"❌ Ascender count mismatch: expected {expected_ascenders}, got {len(ascender_rule.member_nodes)}")
        validation_passed = False
    elif verbose:
        print(f"✅ Ascender nodes: {len(ascender_rule.member_nodes)}")
    
    if len(even_rule.member_nodes) != expected_evens:
        print(f"❌ Even node count mismatch: expected {expected_evens}, got {len(even_rule.member_nodes)}")
        validation_passed = False
    elif verbose:
        print(f"✅ Even rule nodes: {len(even_rule.member_nodes)}")
    
    if len(repeater_rule.member_nodes) != expected_repeaters:
        print(f"❌ Repeater count mismatch: expected {expected_repeaters}, got {len(repeater_rule.member_nodes)}")
        validation_passed = False
    elif verbose:
        print(f"✅ Repeater nodes: {len(repeater_rule.member_nodes)}")
    
    # Check for node overlap
    all_rule_nodes = set(ascender_rule.member_nodes) | set(even_rule.member_nodes) | set(repeater_rule.member_nodes)
    total_rule_nodes = len(ascender_rule.member_nodes) + len(even_rule.member_nodes) + len(repeater_rule.member_nodes)
    
    if len(all_rule_nodes) != total_rule_nodes:
        print(f"❌ Node overlap detected: {total_rule_nodes - len(all_rule_nodes)} overlapping assignments")
        validation_passed = False
    elif verbose:
        print(f"✅ No node overlap: {len(all_rule_nodes)} unique rule nodes")
    
    # Check connectivity
    if hasattr(graph, 'is_connected'):
        if not graph.is_connected():
            print("⚠️ Warning: Graph may not be fully connected")
        elif verbose:
            print("✅ Graph connectivity validated")
    
    if verbose:
        if validation_passed:
            print("\n🎉 Graph validation PASSED")
        else:
            print("\n❌ Graph validation FAILED")
    
    return validation_passed


def main():
    """Main graph generation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate large-scale graph for context boundary analysis")
    parser.add_argument("--output", "-o", default="large_scale_graph", 
                       help="Output path for graph files (default: large_scale_graph)")
    parser.add_argument("--validate", action="store_true", 
                       help="Run validation after generation")
    parser.add_argument("--medium-scale", action="store_true",
                       help="Use medium-scale configuration (1K vertices)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Choose configuration based on scale
    config = MEDIUM_SCALE_CONFIG if args.medium_scale else LARGE_SCALE_CONFIG
    scale_name = "medium-scale" if args.medium_scale else "large-scale"
    
    if verbose:
        print(f"Using {scale_name} configuration:")
        print(f"  Vertices: {config['n']:,}")
        print(f"  Walks per context: {config['num_walks']:,}")
        print(f"  Context windows: {config['context_windows']}")
    
    try:
        # Generate graph
        graph, rules = generate_and_save_graph(
            config=config,
            output_path=args.output,
            verbose=verbose
        )
        
        # Validate if requested
        if args.validate:
            validation_passed = validate_graph(graph, rules, config, verbose)
            if not validation_passed:
                print("❌ Validation failed!")
                sys.exit(1)
        
        if verbose:
            print(f"\n✅ {scale_name.capitalize()} graph successfully generated: {args.output}")
            print(f"Ready for {scale_name} context boundary experiments!")
        
    except Exception as e:
        print(f"❌ Error generating graph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()