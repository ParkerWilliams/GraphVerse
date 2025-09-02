#!/usr/bin/env python3
"""
Test the walk visualization tools.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
from graphverse.graph.walk import generate_valid_walk
from graphverse.visualization.walk_visualizer import (
    visualize_walk_with_text,
    visualize_rule_breaking_walks,
    create_kl_divergence_heatmap,
    analyze_kl_patterns,
    create_aggregated_violation_analysis,
    create_violation_analysis_by_type,
    create_comprehensive_violation_analysis
)


def test_single_walk_visualization():
    """Test visualization of a single walk with all rule types."""
    print("🎨 Testing Single Walk Visualization")
    print("=" * 50)
    
    # Setup graph
    n = 20
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    # Define rules
    repeaters = {5: 2, 12: 3}  # k=2 at node 5, k=3 at node 12
    ascenders = {8, 15}
    evens = {4, 10, 16}
    
    rules = [
        RepeaterRule(repeaters),
        AscenderRule(ascenders),
        EvenRule(evens)
    ]
    
    print(f"Graph: {n} nodes")
    print(f"Repeaters: {repeaters}")
    print(f"Ascenders: {ascenders}")
    print(f"Evens: {evens}")
    
    # Generate a walk
    walk = generate_valid_walk(
        graph=graph,
        start_vertex=0,
        min_length=15,
        max_length=20,
        rules=rules,
        verbose=False
    )
    
    if walk:
        print(f"\nGenerated walk: {walk}")
        
        # Simulate KL divergences (would come from model in practice)
        kl_divergences = generate_mock_kl_divergences(walk, rules)
        
        # Create visualization
        fig = visualize_walk_with_text(
            walk=walk,
            graph=graph,
            rules=rules,
            title="Walk with Repeater, Ascender, and Even Rules",
            kl_divergences=kl_divergences
        )
        
        plt.savefig('walk_visualization_single.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: walk_visualization_single.png")
    else:
        print("❌ Failed to generate walk")


def test_rule_breaking_visualization():
    """Test visualization of rule-breaking walks with KL divergence grouping."""
    print("\n🚫 Testing Rule-Breaking Walk Visualization")
    print("=" * 50)
    
    # Setup graph
    n = 15
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    # Define rules
    rules = [
        RepeaterRule({3: 2, 7: 3}),
        AscenderRule({5, 10}),
        EvenRule({2, 8})
    ]
    
    # Create synthetic rule-breaking walks with different KL patterns
    walks_by_category = {
        'peaked_correct': [],
        'peaked_default': [],
        'uniform': []
    }
    
    # Generate examples for each category
    for category in walks_by_category:
        for i in range(3):
            # Create a walk with violations
            walk, violations = create_rule_breaking_walk(n, rules)
            
            # Generate appropriate KL pattern
            if category == 'peaked_correct':
                kl_divs = generate_peaked_kl(len(walk), peak_value=5.0, peak_pos=0.2)
            elif category == 'peaked_default':
                kl_divs = generate_peaked_kl(len(walk), peak_value=4.0, peak_pos=0.7)
            else:  # uniform
                kl_divs = generate_uniform_kl(len(walk), mean=0.3)
            
            walks_by_category[category].append((walk, kl_divs, violations))
    
    # Create visualization
    fig = visualize_rule_breaking_walks(
        walks_by_category=walks_by_category,
        graph=graph,
        rules=rules
    )
    
    plt.savefig('walk_visualization_violations.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: walk_visualization_violations.png")


def test_violation_detection():
    """Test that violations are correctly detected."""
    print("\n✅ Testing Violation Detection")
    print("=" * 50)
    
    # Setup simple graph
    n = 10
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    # Define rules
    rules = [
        EvenRule({4}),  # Node 4 is even, next must be odd
        AscenderRule({2}),  # Node 2 is ascender, next must be > 2
    ]
    
    # Create walks with known violations
    # Walk 1: violates even rule (4 -> 6, where 6 is even)
    walk1 = [0, 4, 6, 7]  # Violation at position 2
    
    # Walk 2: violates ascender rule (2 -> 1, where 1 < 2)
    walk2 = [0, 2, 1, 5]  # Violation at position 2
    
    # Walk 3: correct walk
    walk3 = [0, 4, 5, 2, 7]  # No violations
    
    print("Test walks:")
    print(f"  Walk 1 (even violation): {walk1}")
    print(f"  Walk 2 (ascender violation): {walk2}")
    print(f"  Walk 3 (no violations): {walk3}")
    
    # Generate mock KL divergences
    kl1 = [0.5, 2.0, 4.0, 1.0]  # Peak at violation
    kl2 = [0.3, 1.5, 3.5, 0.8]  # Peak at violation
    kl3 = [0.4, 0.6, 0.5, 0.7, 0.5]  # Low throughout
    
    # Check violations are detected
    from graphverse.visualization.walk_visualizer import _find_all_violations
    
    violations1 = _find_all_violations(walk1, rules)
    violations2 = _find_all_violations(walk2, rules)
    violations3 = _find_all_violations(walk3, rules)
    
    print("\nDetected violations:")
    print(f"  Walk 1: {violations1}")
    print(f"  Walk 2: {violations2}")
    print(f"  Walk 3: {violations3}")
    
    # Visualize walk with violations
    fig = visualize_walk_with_text(
        walk=walk1,
        graph=graph,
        rules=rules,
        title="Walk with Even Rule Violation",
        rule_violations=violations1,
        kl_divergences=kl1
    )
    
    plt.savefig('walk_violation_detection.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: walk_violation_detection.png")
    

def test_kl_heatmap():
    """Test improved KL divergence visualization."""
    print("\n🔥 Testing Improved KL Divergence Visualization")
    print("=" * 50)
    
    n = 15
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                graph.add_edge(i, j, weight=1.0)
    
    rules = [
        RepeaterRule({4: 2}),
        AscenderRule({7, 11})
    ]
    
    walks = []
    kl_divergences = []
    
    # Generate multiple walks
    for i in range(20):
        walk = generate_valid_walk(
            graph=graph,
            start_vertex=i % n,
            min_length=10,
            max_length=25,
            rules=rules,
            verbose=False
        )
        
        if walk:
            walks.append(walk)
            # Generate varied KL patterns
            if i % 3 == 0:
                kl = generate_peaked_kl(len(walk), peak_value=4.0, peak_pos=0.3)
            elif i % 3 == 1:
                kl = generate_uniform_kl(len(walk), mean=0.5)
            else:
                kl = generate_peaked_kl(len(walk), peak_value=3.0, peak_pos=0.8)
            kl_divergences.append(kl)
    
    if walks:
        print(f"Generated {len(walks)} walks")
        
        # Generate violation positions for each walk
        from graphverse.visualization.walk_visualizer import _find_all_violations
        violation_positions = [_find_all_violations(walk, rules) for walk in walks]
        
        # Create improved visualization
        fig = create_kl_divergence_heatmap(
            walks=walks,
            kl_divergences=kl_divergences,
            max_length=30,
            violation_positions=violation_positions
        )
        
        plt.savefig('walk_kl_heatmap.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: walk_kl_heatmap.png")
        
        # Analyze patterns
        print("\nKL Pattern Analysis:")
        for i, kl in enumerate(kl_divergences[:5]):
            pattern = analyze_kl_patterns(kl)
            print(f"  Walk {i}: {pattern}")


def generate_mock_kl_divergences(walk, rules):
    """Generate mock KL divergences for visualization testing."""
    kl_divs = []
    
    # Identify rule nodes
    rule_nodes = set()
    for rule in rules:
        if hasattr(rule, 'member_nodes'):
            rule_nodes.update(rule.member_nodes)
        elif hasattr(rule, 'members_nodes_dict'):
            rule_nodes.update(rule.members_nodes_dict.keys())
    
    for i, node in enumerate(walk):
        if node in rule_nodes:
            # Peak at rule nodes
            kl = np.random.uniform(2.0, 5.0)
        else:
            # Lower elsewhere
            kl = np.random.uniform(0.1, 1.0)
        
        # Add some trend
        kl += i * 0.05
        kl_divs.append(kl)
    
    return kl_divs


def create_rule_breaking_walk(n, rules):
    """Create a synthetic rule-breaking walk for testing."""
    walk_length = np.random.randint(8, 15)  # Shorter for better display
    walk = []
    violations = {}
    
    # Start with random node
    walk.append(np.random.randint(0, n))
    
    # Build walk with some violations
    for i in range(1, walk_length):
        next_node = np.random.randint(0, n)
        
        # Occasionally force a violation
        if np.random.random() < 0.4 and i > 2:
            violation_type = np.random.choice(['ascender', 'even', 'repeater', 'graph'])
            
            if violation_type == 'ascender':
                # Force descending violation after ascender
                for rule in rules:
                    if hasattr(rule, 'is_ascender_rule') and rule.is_ascender_rule:
                        for asc in rule.member_nodes:
                            if asc == walk[i-1]:
                                next_node = max(0, asc - 1)  # Force descending
                                violations[i] = 'ascender'
                                break
            elif violation_type == 'even':
                # Force even after even rule node
                for rule in rules:
                    if hasattr(rule, 'is_even_rule') and rule.is_even_rule:
                        if walk[i-1] in rule.member_nodes:
                            next_node = 2 * (next_node // 2)  # Make even
                            violations[i] = 'even'
                            break
            elif violation_type == 'repeater':
                # Violate repeater spacing
                violations[i] = 'repeater'
            elif violation_type == 'graph':
                # Simulate invalid edge (not actually invalid in complete graph)
                violations[i] = 'graph'
        
        walk.append(next_node)
    
    return walk, violations


def generate_peaked_kl(length, peak_value=5.0, peak_pos=0.5):
    """Generate peaked KL divergence pattern."""
    kl = []
    peak_idx = int(length * peak_pos)
    
    for i in range(length):
        dist_from_peak = abs(i - peak_idx)
        value = peak_value * np.exp(-0.5 * (dist_from_peak / 3) ** 2)
        value += np.random.uniform(-0.1, 0.1)
        kl.append(max(0, value))
    
    return kl


def generate_uniform_kl(length, mean=0.5):
    """Generate uniform/low KL divergence pattern."""
    return [max(0, np.random.normal(mean, 0.1)) for _ in range(length)]


def generate_wasserstein_distances(kl_divergences):
    """Generate mock Wasserstein distances correlated with KL divergence."""
    # Wasserstein typically correlates with KL but with different scale
    wass_dists = []
    for kl in kl_divergences:
        # Add some noise and scaling
        wass = kl * 0.3 + np.random.normal(0, 0.05)
        wass_dists.append(max(0, wass))
    return wass_dists


def generate_model_distribution(graph, current_vertex, next_vertex, rules, add_noise=True):
    """Generate mock model distribution at a vertex."""
    n = graph.n
    dist = np.zeros(n)
    
    # Start with edge weights as base
    for neighbor in range(n):
        if neighbor != current_vertex:
            # Access adjacency matrix directly
            weight = graph.adjacency[current_vertex, neighbor]
            dist[neighbor] = weight if weight > 0 else 0.1
    
    # Add noise to simulate model uncertainty
    if add_noise:
        noise = np.random.normal(0, 0.1, n)
        dist = dist + np.abs(noise)
    
    # Normalize
    if dist.sum() > 0:
        dist = dist / dist.sum()
    else:
        dist = np.ones(n) / n
    
    return dist


def test_aggregated_violation_analysis():
    """Test aggregated KL divergence analysis before violations."""
    print("\n📊 Testing Aggregated Violation Analysis")
    print("=" * 50)
    
    # Setup graph with varied edge weights
    n = 20
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                # Varied weights to create non-uniform distributions
                weight = 1.0 + np.random.exponential(0.5)
                graph.add_edge(i, j, weight=weight)
    
    # Define rules
    rules = [
        RepeaterRule({5: 2, 10: 3}),
        AscenderRule({3, 7, 12}),
        EvenRule({4, 8, 14})
    ]
    
    # Generate many walks with violations
    walks_with_violations = []
    model_distributions = []
    
    for i in range(50):  # Generate 50 walks
        # Create walk with violations
        walk, violations = create_rule_breaking_walk(n, rules)
        
        if violations:  # Only include walks with violations
            # Generate model distributions and KL divergences
            kl_divs = []
            model_dists = []
            
            for step in range(len(walk)):
                current_vertex = walk[step]
                next_vertex = walk[step + 1] if step < len(walk) - 1 else walk[step]
                
                # Generate model distribution at this step
                model_dist = generate_model_distribution(graph, current_vertex, next_vertex, rules)
                model_dists.append(model_dist)
                
                # Find distance to nearest future violation
                dist_to_violation = float('inf')
                for viol_pos in violations.keys():
                    if viol_pos > step:
                        dist_to_violation = min(dist_to_violation, viol_pos - step)
                
                # KL increases as we approach violation
                if dist_to_violation < float('inf'):
                    base_kl = 0.5 + (10 - min(dist_to_violation, 10)) * 0.3
                    kl = base_kl + np.random.normal(0, 0.2)
                else:
                    kl = np.random.uniform(0.3, 1.0)
                
                kl_divs.append(max(0, kl))
            
            # Generate Wasserstein distances
            wass_dists = generate_wasserstein_distances(kl_divs)
            
            walks_with_violations.append((walk, kl_divs, violations, wass_dists))
            model_distributions.append(model_dists)
    
    print(f"Generated {len(walks_with_violations)} walks with violations")
    
    # Analyze violation types
    violation_counts = {'repeater': 0, 'ascender': 0, 'even': 0, 'graph': 0}
    total_violations = 0
    for _, _, violations, _ in walks_with_violations:
        for vtype in violations.values():
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
            total_violations += 1
    
    print(f"\nViolation breakdown ({total_violations} total):")
    for vtype, count in violation_counts.items():
        if count > 0:
            print(f"  {vtype}: {count} ({100*count/total_violations:.1f}%)")
    
    # Create aggregated analysis with model distributions
    fig = create_aggregated_violation_analysis(
        walks_with_violations=walks_with_violations,
        max_lookback=15,
        model_distributions=model_distributions,
        graph=graph
    )
    
    plt.savefig('walk_aggregated_violations.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: walk_aggregated_violations.png")
    
    # Generate comprehensive plots for all metrics and violation types
    print("\n📊 Generating comprehensive plots for all metrics...")
    print("📍 Only analyzing steps AFTER rule vertex encounters")
    figures = create_comprehensive_violation_analysis(
        walks_with_violations=walks_with_violations,
        max_lookback=15,
        model_distributions=model_distributions,
        graph=graph,
        output_dir='.',
        rules=rules,
        only_after_rule_encounter=True
    )
    
    # Summary of generated plots
    print(f"\n✅ Generated {len(figures) if figures else 0} plots total:")
    print("  - 7 metrics × 4 violation types = 28 individual plots")
    print("  - 7 metrics × 1 aggregated = 7 aggregated plots")
    print("  - Total: 35 plots")


def main():
    print("🧪 WALK VISUALIZATION TEST SUITE")
    print("=" * 50)
    
    test_single_walk_visualization()
    test_violation_detection()
    test_rule_breaking_visualization()
    test_kl_heatmap()
    test_aggregated_violation_analysis()
    
    print("\n" + "=" * 50)
    print("✅ All visualization tests complete!")
    print("Check generated PNG files for results.")


if __name__ == "__main__":
    main()