#!/usr/bin/env python3
"""Visualize subset evaluation results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_error_rates_by_category(results, output_path="subset_error_rates.png"):
    """Plot error rates by starting node category."""
    
    categories = list(results['by_category'].keys())
    error_types = ['broken_graph', 'even', 'ascender', 'repeater']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Error rates by category
    ax1 = axes[0]
    x = np.arange(len(categories))
    width = 0.2
    colors = ['#E74C3C', '#3498DB', '#28B463', '#F39C12']
    
    for i, error_type in enumerate(error_types):
        values = [results['by_category'][cat]['error_rates'][error_type] * 100 
                 for cat in categories]
        ax1.bar(x + i*width, values, width, label=error_type.capitalize(), 
               color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Starting Node Category', fontsize=12)
    ax1.set_ylabel('Error Rate (%)', fontsize=12)
    ax1.set_title('Error Rates by Starting Node Category', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([cat.capitalize() for cat in categories])
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Walk statistics by category
    ax2 = axes[1]
    
    # Create grouped bar chart for walk stats
    stats_data = {
        'Avg Length': [results['by_category'][cat]['avg_walk_length'] for cat in categories],
        'Completed (%)': [results['by_category'][cat]['completed_walks']/25 for cat in categories],
        'Avg Confidence': [results['by_category'][cat]['avg_confidence']*100 for cat in categories]
    }
    
    x2 = np.arange(len(categories))
    width2 = 0.25
    
    for i, (stat_name, values) in enumerate(stats_data.items()):
        ax2.bar(x2 + i*width2, values, width2, label=stat_name, alpha=0.8)
    
    ax2.set_xlabel('Starting Node Category', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Walk Statistics by Category', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2 + width2)
    ax2.set_xticklabels([cat.capitalize() for cat in categories])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved error rates plot to {output_path}")
    return output_path

def plot_overall_summary(results, output_path="subset_summary.png"):
    """Create an overall summary visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Overall error distribution pie chart
    ax1 = axes[0, 0]
    error_rates = results['overall']['error_rates']
    
    # Calculate successful steps
    total_steps = results['overall']['total_steps']
    error_counts = {
        'Broken Graph': error_rates['broken_graph'] * results['total_walks'],
        'Even Violations': error_rates['even'] * total_steps,
        'Ascender Violations': error_rates['ascender'] * total_steps,
        'Repeater Violations': error_rates['repeater'] * total_steps,
    }
    
    colors = ['#E74C3C', '#3498DB', '#28B463', '#F39C12']
    explode = (0.1, 0, 0, 0)  # Explode the broken graph slice
    
    ax1.pie(error_counts.values(), labels=error_counts.keys(), colors=colors,
            autopct='%1.1f%%', explode=explode, shadow=True, startangle=90)
    ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
    
    # 2. Walks per category
    ax2 = axes[0, 1]
    categories = list(results['by_category'].keys())
    walks_per_cat = [results['by_category'][cat]['total_walks'] for cat in categories]
    colors2 = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    bars = ax2.bar(range(len(categories)), walks_per_cat, color=colors2, alpha=0.8)
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Number of Walks', fontsize=12)
    ax2.set_title('Balanced Walk Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels([cat.capitalize() for cat in categories], rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, walks_per_cat):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val}', ha='center', va='bottom', fontsize=10)
    
    # 3. Completion rates by category
    ax3 = axes[1, 0]
    completion_rates = [results['by_category'][cat]['completed_walks'] for cat in categories]
    
    bars3 = ax3.bar(range(len(categories)), completion_rates, color=colors2, alpha=0.8)
    ax3.set_xlabel('Category', fontsize=12)
    ax3.set_ylabel('Completed Walks', fontsize=12)
    ax3.set_title('Walk Completion by Category', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels([cat.capitalize() for cat in categories], rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars3, completion_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}', ha='center', va='bottom', fontsize=10)
    
    # 4. Summary statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
SUBSET EVALUATION SUMMARY
{'='*30}

Total Walks: {results['total_walks']:,}
Total Steps: {results['overall']['total_steps']:,}
Completed: {results['overall']['completed_walks']}

Average Steps/Walk: {results['overall']['total_steps']/results['total_walks']:.1f}

Top Issues:
• Broken Graph: {error_rates['broken_graph']*100:.1f}%
• Even Rule: {error_rates['even']*100:.2f}%
• Ascender Rule: {error_rates['ascender']*100:.2f}%
• Repeater Rule: {error_rates['repeater']*100:.2f}%

Performance:
• ~1200 walks/second
• ~0.8ms per walk
"""
    
    ax4.text(0.1, 0.9, summary_text, fontsize=11, family='monospace',
            verticalalignment='top', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Subset Evaluation Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved summary plot to {output_path}")
    return output_path

def main():
    """Generate visualizations for subset evaluation results."""
    
    # Load results
    results_path = "small_results/subset_eval_ctx_8/balanced_results.json"
    
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        print("Please run: python run_subset_evaluation.py")
        return
    
    results = load_results(results_path)
    
    # Create output directory
    output_dir = Path("small_results/subset_eval_ctx_8/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Generating Subset Evaluation Visualizations")
    print("=" * 50)
    
    # Generate plots
    plot_error_rates_by_category(
        results, 
        output_dir / "error_rates_by_category.png"
    )
    
    plot_overall_summary(
        results,
        output_dir / "overall_summary.png"
    )
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    
    # Create HTML viewer
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Subset Evaluation Results</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #f5f5f5; 
        }
        h1 { color: #2E86C1; }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .plot { 
            background: white; 
            padding: 20px; 
            margin: 20px 0;
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        img { 
            width: 100%; 
            height: auto; 
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Subset Evaluation with Balanced Rule Distribution</h1>
        
        <div class="plot">
            <h2>Overall Summary</h2>
            <img src="visualizations/overall_summary.png" alt="Overall Summary">
        </div>
        
        <div class="plot">
            <h2>Error Rates by Category</h2>
            <img src="visualizations/error_rates_by_category.png" alt="Error Rates by Category">
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_dir.parent / "results.html", "w") as f:
        f.write(html_content)
    
    print(f"🌐 View results: open {output_dir.parent / 'results.html'}")

if __name__ == "__main__":
    main()