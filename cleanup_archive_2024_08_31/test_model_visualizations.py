#!/usr/bin/env python3
"""Test script to run a small experiment and generate all visualizations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from graphverse.graph.base import Graph
from graphverse.graph.rules import RepeaterRule, AscenderRule, EvenRule
from graphverse.llm.model_enhanced import EnhancedWalkTransformer
from graphverse.llm.model_visualization import (
    create_model_architecture_diagram,
    plot_training_dynamics,
    plot_probability_distribution_evolution,
    plot_rule_violation_heatmap,
    plot_perplexity_by_rule_exposure,
    create_comprehensive_evaluation_report
)
import matplotlib.pyplot as plt
import os

def test_visualizations():
    """Generate all model visualizations with test data."""
    
    print("🧪 Testing Model Visualization Suite")
    print("=" * 50)
    
    # Create output directory
    output_dir = "test_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Model Architecture Diagram
    print("\n1️⃣ Generating model architecture diagram...")
    model_config = {
        'hidden_size': 384,
        'num_layers': 4,
        'num_heads': 6,
        'total_params': 9466740
    }
    create_model_architecture_diagram(model_config, f"{output_dir}/model_architecture.png")
    
    # 2. Training Dynamics
    print("\n2️⃣ Generating training dynamics plot...")
    # Simulate training data
    epoch_losses = [3.2, 2.1, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02, 1.01]
    learning_rates = np.concatenate([
        np.linspace(0.0001, 0.001, 100),  # Warmup
        np.linspace(0.001, 0.0001, 900)   # Cosine decay
    ])
    plot_training_dynamics(epoch_losses, learning_rates[:100], f"{output_dir}/training_dynamics.png")
    
    # 3. Probability Distribution Evolution
    print("\n3️⃣ Generating probability distribution evolution...")
    # Simulate probability distributions
    vocab_size = 1000
    model_distributions = []
    for i in range(10):
        # Create increasingly peaked distributions
        dist = np.random.dirichlet(np.ones(vocab_size) * (10 - i))
        model_distributions.append(dist)
    plot_probability_distribution_evolution(
        model_distributions,
        output_path=f"{output_dir}/prob_evolution.png"
    )
    
    # 4. Rule Violation Heatmap
    print("\n4️⃣ Generating rule violation heatmap...")
    # Simulate evaluation results
    evaluation_results = {
        'walks': [
            {'violations': {5: 'repeater', 10: 'ascender', 15: 'even', 20: 'graph'}}
            for _ in range(100)
        ]
    }
    plot_rule_violation_heatmap(
        evaluation_results,
        [],
        f"{output_dir}/rule_violations_heatmap.png"
    )
    
    # 5. Perplexity by Rule Exposure
    print("\n5️⃣ Generating perplexity by rule exposure plot...")
    # Simulate token-level data
    token_level_data = []
    for i in range(1000):
        token_level_data.append({
            'confidence': np.random.beta(2, 5),  # Skewed confidence
            'context_tokens': ['repeater' if np.random.random() > 0.5 else 'normal'] * 5,
            'perplexity': np.random.gamma(2, 2)
        })
    plot_perplexity_by_rule_exposure(
        token_level_data,
        f"{output_dir}/perplexity_by_rules.png"
    )
    
    # 6. Comprehensive Report
    print("\n6️⃣ Generating comprehensive evaluation report...")
    training_metrics = {
        'epoch_losses': epoch_losses,
        'learning_rates': learning_rates[:100]
    }
    
    # Add model distributions to evaluation results
    evaluation_results['model_distributions'] = model_distributions
    
    generated_plots = create_comprehensive_evaluation_report(
        model_config=model_config,
        training_metrics=training_metrics,
        evaluation_results=evaluation_results,
        token_level_data=token_level_data,
        output_dir=output_dir
    )
    
    print("\n" + "=" * 50)
    print(f"✅ Generated {len(os.listdir(output_dir))} visualization files:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.png'):
            print(f"   📊 {file}")
    
    print(f"\n📁 All visualizations saved to: {output_dir}/")
    
    # Create an HTML index to view all plots
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GraphVerse Model Visualizations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #2E86C1; }
            .gallery { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
            .plot { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .plot img { width: 100%; height: auto; }
            .plot h3 { margin: 10px 0; color: #34495E; }
        </style>
    </head>
    <body>
        <h1>🧠 GraphVerse Model Visualizations</h1>
        <div class="gallery">
    """
    
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.png'):
            title = file.replace('_', ' ').replace('.png', '').title()
            html_content += f"""
            <div class="plot">
                <h3>{title}</h3>
                <img src="{file}" alt="{title}">
            </div>
            """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(f"{output_dir}/index.html", "w") as f:
        f.write(html_content)
    
    print(f"🌐 View all plots: open {output_dir}/index.html")

if __name__ == "__main__":
    test_visualizations()