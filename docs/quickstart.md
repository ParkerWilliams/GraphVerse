# Quick Start Guide

Get GraphVerse running in 5 minutes and generate your first rule learning analysis.

## Installation

1. **Clone and install**:
```bash
git clone https://github.com/yourusername/GraphVerse.git
cd GraphVerse
pip install -e .
```

2. **Verify installation**:
```bash
python -c "import graphverse; print('Installation successful!')"
```

## Your First Experiment

### Option 1: Complete Analysis (Recommended)

Run the full analysis pipeline:

```bash
cd scripts
python complete_rule_analysis_demo.py
```

This will:
- Generate a 100-node graph with embedded rules
- Train a small transformer (2-layer, 128-dim)
- Track rule violations during training
- Create 15+ visualizations
- Generate an automated blog post

**Expected runtime**: 10-15 minutes  
**Output**: `experiments/run_YYYYMMDD_HHMMSS/`

### Option 2: Token Analysis Only

For a quick token-level analysis demo:

```bash
cd scripts  
python example_token_analysis.py
```

**Expected runtime**: 3-5 minutes

## Understanding the Output

After running the complete analysis, you'll find:

```
experiments/run_YYYYMMDD_HHMMSS/
├── blog_post.md                    # Comprehensive analysis writeup
├── config.json                     # Experiment configuration
├── training_progression.json       # Learning curve data
├── checkpoints/                    # Model checkpoints
│   ├── model_epoch_0.pth
│   ├── model_epoch_2.pth
│   └── ...
├── evaluation/                     # Detailed analysis data
│   ├── token_level_data.json       # Token-by-token analysis
│   ├── rule_violations.json        # Violation summaries  
│   ├── error_summary.json          # Overall statistics
│   └── *.png                      # Basic plots
└── learning_progression_viz/       # Advanced visualizations
    ├── combined_rule_progression.png
    ├── confidence_evolution.png
    ├── rule_difficulty_ranking.png
    └── ...
```

## Key Files to Check

1. **`blog_post.md`** - Start here! Comprehensive analysis with insights
2. **`learning_progression_viz/combined_rule_progression.png`** - Shows how each rule is learned
3. **`evaluation/token_level_data.json`** - Raw data for custom analysis
4. **`learning_progression_viz/rule_difficulty_ranking.png`** - Which rules are hardest to learn

## Customization

### Quick Config Changes

Edit the demo script directly:
```python
# In complete_rule_analysis_demo.py, modify:
config = {
    'n': 200,              # Larger graph  
    'num_epochs': 20,      # More training
    'hidden_size': 512,    # Bigger model
    # ...
}
```

### Use Predefined Configs

```bash
# Small/fast demo
python complete_rule_analysis_demo.py --config ../configs/small_demo_config.json

# Full-scale experiment  
python complete_rule_analysis_demo.py --config ../configs/default_config.json
```

## Multi-Model Comparison

Compare different model architectures:

```bash
cd scripts
python multi_model_comparison.py
```

This trains 3-5 different model sizes and compares their rule learning abilities.

**Expected runtime**: 30-60 minutes  
**Output**: `multi_model_experiments_YYYYMMDD_HHMMSS/`

## Troubleshooting

### Common Issues

**ModuleNotFoundError**: Make sure you're in the `scripts/` directory:
```bash
cd GraphVerse/scripts
python complete_rule_analysis_demo.py
```

**CUDA out of memory**: Reduce batch size or model size in config:
```python
config['batch_size'] = 16    # Instead of 32
config['hidden_size'] = 128  # Instead of 256  
```

**Slow training**: Use the small demo config:
```bash
python complete_rule_analysis_demo.py --config ../configs/small_demo_config.json
```

### Getting Help

- Check the [full documentation](../README.md)
- Look at example outputs in `experiments/`
- Open an [issue](https://github.com/yourusername/GraphVerse/issues) if you're stuck

## Next Steps

Once you have basic results:

1. **Analyze the blog post** - Read the automated analysis in `blog_post.md`
2. **Explore visualizations** - Check out the plots in `learning_progression_viz/`  
3. **Dive into token data** - Use `evaluation/token_level_data.json` for custom analysis
4. **Try multi-model comparison** - Run the comparative analysis
5. **Modify rules** - Edit the rule definitions to test new constraints

## What You Should See

After a successful run, you should observe:

- **Rule violation rates decrease** from ~80-100% to 5-25% during training
- **Confidence gap emerges** between valid (0.7-0.9) and invalid (0.4-0.6) predictions  
- **Rule difficulty hierarchy**: Invalid edges learned fastest, repeater rules slowest
- **Learning curves** show different rules learned at different rates

If you don't see these patterns, check your configuration or try the small demo first.

---

**Next**: Read the [API Reference](api.md) for advanced usage