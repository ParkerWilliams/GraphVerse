# Large-Scale GraphVerse Context Boundary Analysis

This document provides complete instructions for running the large-scale experiment to analyze how repeater rules perform when they cross context window boundaries.

## 🎯 Research Question

**How do repeaters with length k perform as they cross the context window boundary?**

The experiment analyzes 6 context window sizes (8, 16, 32, 64, 128, 256) with 1M walks each to characterize repeater learning degradation at context boundaries.

## 🏗️ Infrastructure Overview

- **Scale**: 10K vertices, 6M total walks, 6 context windows
- **Memory**: Trajectory sampling (2% rate) for memory efficiency
- **Processing**: Batch processing with checkpointing every 100K walks
- **Monitoring**: Real-time memory and progress tracking
- **Recovery**: Automatic resume from interruptions

## 🚀 Quick Start

### Single Command (Complete Pipeline)

```bash
# Full experiment - will run for ~7 days
source venv/bin/activate
python run_large_scale_analysis.py

# Quick test with small graph
python run_large_scale_analysis.py --quick-test

# Check what would be done without executing
python run_large_scale_analysis.py --dry-run
```

### Step by Step

```bash
# 1. Check prerequisites
python scripts/check_prerequisites.py

# 2. Generate 10K vertex graph
python scripts/generate_large_scale_graph.py --validate

# 3. Train models for all context windows (~4-8 hours)
python scripts/train_large_scale_models.py --validate

# 4. Run large-scale experiment (~7 days)
python scripts/run_large_scale_experiment.py

# 5. Monitor progress
python scripts/monitor_experiment.py large_scale_results --watch
```

## 📋 Prerequisites

### System Requirements
- **Memory**: 32GB+ RAM recommended (minimum 16GB)
- **Storage**: 100GB+ free disk space
- **Time**: ~7 days for complete analysis
- **Python**: 3.7+ with PyTorch
- **Optional**: GPU for faster training

### Dependencies
```bash
pip install torch numpy scipy tqdm psutil
```

### Verification
```bash
python scripts/check_prerequisites.py
```

## 🔧 Configuration

### Large-Scale Configuration
```python
# configs/large_scale_config.py
LARGE_SCALE_CONFIG = {
    "n": 10000,                    # 10K vertices
    "num_walks": 1000000,          # 1M walks per context
    "context_windows": [8, 16, 32, 64, 128, 256],
    "trajectory_sampling": {
        "sample_rate": 0.02,       # Store 2% full trajectories
        "stratified": True
    },
    "batch_processing": {
        "walk_batch_size": 50000,  # 50K walks per batch
        "memory_limit_gb": 32
    }
}
```

### Memory Optimization
```bash
# Check memory requirements
python -c "
from configs.large_scale_config import estimate_memory_requirements, LARGE_SCALE_CONFIG
print(estimate_memory_requirements(LARGE_SCALE_CONFIG))
"

# Get sampling recommendations
python -c "
from graphverse.utils.memory_monitor import MemoryOptimizer
print(MemoryOptimizer.recommend_sample_rate(1000000, 32.0))
"
```

## 📊 Monitoring

### Real-Time Monitoring
```bash
# Watch mode (updates every 30 seconds)
python scripts/monitor_experiment.py large_scale_results --watch

# Single status check
python scripts/monitor_experiment.py large_scale_results

# Multi-experiment dashboard
python scripts/monitor_experiment.py exp1 exp2 exp3 --multi --watch
```

### Progress Checkpoints
- Automatic checkpoints every 100K walks
- Recovery from any checkpoint
- Memory usage alerts at 80% capacity
- Progress estimation and ETA calculation

## 🎛️ Experiment Control

### Resuming Interrupted Experiments
```bash
# Resume from where you left off
python run_large_scale_analysis.py --experiment-only

# Resume specific components
python run_large_scale_analysis.py --skip-prereqs --skip-graph
```

### Running Partial Experiments
```bash
# Specific context windows only
python run_large_scale_analysis.py --contexts 8 16 32

# Skip certain phases
python run_large_scale_analysis.py --skip-training

# Force regeneration
python run_large_scale_analysis.py --force-graph --force-models
```

### Device Configuration
```bash
# Use GPU if available
python run_large_scale_analysis.py --device auto

# Force CPU usage
python run_large_scale_analysis.py --device cpu

# Specific GPU
python run_large_scale_analysis.py --device cuda
```

## 📈 Results Analysis

### Experiment Output Structure
```
large_scale_results/
├── experiment_config.json          # Overall configuration
├── results_mapping.json            # Results summary
├── context_8/                      # Individual context experiments
│   ├── evaluation/
│   │   ├── final_results.json      # Aggregated error rates
│   │   ├── trajectory_metadata.pkl # Full trajectory data
│   │   └── trajectories.h5         # Compressed numerical data
│   ├── batches/                    # Batch processing results
│   └── checkpoints/                # Progress checkpoints
├── context_16/
└── ...
```

### Key Results Files
- `final_results.json`: Aggregated error rates by rule type
- `trajectory_metadata.pkl`: Complete trajectory uncertainty data
- `trajectories.h5`: Compressed probability distributions
- `error_rates_by_context.csv`: Summary for plotting

### Analysis Scripts
```bash
# Analyze completed results
python scripts/run_large_scale_experiment.py --analyze --output large_scale_results

# Generate plots and summaries
python -c "
from graphverse.llm.evaluation_vis import plot_context_window_analysis
plot_context_window_analysis('large_scale_results')
"
```

## 🔬 Research Impact

### Expected Outcomes
1. **Quantitative characterization** of repeater learning limits
2. **Context boundary effects** on rule compliance
3. **Uncertainty trajectory analysis** showing where models fail
4. **Statistical significance** for context window recommendations

### Data Products
- 6M walk evaluations with step-by-step uncertainty
- 120K full probability distribution trajectories  
- Comprehensive repeater performance degradation curves
- Context window optimization guidelines

## 🛠️ Troubleshooting

### Common Issues

**Memory Errors**
```bash
# Reduce trajectory sampling rate
python -c "
config = {'trajectory_sampling': {'sample_rate': 0.01}}
# Update config and retry
"

# Reduce batch size
python scripts/run_large_scale_experiment.py --contexts 8 16  # Fewer contexts
```

**Storage Issues**
```bash
# Check disk space requirements
python -c "
from configs.large_scale_config import estimate_memory_requirements, LARGE_SCALE_CONFIG
est = estimate_memory_requirements(LARGE_SCALE_CONFIG)
print(f'Storage needed: {est[\"total_estimated\"]:.1f} GB')
"

# Enable compression
# Edit configs/large_scale_config.py: "compress_trajectories": True
```

**GPU Issues**
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Force CPU if GPU causes issues
python run_large_scale_analysis.py --device cpu
```

### Debugging Mode
```bash
# Verbose output with debugging
python run_large_scale_analysis.py --contexts 8 --quick-test

# Check individual components
python scripts/generate_large_scale_graph.py --validate
python scripts/train_large_scale_models.py --contexts 8 --validate
python scripts/check_prerequisites.py
```

### Recovery Procedures
```bash
# Clean restart
rm -rf large_scale_graph* large_scale_models/ large_scale_results/
python run_large_scale_analysis.py

# Partial recovery
python run_large_scale_analysis.py --skip-graph --force-models

# Resume from checkpoint
python scripts/run_large_scale_experiment.py --output large_scale_results
```

## 📞 Support

For issues or questions:
1. Check prerequisites: `python scripts/check_prerequisites.py`
2. Run quick test: `python run_large_scale_analysis.py --quick-test --dry-run`
3. Monitor progress: `python scripts/monitor_experiment.py <results_dir> --watch`
4. Check system resources: `htop` or `nvidia-smi`

## 📄 Citation

```bibtex
@article{graphverse_context_boundary_2024,
  title={Context Window Boundary Effects on Repeater Rule Learning in Graph Neural Networks},
  author={Your Name},
  journal={Under Review},
  year={2024},
  note={Large-scale analysis with 6M walks across 6 context window sizes}
}
```