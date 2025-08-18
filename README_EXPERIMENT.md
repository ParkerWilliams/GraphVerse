# 🚀 Large-Scale Context Boundary Experiment

## One Command to Run Everything

```bash
# Activate environment and run complete experiment
source venv/bin/activate
python run_large_scale_analysis.py
```

This single command will:
1. ✅ Check system prerequisites (32GB RAM, 100GB disk)
2. 📊 Generate 10K vertex graph with rule assignments  
3. 🤖 Train 6 models for context windows [8, 16, 32, 64, 128, 256]
4. 🧪 Run 6M walks (1M per context) with uncertainty tracking
5. 📈 Analyze repeater performance across context boundaries

**Expected Runtime**: ~7 days for complete analysis

## Quick Test (5 minutes)

```bash
# Small test to verify everything works
python run_large_scale_analysis.py --quick-test
```

## Monitor Progress

```bash
# Real-time monitoring (in another terminal)
python scripts/monitor_experiment.py large_scale_results --watch
```

## Prerequisites Check

```bash
# Verify your system is ready
python scripts/check_prerequisites.py
```

## What You Get

- **6M evaluated walks** across 6 context window sizes
- **Step-by-step uncertainty** for 120K full trajectory samples  
- **Complete characterization** of repeater context boundary effects
- **Statistical significance** for context window recommendations

## Research Question

**How do repeaters with length k perform as they cross the context window boundary?**

This experiment provides the definitive answer by analyzing millions of walks with sophisticated uncertainty tracking.

---

📖 **Full Documentation**: [LARGE_SCALE_EXPERIMENT.md](LARGE_SCALE_EXPERIMENT.md)

🔧 **Troubleshooting**: Run `python run_large_scale_analysis.py --help` for all options