#!/usr/bin/env python3
"""
GraphVerse Small-Scale Context Boundary Analysis

Single entry point script for running quick experiments with:
1. Prerequisites validation
2. Graph generation (100 vertices)
3. Model training (context window: 8)
4. Small-scale evaluation (10K walks)
5. Results analysis including violation entropy analysis

This script provides the fastest way to test GraphVerse functionality,
perfect for debugging, development, and testing the new violation entropy analysis.

Runtime: ~15-30 minutes total
Memory: ~2GB RAM
Scale: 10K total walks with comprehensive entropy analysis
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.check_prerequisites import PrerequisiteChecker
from configs.small_scale_config import SMALL_SCALE_CONFIG, validate_small_scale_config, estimate_memory_requirements


def run_command(command, description, check_exit=True, verbose=True):
    """
    Run a shell command with error checking.
    
    Args:
        command: Command to run (list or string)
        description: Description for logging
        check_exit: Whether to check exit code
        verbose: Whether to print command output
        
    Returns:
        CompletedProcess result
    """
    if verbose:
        print(f"\\n{'='*60}")
        print(f"RUNNING: {description}")
        print(f"{'='*60}")
        if isinstance(command, list) and len(command) >= 3 and command[1] == "-c":
            print(f"Command: {command[0]} -c <python_script>")
        else:
            print(f"Command: {' '.join(command) if isinstance(command, list) else command}")
    
    start_time = time.time()
    
    try:
        if isinstance(command, str):
            command = command.split()
        
        result = subprocess.run(
            command,
            capture_output=not verbose,
            text=True,
            check=check_exit
        )
        
        duration = time.time() - start_time
        
        if verbose:
            print(f"\\n✅ {description} completed in {duration:.1f} seconds")
        
        return result
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\\n❌ {description} failed after {duration:.1f} seconds")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\\n{e.stderr}")
        raise e


def check_prerequisites(graph_path=None, models_dir=None, verbose=True):
    """Check system prerequisites for small-scale experiments."""
    if verbose:
        print("🔍 Checking system prerequisites for small-scale analysis...")
        
        # Show small-scale requirements
        memory_est = estimate_memory_requirements(SMALL_SCALE_CONFIG)
        validation = validate_small_scale_config(SMALL_SCALE_CONFIG)
        
        print(f"\\n📊 Small-Scale Requirements:")
        print(f"  • Estimated memory: {memory_est['total_estimated']:.1f}GB")
        print(f"  • Estimated runtime: {validation['estimated_runtime_hours']:.1f} hours")
        print(f"  • Graph size: {SMALL_SCALE_CONFIG['n']:,} vertices")
        print(f"  • Total walks: {len(SMALL_SCALE_CONFIG['context_windows']) * SMALL_SCALE_CONFIG['num_walks']:,}")
    
    checker = PrerequisiteChecker(verbose=verbose)
    summary = checker.run_all_checks(graph_path=graph_path, models_dir=models_dir)
    
    if not summary["can_proceed"]:
        print("\\n❌ Prerequisites check failed. Please fix the errors above and try again.")
        return False
    
    if verbose and summary["warnings"]:
        print(f"\\n⚠️ {len(summary['warnings'])} warnings found. Proceeding anyway...")
    
    return True


def generate_graph(output_path="small_graph_100", force=False, verbose=True):
    """Generate small-scale graph."""
    # Check if graph already exists
    if not force and os.path.exists(f"{output_path}.npy"):
        if verbose:
            print(f"📊 Graph already exists: {output_path}.npy")
            print("Use --force-graph to regenerate")
        return True
    
    if verbose:
        print("📊 Generating small-scale graph (100 vertices)...")
    
    # Create a custom generate command using the small scale config
    command = [
        sys.executable, "-c", f"""
import sys
sys.path.insert(0, ".")
from configs.small_scale_config import SMALL_SCALE_CONFIG
from scripts.generate_graph import generate_and_save_graph

# Generate graph using small scale config
graph, rules = generate_and_save_graph(
    config=SMALL_SCALE_CONFIG,
    output_path="{output_path}",
    verbose={verbose}
)
print(f"✅ Small-scale graph generated: {output_path}")
"""
    ]
    
    try:
        run_command(command, "Small-scale graph generation", verbose=verbose)
        return True
    except subprocess.CalledProcessError:
        return False


def train_models(graph_path="small_graph_100", output_dir="small_models", 
                force=False, contexts=None, verbose=True):
    """Train models for small-scale context windows."""
    # Check if models already exist
    if not force and os.path.exists(output_dir):
        existing_models = [f for f in os.listdir(output_dir) if f.startswith("model_ctx_") and f.endswith(".pt")]
        if existing_models and verbose:
            print(f"🤖 {len(existing_models)} models already exist in {output_dir}")
            print("Use --force-models to retrain")
            return True
    
    if verbose:
        print("🤖 Training small-scale models...")
        print("⏰ This should take ~5-10 minutes...")
        print(f"📋 Context windows: {contexts or SMALL_SCALE_CONFIG['context_windows']}")
    
    # Create custom training command using small scale config
    contexts_str = str(contexts or SMALL_SCALE_CONFIG['context_windows']).replace(' ', '')
    
    command = [
        sys.executable, "-c", f"""
import sys
sys.path.insert(0, ".")
from configs.small_scale_config import SMALL_SCALE_CONFIG
from scripts.train_models import train_all_models

# Train models using small scale config
try:
    model_paths = train_all_models(
        graph_path="{graph_path}",
        output_dir="{output_dir}",
        context_windows={contexts_str},
        config=SMALL_SCALE_CONFIG,
        verbose={verbose}
    )
    print(f"✅ Small-scale models trained in {output_dir}")
    print(f"Models: {{list(model_paths.keys())}}")
except Exception as e:
    print(f"❌ Model training failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    ]
    
    try:
        run_command(command, "Small-scale model training", verbose=verbose)
        return True
    except subprocess.CalledProcessError:
        return False


def run_experiment(graph_path="small_graph_100", models_dir="small_models",
                  output_dir="small_results", contexts=None, device="auto", verbose=True):
    """Run the small-scale experiment."""
    if verbose:
        print("🚀 Running small-scale context boundary experiment...")
        print("⏰ This should take ~10-20 minutes for complete analysis...")
        print("🔬 Includes comprehensive violation entropy analysis!")
        
        total_walks = len(contexts or SMALL_SCALE_CONFIG['context_windows']) * SMALL_SCALE_CONFIG['num_walks']
        print(f"📊 Total walks to process: {total_walks:,}")
    
    # Create custom experiment command using small scale config
    contexts_str = str(contexts or SMALL_SCALE_CONFIG['context_windows']).replace(' ', '')
    
    command = [
        sys.executable, "-c", f"""
import sys
sys.path.insert(0, ".")
from configs.small_scale_config import SMALL_SCALE_CONFIG
from scripts.run_experiment import run_all_experiments

# Run experiment using small scale config  
try:
    result_paths = run_all_experiments(
        graph_path="{graph_path}",
        models_dir="{models_dir}",
        output_dir="{output_dir}",
        context_windows={contexts_str},
        device="{device}",
        verbose={verbose}
    )
    print(f"✅ Small-scale experiment completed in {output_dir}")
    print(f"Results: {{list(result_paths.keys())}}")
except Exception as e:
    print(f"❌ Experiment failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    ]
    
    try:
        run_command(command, "Small-scale experiment", verbose=verbose)
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Custom experiment runner not available, trying standard approach...")
        
        # Fallback to standard approach
        command = [
            sys.executable, "scripts/run_experiment.py",
            "--graph", graph_path,
            "--models", models_dir,
            "--output", output_dir,
            "--device", device
        ]
        
        if contexts:
            command.extend(["--contexts"] + [str(c) for c in contexts])
        
        if not verbose:
            command.append("--quiet")
        
        try:
            run_command(command, "Small-scale experiment (fallback)", verbose=verbose)
            return True
        except subprocess.CalledProcessError:
            return False


def start_monitoring(results_dir="small_results"):
    """Start experiment monitoring in background."""
    print("📊 Starting experiment monitor...")
    print(f"Monitor your experiment with:")
    print(f"  python scripts/monitor_experiment.py {results_dir} --watch")
    print(f"  python scripts/monitor_experiment.py {results_dir}")


def main():
    """Main orchestration function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run complete small-scale GraphVerse context boundary analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OVERVIEW:
This script runs a small-scale version of the GraphVerse context boundary analysis,
designed for rapid testing, debugging, and development of the violation entropy analysis.

• 100 vertex graph with realistic rule distribution
• Model trained on context window 8
• 10,000 walks with comprehensive entropy tracking
• Complete violation entropy analysis pipeline
• Complete analysis in 15-30 minutes

PHASES:
1. Prerequisites check (system requirements, dependencies)
2. Graph generation (100 vertices with rule assignments)
3. Model training (1 model, ~5-10 minutes)
4. Small-scale evaluation (10K walks, ~10-20 minutes)
5. Results analysis including violation entropy plots

SYSTEM REQUIREMENTS:
• 2GB+ RAM recommended
• 5GB+ free disk space
• Python 3.7+ with PyTorch
• Optional: GPU for faster training

NEW FEATURES:
• Violation entropy analysis over time
• KL divergence from all baselines before violations
• Entropy collapse dynamics visualization
• Oracle divergence pattern analysis
• Individual violation case studies

EXAMPLE USAGE:
  python run_small_scale_analysis.py                    # Full pipeline
  python run_small_scale_analysis.py --experiment-only  # Skip setup phases
  python run_small_scale_analysis.py --dry-run          # Show execution plan
        """
    )
    
    # Main execution control
    parser.add_argument("--skip-prereqs", action="store_true",
                       help="Skip prerequisites check")
    parser.add_argument("--skip-graph", action="store_true",
                       help="Skip graph generation")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training")
    parser.add_argument("--skip-experiment", action="store_true",
                       help="Skip experiment execution")
    parser.add_argument("--experiment-only", action="store_true",
                       help="Only run experiment (skip setup phases)")
    
    # Force regeneration
    parser.add_argument("--force-graph", action="store_true",
                       help="Force graph regeneration even if exists")
    parser.add_argument("--force-models", action="store_true",
                       help="Force model retraining even if exist")
    
    # Path configuration
    parser.add_argument("--graph-path", default="small_graph_100",
                       help="Path for graph files (default: small_graph_100)")
    parser.add_argument("--models-dir", default="small_models",
                       help="Directory for trained models (default: small_models)")
    parser.add_argument("--results-dir", default="small_results",
                       help="Directory for experiment results (default: small_results)")
    
    # Experiment configuration
    parser.add_argument("--contexts", nargs="+", type=int,
                       help="Context window sizes to test (default: [8])")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                       help="Device for training/evaluation (default: auto)")
    
    # Test and development
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")
    
    # Output control
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")
    parser.add_argument("--monitor", action="store_true",
                       help="Start monitoring after launching experiment")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Experiment-only mode
    if args.experiment_only:
        args.skip_prereqs = True
        args.skip_graph = True
        args.skip_training = True
    
    # Set default contexts if not specified
    if not args.contexts:
        args.contexts = SMALL_SCALE_CONFIG["context_windows"]
    
    if verbose:
        print("=" * 80)
        print("GRAPHVERSE SMALL-SCALE CONTEXT BOUNDARY ANALYSIS")
        print("=" * 80)
        print(f"📊 Graph path: {args.graph_path}")
        print(f"🤖 Models directory: {args.models_dir}")
        print(f"📈 Results directory: {args.results_dir}")
        print(f"🎯 Context windows: {args.contexts}")
        print(f"💻 Device: {args.device}")
        
        # Show scale comparison
        total_walks = len(args.contexts) * SMALL_SCALE_CONFIG['num_walks']
        print(f"\\n📊 Small-Scale Parameters:")
        print(f"  • Graph size: {SMALL_SCALE_CONFIG['n']:,} vertices")
        print(f"  • Walks per context: {SMALL_SCALE_CONFIG['num_walks']:,}")
        print(f"  • Total walks: {total_walks:,}")
        print(f"  • Context windows: {len(args.contexts)}")
        print("  • 🔬 VIOLATION ENTROPY ANALYSIS ENABLED")
        
        if args.dry_run:
            print("🔍 Mode: DRY RUN")
    
    if args.dry_run:
        print("\\n📋 Execution plan:")
        if not args.skip_prereqs: print("  1. Check prerequisites")
        if not args.skip_graph: print("  2. Generate small-scale graph (100 vertices)")
        if not args.skip_training: print("  3. Train small-scale model (context 8)")
        if not args.skip_experiment: print("  4. Run small-scale experiment (10K walks)")
        print("  5. Analysis and monitoring")
        print("  6. 🔬 Violation entropy analysis with comprehensive plots")
        print("\\nUse without --dry-run to execute.")
        return
    
    start_time = time.time()
    
    try:
        # Phase 1: Prerequisites Check
        if not args.skip_prereqs:
            if not check_prerequisites(
                graph_path=args.graph_path if not args.skip_graph else None,
                models_dir=args.models_dir if not args.skip_training else None,
                verbose=verbose
            ):
                sys.exit(1)
        
        # Phase 2: Graph Generation
        if not args.skip_graph:
            if not generate_graph(
                output_path=args.graph_path,
                force=args.force_graph,
                verbose=verbose
            ):
                print("❌ Graph generation failed")
                sys.exit(1)
        
        # Phase 3: Model Training
        if not args.skip_training:
            if not train_models(
                graph_path=args.graph_path,
                output_dir=args.models_dir,
                force=args.force_models,
                contexts=args.contexts,
                verbose=verbose
            ):
                print("❌ Model training failed")
                sys.exit(1)
        
        # Phase 4: Small-Scale Experiment
        if not args.skip_experiment:
            if args.monitor:
                # Start monitoring in background before experiment
                start_monitoring(args.results_dir)
            
            if not run_experiment(
                graph_path=args.graph_path,
                models_dir=args.models_dir,
                output_dir=args.results_dir,
                contexts=args.contexts,
                device=args.device,
                verbose=verbose
            ):
                print("❌ Experiment failed")
                sys.exit(1)
        
        # Phase 5: Final Summary
        total_time = time.time() - start_time
        
        if verbose:
            print("\\n" + "=" * 80)
            print("🎉 SMALL-SCALE ANALYSIS PIPELINE COMPLETED")
            print("=" * 80)
            print(f"⏱️ Total time: {total_time/60:.1f} minutes")
            print(f"📊 Graph: {args.graph_path}")
            print(f"🤖 Models: {args.models_dir}")
            print(f"📈 Results: {args.results_dir}")
            
            print(f"\\n📊 Monitor your results:")
            print(f"  python scripts/monitor_experiment.py {args.results_dir}")
            print(f"  python scripts/monitor_experiment.py {args.results_dir} --watch")
            
            print(f"\\n🔬 Violation Entropy Analysis:")
            print(f"  • Check {args.results_dir} for entropy timeline plots")
            print(f"  • Individual violation case studies available")
            print(f"  • Oracle divergence patterns analyzed")
            print(f"  • Comprehensive entropy metrics tracked over time")
            
            total_walks = len(args.contexts) * SMALL_SCALE_CONFIG['num_walks']
            print(f"\\n🚀 Small-Scale Research Complete:")
            print(f"  • Fast characterization of repeater context boundary effects")
            print(f"  • {total_walks:,} walks with step-by-step entropy tracking")
            print(f"  • Ready for violation entropy pattern analysis")
            print(f"  • Perfect for development and debugging")
        
    except KeyboardInterrupt:
        print("\\n\\n⏹️ Pipeline interrupted by user")
        total_time = time.time() - start_time
        print(f"⏱️ Runtime before interruption: {total_time/60:.1f} minutes")
        
        print(f"\\n🔄 To resume where you left off:")
        print(f"  python {sys.argv[0]} --skip-prereqs --skip-graph --skip-training")
        
        sys.exit(130)
    
    except Exception as e:
        print(f"\\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()