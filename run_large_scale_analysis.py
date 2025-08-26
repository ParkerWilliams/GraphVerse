#!/usr/bin/env python3
"""
GraphVerse Large-Scale Context Boundary Analysis

Single entry point script for running the complete large-scale experiment pipeline:
1. Prerequisites validation
2. Graph generation (10K vertices)
3. Model training (all context windows)
4. Large-scale evaluation (1M walks per context)
5. Results analysis and monitoring

This script orchestrates the complete research pipeline for analyzing how
repeater rules perform when they cross context window boundaries.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.check_prerequisites import PrerequisiteChecker


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
        print(f"\n{'='*60}")
        print(f"RUNNING: {description}")
        print(f"{'='*60}")
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
            print(f"\n✅ {description} completed in {duration:.1f} seconds")
        
        return result
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ {description} failed after {duration:.1f} seconds")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        raise e


def check_prerequisites(graph_path=None, models_dir=None, verbose=True):
    """Check system prerequisites."""
    if verbose:
        print("🔍 Checking system prerequisites...")
    
    checker = PrerequisiteChecker(verbose=verbose)
    summary = checker.run_all_checks(graph_path=graph_path, models_dir=models_dir)
    
    if not summary["can_proceed"]:
        print("\n❌ Prerequisites check failed. Please fix the errors above and try again.")
        return False
    
    if verbose and summary["warnings"]:
        print(f"\n⚠️ {len(summary['warnings'])} warnings found. Proceeding anyway...")
    
    return True


def generate_graph(output_path="large_scale_graph", force=False, verbose=True, medium_scale=False):
    """Generate large-scale graph."""
    # Check if graph already exists
    if not force and os.path.exists(f"{output_path}.npy"):
        if verbose:
            print(f"📊 Graph already exists: {output_path}.npy")
            print("Use --force-graph to regenerate")
        return True
    
    scale_name = "medium-scale" if medium_scale else "large-scale"
    if verbose:
        print(f"📊 Generating {scale_name} graph...")
    
    command = [
        sys.executable, "scripts/generate_graph.py",
        "--output", output_path,
        "--validate"
    ]
    
    if medium_scale:
        command.append("--medium-scale")
    
    if not verbose:
        command.append("--quiet")
    
    try:
        run_command(command, f"{scale_name.capitalize()} graph generation", verbose=verbose)
        return True
    except subprocess.CalledProcessError:
        return False


def train_models(graph_path="large_scale_graph", output_dir="large_scale_models", 
                force=False, contexts=None, verbose=True, medium_scale=False):
    """Train models for all context windows."""
    # Check if models already exist
    if not force and os.path.exists(output_dir):
        existing_models = [f for f in os.listdir(output_dir) if f.startswith("model_ctx_") and f.endswith(".pt")]
        if existing_models and verbose:
            print(f"🤖 {len(existing_models)} models already exist in {output_dir}")
            print("Use --force-models to retrain")
            return True
    
    scale_name = "medium-scale" if medium_scale else "large-scale"
    if verbose:
        print(f"🤖 Training {scale_name} models for all context windows...")
        if medium_scale:
            print("⏰ This should take ~1-2 hours...")
        else:
            print("⏰ This may take several hours...")
    
    command = [
        sys.executable, "scripts/train_models.py",
        "--graph", graph_path,
        "--output", output_dir,
        "--validate"
    ]
    
    if medium_scale:
        command.append("--medium-scale")
    
    if contexts:
        command.extend(["--contexts"] + [str(c) for c in contexts])
    
    if not verbose:
        command.append("--quiet")
    
    try:
        run_command(command, f"{scale_name.capitalize()} model training", verbose=verbose)
        return True
    except subprocess.CalledProcessError:
        return False


def run_experiment(graph_path="large_scale_graph", models_dir="large_scale_models",
                  output_dir="large_scale_results", contexts=None, device="auto", verbose=True):
    """Run the large-scale experiment."""
    if verbose:
        print("🚀 Running large-scale context boundary experiment...")
        print("⏰ This will take ~7 days for the complete analysis...")
    
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
        run_command(command, "Large-scale experiment", verbose=verbose)
        return True
    except subprocess.CalledProcessError:
        return False


def start_monitoring(results_dir="large_scale_results"):
    """Start experiment monitoring in background."""
    print("📊 Starting experiment monitor...")
    print(f"Monitor your experiment with:")
    print(f"  python scripts/monitor_experiment.py {results_dir} --watch")
    print(f"  python scripts/monitor_experiment.py {results_dir}")


def main():
    """Main orchestration function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run complete large-scale GraphVerse context boundary analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OVERVIEW:
This script runs the complete research pipeline for analyzing how repeater rules
perform when they cross context window boundaries. The experiment involves:

• 10,000 vertex graph with realistic rule distribution
• Models trained on context windows: 8, 16, 32, 64, 128, 256  
• 1,000,000 walks per context window (6M total)
• Step-by-step uncertainty tracking with trajectory sampling
• Comprehensive analysis of repeater performance degradation

PHASES:
1. Prerequisites check (system requirements, dependencies)
2. Graph generation (10K vertices with rule assignments)
3. Model training (6 models, ~4-8 hours total)
4. Large-scale evaluation (6M walks, ~7 days total)
5. Results analysis and monitoring

SYSTEM REQUIREMENTS:
• 32GB+ RAM recommended
• 100GB+ free disk space
• Python 3.7+ with PyTorch
• Optional: GPU for faster training

MONITORING:
Use 'python scripts/monitor_experiment.py <results_dir> --watch' to monitor progress.

EXAMPLE USAGE:
  python run_large_scale_analysis.py                    # Full pipeline
  python run_large_scale_analysis.py --quick-test       # Small test run
  python run_large_scale_analysis.py --experiment-only  # Skip setup phases
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
    parser.add_argument("--graph-path", default="large_scale_graph",
                       help="Path for graph files (default: large_scale_graph)")
    parser.add_argument("--models-dir", default="large_scale_models",
                       help="Directory for trained models (default: large_scale_models)")
    parser.add_argument("--results-dir", default="large_scale_results",
                       help="Directory for experiment results (default: large_scale_results)")
    
    # Experiment configuration
    parser.add_argument("--contexts", nargs="+", type=int,
                       help="Context window sizes to test (default: 8 16 32 64 128 256)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                       help="Device for training/evaluation (default: auto)")
    
    # Test and development
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test with small graph and limited contexts")
    parser.add_argument("--medium-scale", action="store_true",
                       help="Medium-scale experiment (1K vertices, 100K walks, 2-6 hours)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")
    
    # Output control
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")
    parser.add_argument("--monitor", action="store_true",
                       help="Start monitoring after launching experiment")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Quick test configuration
    if args.quick_test:
        if verbose:
            print("🧪 QUICK TEST MODE - Using reduced parameters")
        
        # Override paths and contexts for testing
        args.graph_path = "test_graph_1000"
        args.models_dir = "test_models"
        args.results_dir = "test_results"
        args.contexts = [8, 16] if not args.contexts else args.contexts[:2]
    
    # Medium-scale configuration
    if args.medium_scale:
        if verbose:
            print("🏗️ MEDIUM-SCALE MODE - Optimized for development and pilot studies")
        
        # Override paths and contexts for medium scale
        args.graph_path = "medium_graph_1000"
        args.models_dir = "medium_models"
        args.results_dir = "medium_results"
        args.contexts = [4, 6, 8, 12, 16, 24] if not args.contexts else args.contexts
    
    # Experiment-only mode
    if args.experiment_only:
        args.skip_prereqs = True
        args.skip_graph = True
        args.skip_training = True
    
    if verbose:
        print("=" * 80)
        print("GRAPHVERSE LARGE-SCALE CONTEXT BOUNDARY ANALYSIS")
        print("=" * 80)
        print(f"📊 Graph path: {args.graph_path}")
        print(f"🤖 Models directory: {args.models_dir}")
        print(f"📈 Results directory: {args.results_dir}")
        print(f"🎯 Context windows: {args.contexts or 'default [8, 16, 32, 64, 128, 256]'}")
        print(f"💻 Device: {args.device}")
        
        if args.quick_test:
            print("🧪 Mode: QUICK TEST")
        
        if args.dry_run:
            print("🔍 Mode: DRY RUN")
    
    if args.dry_run:
        print("\n📋 Execution plan:")
        if not args.skip_prereqs: print("  1. Check prerequisites")
        if not args.skip_graph: print("  2. Generate graph")
        if not args.skip_training: print("  3. Train models")
        if not args.skip_experiment: print("  4. Run experiment")
        print("  5. Analysis and monitoring")
        print("\nUse without --dry-run to execute.")
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
                verbose=verbose,
                medium_scale=args.medium_scale
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
                verbose=verbose,
                medium_scale=args.medium_scale
            ):
                print("❌ Model training failed")
                sys.exit(1)
        
        # Phase 4: Large-Scale Experiment
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
            print("\n" + "=" * 80)
            print("🎉 LARGE-SCALE ANALYSIS PIPELINE COMPLETED")
            print("=" * 80)
            print(f"⏱️ Total time: {total_time/3600:.1f} hours")
            print(f"📊 Graph: {args.graph_path}")
            print(f"🤖 Models: {args.models_dir}")
            print(f"📈 Results: {args.results_dir}")
            
            print(f"\n📊 Monitor your results:")
            print(f"  python scripts/monitor_experiment.py {args.results_dir}")
            print(f"  python scripts/monitor_experiment.py {args.results_dir} --watch")
            
            print(f"\n📈 Analyze results:")
            print(f"  python scripts/run_experiment.py --analyze --output {args.results_dir}")
            
            if not args.quick_test:
                print(f"\n🔬 Research Impact:")
                print(f"  • Complete characterization of repeater context boundary effects")
                print(f"  • 6M walks across 6 context window sizes")
                print(f"  • Step-by-step uncertainty trajectory analysis")
                print(f"  • Statistical significance for repeater learning limits")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Pipeline interrupted by user")
        total_time = time.time() - start_time
        print(f"⏱️ Runtime before interruption: {total_time/3600:.1f} hours")
        
        print(f"\n🔄 To resume where you left off:")
        print(f"  python {sys.argv[0]} --skip-prereqs --skip-graph --skip-training")
        
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()