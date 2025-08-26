#!/usr/bin/env python3
"""
GraphVerse Medium-Scale Context Boundary Analysis

Single entry point script for running the complete medium-scale experiment pipeline:
1. Prerequisites validation
2. Graph generation (1K vertices)
3. Model training (context windows: 4, 6, 8, 12, 16, 24)
4. Medium-scale evaluation (100K walks per context)
5. Results analysis and monitoring

This script provides a faster, development-friendly version of the large-scale
analysis, perfect for pilot studies, development, and resource-constrained environments.

Runtime: ~2-6 hours total (vs 7 days for large-scale)
Memory: ~8GB RAM (vs 32GB for large-scale)
Scale: 600K total walks (vs 6M for large-scale)
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.check_prerequisites import PrerequisiteChecker
from configs.medium_scale_config import MEDIUM_SCALE_CONFIG, validate_medium_scale_config, estimate_memory_requirements


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
    """Check system prerequisites for medium-scale experiments."""
    if verbose:
        print("🔍 Checking system prerequisites for medium-scale analysis...")
        
        # Show medium-scale requirements
        memory_est = estimate_memory_requirements(MEDIUM_SCALE_CONFIG)
        validation = validate_medium_scale_config(MEDIUM_SCALE_CONFIG)
        
        print(f"\\n📊 Medium-Scale Requirements:")
        print(f"  • Estimated memory: {memory_est['total_estimated']:.1f}GB")
        print(f"  • Estimated runtime: {validation['estimated_runtime_hours']:.1f} hours")
        print(f"  • Graph size: {MEDIUM_SCALE_CONFIG['n']:,} vertices")
        print(f"  • Total walks: {len(MEDIUM_SCALE_CONFIG['context_windows']) * MEDIUM_SCALE_CONFIG['num_walks']:,}")
    
    checker = PrerequisiteChecker(verbose=verbose)
    summary = checker.run_all_checks(graph_path=graph_path, models_dir=models_dir)
    
    if not summary["can_proceed"]:
        print("\\n❌ Prerequisites check failed. Please fix the errors above and try again.")
        return False
    
    if verbose and summary["warnings"]:
        print(f"\\n⚠️ {len(summary['warnings'])} warnings found. Proceeding anyway...")
    
    return True


def generate_graph(output_path="medium_graph_1000", force=False, verbose=True):
    """Generate medium-scale graph."""
    # Check if graph already exists
    if not force and os.path.exists(f"{output_path}.npy"):
        if verbose:
            print(f"📊 Graph already exists: {output_path}.npy")
            print("Use --force-graph to regenerate")
        return True
    
    if verbose:
        print("📊 Generating medium-scale graph (1K vertices)...")
    
    command = [
        sys.executable, "scripts/generate_graph.py",
        "--output", output_path,
        "--medium-scale",
        "--validate"
    ]
    
    if not verbose:
        command.append("--quiet")
    
    try:
        run_command(command, "Medium-scale graph generation", verbose=verbose)
        return True
    except subprocess.CalledProcessError:
        return False


def train_models(graph_path="medium_graph_1000", output_dir="medium_models", 
                force=False, contexts=None, verbose=True):
    """Train models for all medium-scale context windows."""
    # Check if models already exist
    if not force and os.path.exists(output_dir):
        existing_models = [f for f in os.listdir(output_dir) if f.startswith("model_ctx_") and f.endswith(".pt")]
        if existing_models and verbose:
            print(f"🤖 {len(existing_models)} models already exist in {output_dir}")
            print("Use --force-models to retrain")
            return True
    
    if verbose:
        print("🤖 Training medium-scale models...")
        print("⏰ This should take ~1-2 hours...")
        print(f"📋 Context windows: {contexts or MEDIUM_SCALE_CONFIG['context_windows']}")
    
    command = [
        sys.executable, "scripts/train_models.py",
        "--graph", graph_path,
        "--output", output_dir,
        "--medium-scale",
        "--validate"
    ]
    
    if contexts:
        command.extend(["--contexts"] + [str(c) for c in contexts])
    
    if not verbose:
        command.append("--quiet")
    
    try:
        run_command(command, "Medium-scale model training", verbose=verbose)
        return True
    except subprocess.CalledProcessError:
        return False


def run_experiment(graph_path="medium_graph_1000", models_dir="medium_models",
                  output_dir="medium_results", contexts=None, device="auto", verbose=True):
    """Run the medium-scale experiment."""
    if verbose:
        print("🚀 Running medium-scale context boundary experiment...")
        print("⏰ This should take ~2-4 hours for complete analysis...")
        
        total_walks = len(contexts or MEDIUM_SCALE_CONFIG['context_windows']) * MEDIUM_SCALE_CONFIG['num_walks']
        print(f"📊 Total walks to process: {total_walks:,}")
    
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
        run_command(command, "Medium-scale experiment", verbose=verbose)
        return True
    except subprocess.CalledProcessError:
        return False


def start_monitoring(results_dir="medium_results"):
    """Start experiment monitoring in background."""
    print("📊 Starting experiment monitor...")
    print(f"Monitor your experiment with:")
    print(f"  python scripts/monitor_experiment.py {results_dir} --watch")
    print(f"  python scripts/monitor_experiment.py {results_dir}")


def main():
    """Main orchestration function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run complete medium-scale GraphVerse context boundary analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OVERVIEW:
This script runs a medium-scale version of the GraphVerse context boundary analysis,
designed for faster iteration, development, and resource-constrained environments.

• 1,000 vertex graph with realistic rule distribution
• Models trained on context windows: 4, 6, 8, 12, 16, 24
• 100,000 walks per context window (600K total)
• Step-by-step uncertainty tracking with trajectory sampling
• Complete analysis in 2-6 hours (vs 7 days for large-scale)

PHASES:
1. Prerequisites check (system requirements, dependencies)
2. Graph generation (1K vertices with rule assignments)
3. Model training (6 models, ~1-2 hours total)
4. Medium-scale evaluation (600K walks, ~2-4 hours total)
5. Results analysis and monitoring

SYSTEM REQUIREMENTS:
• 8GB+ RAM recommended
• 20GB+ free disk space
• Python 3.7+ with PyTorch
• Optional: GPU for faster training

MONITORING:
Use 'python scripts/monitor_experiment.py <results_dir> --watch' to monitor progress.

EXAMPLE USAGE:
  python run_medium_scale_analysis.py                    # Full pipeline
  python run_medium_scale_analysis.py --quick-test       # Even smaller test run
  python run_medium_scale_analysis.py --experiment-only  # Skip setup phases
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
    parser.add_argument("--graph-path", default="medium_graph_1000",
                       help="Path for graph files (default: medium_graph_1000)")
    parser.add_argument("--models-dir", default="medium_models",
                       help="Directory for trained models (default: medium_models)")
    parser.add_argument("--results-dir", default="medium_results",
                       help="Directory for experiment results (default: medium_results)")
    
    # Experiment configuration
    parser.add_argument("--contexts", nargs="+", type=int,
                       help="Context window sizes to test (default: 4 6 8 12 16 24)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                       help="Device for training/evaluation (default: auto)")
    
    # Test and development
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test with minimal contexts")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")
    
    # Output control
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")
    parser.add_argument("--monitor", action="store_true",
                       help="Start monitoring after launching experiment")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Quick test configuration (even smaller than medium-scale)
    if args.quick_test:
        if verbose:
            print("🧪 QUICK TEST MODE - Using minimal parameters")
        
        # Override paths and contexts for testing
        args.graph_path = "test_graph_500"
        args.models_dir = "test_models_small"
        args.results_dir = "test_results_small"
        args.contexts = [4, 8] if not args.contexts else args.contexts[:2]
    
    # Experiment-only mode
    if args.experiment_only:
        args.skip_prereqs = True
        args.skip_graph = True
        args.skip_training = True
    
    # Set default contexts if not specified
    if not args.contexts:
        args.contexts = MEDIUM_SCALE_CONFIG["context_windows"]
    
    if verbose:
        print("=" * 80)
        print("GRAPHVERSE MEDIUM-SCALE CONTEXT BOUNDARY ANALYSIS")
        print("=" * 80)
        print(f"📊 Graph path: {args.graph_path}")
        print(f"🤖 Models directory: {args.models_dir}")
        print(f"📈 Results directory: {args.results_dir}")
        print(f"🎯 Context windows: {args.contexts}")
        print(f"💻 Device: {args.device}")
        
        # Show scale comparison
        total_walks = len(args.contexts) * MEDIUM_SCALE_CONFIG['num_walks']
        print(f"\\n📊 Medium-Scale Parameters:")
        print(f"  • Graph size: {MEDIUM_SCALE_CONFIG['n']:,} vertices")
        print(f"  • Walks per context: {MEDIUM_SCALE_CONFIG['num_walks']:,}")
        print(f"  • Total walks: {total_walks:,}")
        print(f"  • Context windows: {len(args.contexts)}")
        
        if args.quick_test:
            print("🧪 Mode: QUICK TEST")
        
        if args.dry_run:
            print("🔍 Mode: DRY RUN")
    
    if args.dry_run:
        print("\\n📋 Execution plan:")
        if not args.skip_prereqs: print("  1. Check prerequisites")
        if not args.skip_graph: print("  2. Generate medium-scale graph")
        if not args.skip_training: print("  3. Train medium-scale models")
        if not args.skip_experiment: print("  4. Run medium-scale experiment")
        print("  5. Analysis and monitoring")
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
        
        # Phase 4: Medium-Scale Experiment
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
            print("🎉 MEDIUM-SCALE ANALYSIS PIPELINE COMPLETED")
            print("=" * 80)
            print(f"⏱️ Total time: {total_time/3600:.1f} hours")
            print(f"📊 Graph: {args.graph_path}")
            print(f"🤖 Models: {args.models_dir}")
            print(f"📈 Results: {args.results_dir}")
            
            print(f"\\n📊 Monitor your results:")
            print(f"  python scripts/monitor_experiment.py {args.results_dir}")
            print(f"  python scripts/monitor_experiment.py {args.results_dir} --watch")
            
            print(f"\\n📈 Analyze results:")
            print(f"  python scripts/run_large_scale_experiment.py --analyze --output {args.results_dir}")
            
            if not args.quick_test:
                total_walks = len(args.contexts) * MEDIUM_SCALE_CONFIG['num_walks']
                print(f"\\n🔬 Research Pilot Complete:")
                print(f"  • Medium-scale characterization of repeater context boundary effects")
                print(f"  • {total_walks:,} walks across {len(args.contexts)} context window sizes")
                print(f"  • Step-by-step uncertainty trajectory analysis")
                print(f"  • Validation for scaling to large-scale experiments")
        
    except KeyboardInterrupt:
        print("\\n\\n⏹️ Pipeline interrupted by user")
        total_time = time.time() - start_time
        print(f"⏱️ Runtime before interruption: {total_time/3600:.1f} hours")
        
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