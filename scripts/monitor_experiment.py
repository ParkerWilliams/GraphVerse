#!/usr/bin/env python3
"""
Experiment Progress Monitor

Real-time monitoring and status reporting for large-scale GraphVerse experiments.
"""

import os
import sys
import json
import time
import pickle
import glob
from datetime import datetime, timedelta
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from graphverse.utils.checkpoint_manager import CheckpointManager
    from graphverse.utils.memory_monitor import MemoryMonitor
    import psutil
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    IMPORT_ERROR = str(e)


class ExperimentMonitor:
    """Monitor running and completed experiments."""
    
    def __init__(self, experiment_dir, verbose=True):
        """
        Initialize experiment monitor.
        
        Args:
            experiment_dir: Directory containing experiment data
            verbose: Whether to print status updates
        """
        self.experiment_dir = experiment_dir
        self.verbose = verbose
        
        # Load experiment configuration if available
        self.config = self.load_experiment_config()
        self.checkpoint_manager = None
        
        if MONITORING_AVAILABLE and os.path.exists(experiment_dir):
            try:
                self.checkpoint_manager = CheckpointManager(experiment_dir)
            except Exception:
                pass  # Checkpointing not available for this experiment
    
    def load_experiment_config(self):
        """Load experiment configuration."""
        config_files = [
            "experiment_config.json",
            "config.json",
            "training_config.json"
        ]
        
        for config_file in config_files:
            config_path = os.path.join(self.experiment_dir, config_file)
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        return json.load(f)
                except Exception:
                    continue
        
        return {}
    
    def get_experiment_status(self):
        """Get current experiment status."""
        if not os.path.exists(self.experiment_dir):
            return {"status": "not_found", "message": f"Experiment directory not found: {self.experiment_dir}"}
        
        status = {
            "experiment_dir": self.experiment_dir,
            "status": "unknown",
            "progress": {},
            "performance": {},
            "errors": [],
            "last_updated": time.time()
        }
        
        # Check if experiment is running or completed
        status.update(self.check_completion_status())
        status["progress"].update(self.analyze_progress())
        status["performance"].update(self.analyze_performance())
        status["errors"].extend(self.check_for_errors())
        
        return status
    
    def check_completion_status(self):
        """Check if experiment is running, completed, or failed."""
        # Check for completion markers
        results_mapping_path = os.path.join(self.experiment_dir, "results_mapping.json")
        final_results_path = os.path.join(self.experiment_dir, "evaluation", "final_results.json")
        
        if os.path.exists(results_mapping_path) or os.path.exists(final_results_path):
            return {"status": "completed"}
        
        # Check for recent checkpoint activity (indicates running)
        if self.checkpoint_manager:
            checkpoints = self.checkpoint_manager.list_checkpoints()
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x["timestamp"])
                checkpoint_time = datetime.strptime(latest_checkpoint["timestamp"], "%Y%m%d_%H%M%S")
                time_since_checkpoint = datetime.now() - checkpoint_time
                
                if time_since_checkpoint < timedelta(hours=2):
                    return {"status": "running", "last_checkpoint": latest_checkpoint["timestamp"]}
                else:
                    return {"status": "stalled", "last_checkpoint": latest_checkpoint["timestamp"]}
        
        # Check for recent file activity
        recent_files = self.find_recent_files(hours=2)
        if recent_files:
            return {"status": "running", "recent_activity": len(recent_files)}
        
        # Check if experiment was started but not completed
        if any(os.path.exists(os.path.join(self.experiment_dir, d)) for d in ["checkpoints", "batches", "evaluation"]):
            return {"status": "incomplete"}
        
        return {"status": "not_started"}
    
    def analyze_progress(self):
        """Analyze experiment progress."""
        progress = {}
        
        # Load configuration to get targets
        total_walks_target = 0
        total_contexts_target = 0
        
        if self.config:
            if "num_walks" in self.config and "context_windows" in self.config:
                total_walks_target = self.config["num_walks"] * len(self.config["context_windows"])
                total_contexts_target = len(self.config["context_windows"])
            elif "total_walks" in self.config:
                total_walks_target = self.config["total_walks"]
        
        progress["total_walks_target"] = total_walks_target
        progress["total_contexts_target"] = total_contexts_target
        
        # Analyze checkpoint progress
        if self.checkpoint_manager:
            checkpoints = self.checkpoint_manager.list_checkpoints()
            if checkpoints:
                latest_checkpoint_data = self.checkpoint_manager.load_checkpoint()
                if latest_checkpoint_data and "state" in latest_checkpoint_data:
                    state = latest_checkpoint_data["state"]
                    progress["completed_items"] = state.get("completed_items", 0)
                    progress["total_items"] = state.get("total_items", total_walks_target)
                    
                    if progress["total_items"] > 0:
                        progress["progress_percentage"] = (progress["completed_items"] / progress["total_items"]) * 100
        
        # Count completed batch files
        batch_dir = os.path.join(self.experiment_dir, "batches")
        if os.path.exists(batch_dir):
            batch_files = glob.glob(os.path.join(batch_dir, "batch_*.pkl"))
            progress["completed_batches"] = len(batch_files)
        
        # Count completed contexts
        contexts_completed = 0
        if os.path.exists(self.experiment_dir):
            context_dirs = [d for d in os.listdir(self.experiment_dir) 
                           if d.startswith("context_") and os.path.isdir(os.path.join(self.experiment_dir, d))]
            
            for context_dir in context_dirs:
                context_path = os.path.join(self.experiment_dir, context_dir)
                final_results = os.path.join(context_path, "evaluation", "final_results.json")
                if os.path.exists(final_results):
                    contexts_completed += 1
        
        progress["completed_contexts"] = contexts_completed
        
        if total_contexts_target > 0:
            progress["context_progress_percentage"] = (contexts_completed / total_contexts_target) * 100
        
        return progress
    
    def analyze_performance(self):
        """Analyze experiment performance metrics."""
        performance = {}
        
        # Load memory monitoring data if available
        memory_files = glob.glob(os.path.join(self.experiment_dir, "**/memory_*.json"), recursive=True)
        if memory_files:
            try:
                with open(memory_files[-1], "r") as f:
                    memory_data = json.load(f)
                performance["memory"] = memory_data
            except Exception:
                pass
        
        # Analyze batch processing performance
        batch_dir = os.path.join(self.experiment_dir, "batches")
        if os.path.exists(batch_dir):
            batch_files = glob.glob(os.path.join(batch_dir, "batch_*.pkl"))
            
            if batch_files:
                batch_times = []
                batch_rates = []
                
                for batch_file in batch_files:
                    try:
                        with open(batch_file, "rb") as f:
                            batch_data = pickle.load(f)
                        
                        if "batch_time" in batch_data:
                            batch_times.append(batch_data["batch_time"])
                        
                        if "batch_rate" in batch_data:
                            batch_rates.append(batch_data["batch_rate"])
                    except Exception:
                        continue
                
                if batch_times:
                    performance["avg_batch_time"] = sum(batch_times) / len(batch_times)
                    performance["total_processing_time"] = sum(batch_times)
                
                if batch_rates:
                    performance["avg_walks_per_second"] = sum(batch_rates) / len(batch_rates)
        
        # Estimate completion time
        progress = self.analyze_progress()
        
        if (progress.get("progress_percentage", 0) > 0 and 
            performance.get("avg_walks_per_second", 0) > 0):
            
            remaining_walks = progress.get("total_items", 0) - progress.get("completed_items", 0)
            if remaining_walks > 0:
                eta_seconds = remaining_walks / performance["avg_walks_per_second"]
                performance["estimated_completion"] = time.time() + eta_seconds
                performance["eta_hours"] = eta_seconds / 3600
        
        return performance
    
    def check_for_errors(self):
        """Check for experiment errors."""
        errors = []
        
        # Check for error logs
        error_files = glob.glob(os.path.join(self.experiment_dir, "**/error*.log"), recursive=True)
        for error_file in error_files:
            try:
                with open(error_file, "r") as f:
                    content = f.read().strip()
                    if content:
                        errors.append({
                            "type": "log_file",
                            "file": error_file,
                            "content": content[-500:]  # Last 500 chars
                        })
            except Exception:
                pass
        
        # Check checkpoint error counts
        if self.checkpoint_manager:
            latest_checkpoint = self.checkpoint_manager.load_checkpoint()
            if latest_checkpoint and "state" in latest_checkpoint:
                state = latest_checkpoint["state"]
                error_count = state.get("error_count", 0)
                if error_count > 0:
                    errors.append({
                        "type": "checkpoint_errors",
                        "count": error_count,
                        "message": f"{error_count} errors recorded in checkpoint"
                    })
        
        return errors
    
    def find_recent_files(self, hours=2):
        """Find files modified in the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_files = []
        
        try:
            for root, dirs, files in os.walk(self.experiment_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) > cutoff_time:
                        recent_files.append(file_path)
        except Exception:
            pass
        
        return recent_files
    
    def print_status(self, status=None):
        """Print formatted status report."""
        if status is None:
            status = self.get_experiment_status()
        
        print("=" * 80)
        print("EXPERIMENT STATUS REPORT")
        print("=" * 80)
        print(f"Directory: {self.experiment_dir}")
        print(f"Status: {status['status'].upper()}")
        print(f"Last updated: {datetime.fromtimestamp(status['last_updated']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Progress information
        progress = status.get("progress", {})
        if progress:
            print(f"\nPROGRESS:")
            
            if "progress_percentage" in progress:
                print(f"  Overall: {progress['progress_percentage']:.1f}%")
                print(f"  Walks: {progress.get('completed_items', 0):,} / {progress.get('total_items', 0):,}")
            
            if "context_progress_percentage" in progress:
                print(f"  Contexts: {progress['context_progress_percentage']:.1f}%")
                print(f"  Contexts: {progress.get('completed_contexts', 0)} / {progress.get('total_contexts_target', 0)}")
            
            if "completed_batches" in progress:
                print(f"  Batches: {progress['completed_batches']}")
        
        # Performance information
        performance = status.get("performance", {})
        if performance:
            print(f"\nPERFORMANCE:")
            
            if "avg_walks_per_second" in performance:
                print(f"  Rate: {performance['avg_walks_per_second']:.1f} walks/second")
            
            if "avg_batch_time" in performance:
                print(f"  Avg batch time: {performance['avg_batch_time']/60:.1f} minutes")
            
            if "eta_hours" in performance:
                eta = performance["eta_hours"]
                if eta < 24:
                    print(f"  ETA: {eta:.1f} hours")
                else:
                    print(f"  ETA: {eta/24:.1f} days")
        
        # Errors
        errors = status.get("errors", [])
        if errors:
            print(f"\nERRORS ({len(errors)}):")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  • {error.get('type', 'unknown')}: {error.get('message', error.get('content', 'No details'))[:100]}")
            
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        print("=" * 80)


def monitor_multiple_experiments(experiment_dirs, watch_mode=False, interval=30):
    """Monitor multiple experiments."""
    monitors = [ExperimentMonitor(exp_dir, verbose=False) for exp_dir in experiment_dirs]
    
    def print_summary():
        print("\n" + "=" * 120)
        print("MULTI-EXPERIMENT DASHBOARD")
        print("=" * 120)
        
        for i, monitor in enumerate(monitors):
            status = monitor.get_experiment_status()
            progress = status.get("progress", {})
            performance = status.get("performance", {})
            
            exp_name = os.path.basename(monitor.experiment_dir)
            status_emoji = {
                "running": "🟢",
                "completed": "✅", 
                "stalled": "🟡",
                "incomplete": "🔴",
                "not_started": "⚪",
                "not_found": "❌"
            }.get(status["status"], "❓")
            
            progress_pct = progress.get("progress_percentage", 0)
            eta_hours = performance.get("eta_hours", 0)
            rate = performance.get("avg_walks_per_second", 0)
            
            print(f"{status_emoji} {exp_name:<30} "
                  f"{status['status']:<12} "
                  f"{progress_pct:>6.1f}% "
                  f"{rate:>6.1f} w/s "
                  f"{eta_hours:>8.1f}h ETA")
    
    if watch_mode:
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
                print_summary()
                print(f"\nRefreshing every {interval} seconds... (Ctrl+C to exit)")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        print_summary()


def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor GraphVerse experiment progress")
    parser.add_argument("experiment_dirs", nargs="+",
                       help="Experiment directories to monitor")
    parser.add_argument("--watch", "-w", action="store_true",
                       help="Watch mode - continuously monitor")
    parser.add_argument("--interval", "-i", type=int, default=30,
                       help="Refresh interval for watch mode (seconds)")
    parser.add_argument("--json", action="store_true",
                       help="Output status as JSON")
    parser.add_argument("--multi", action="store_true",
                       help="Multi-experiment dashboard view")
    
    args = parser.parse_args()
    
    if not MONITORING_AVAILABLE:
        print(f"❌ Monitoring not available: {IMPORT_ERROR}")
        sys.exit(1)
    
    if args.multi or len(args.experiment_dirs) > 1:
        monitor_multiple_experiments(args.experiment_dirs, args.watch, args.interval)
    else:
        # Single experiment monitoring
        monitor = ExperimentMonitor(args.experiment_dirs[0])
        
        if args.watch:
            try:
                while True:
                    os.system('clear' if os.name == 'posix' else 'cls')
                    monitor.print_status()
                    print(f"\nRefreshing every {args.interval} seconds... (Ctrl+C to exit)")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
        else:
            status = monitor.get_experiment_status()
            
            if args.json:
                print(json.dumps(status, indent=2, default=str))
            else:
                monitor.print_status(status)


if __name__ == "__main__":
    main()