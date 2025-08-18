#!/usr/bin/env python3
"""
Prerequisites Checker for Large-Scale GraphVerse Experiments

Validates system requirements, dependencies, and environment setup
before running large-scale experiments.
"""

import os
import sys
import shutil
import psutil
import time
from pathlib import Path

# Add GraphVerse to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import numpy as np
    from configs.large_scale_config import LARGE_SCALE_CONFIG, validate_large_scale_config, estimate_memory_requirements
    GRAPHVERSE_AVAILABLE = True
except ImportError as e:
    GRAPHVERSE_AVAILABLE = False
    IMPORT_ERROR = str(e)


class PrerequisiteChecker:
    """Comprehensive prerequisites validation."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.checks = {}
        self.warnings = []
        self.errors = []
        
    def log(self, message, level="info"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            if level == "error":
                print(f"❌ {message}")
            elif level == "warning":
                print(f"⚠️ {message}")
            elif level == "success":
                print(f"✅ {message}")
            else:
                print(f"ℹ️ {message}")
    
    def check_python_version(self):
        """Check Python version compatibility."""
        self.log("Checking Python version...")
        
        version = sys.version_info
        required_major, required_minor = 3, 7
        
        if version.major < required_major or (version.major == required_major and version.minor < required_minor):
            self.errors.append(f"Python {required_major}.{required_minor}+ required, found {version.major}.{version.minor}")
            self.log(f"Python version {version.major}.{version.minor} is too old (need {required_major}.{required_minor}+)", "error")
            self.checks["python_version"] = False
        else:
            self.log(f"Python {version.major}.{version.minor}.{version.micro}", "success")
            self.checks["python_version"] = True
    
    def check_system_resources(self):
        """Check system memory, CPU, and disk space."""
        self.log("Checking system resources...")
        
        # Memory check
        memory_gb = psutil.virtual_memory().total / 1024**3
        recommended_memory = 32
        
        if memory_gb < recommended_memory:
            self.warnings.append(f"Low system memory: {memory_gb:.1f}GB (recommended: {recommended_memory}GB)")
            self.log(f"System memory: {memory_gb:.1f}GB (recommended: {recommended_memory}GB)", "warning")
        else:
            self.log(f"System memory: {memory_gb:.1f}GB", "success")
        
        self.checks["memory_gb"] = memory_gb
        
        # CPU check
        cpu_count = os.cpu_count()
        self.log(f"CPU cores: {cpu_count}")
        self.checks["cpu_cores"] = cpu_count
        
        # Disk space check
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / 1024**3
        required_disk = 100  # GB
        
        if free_gb < required_disk:
            self.errors.append(f"Insufficient disk space: {free_gb:.1f}GB free (need {required_disk}GB)")
            self.log(f"Disk space: {free_gb:.1f}GB free (need {required_disk}GB)", "error")
            self.checks["disk_space"] = False
        else:
            self.log(f"Disk space: {free_gb:.1f}GB free", "success")
            self.checks["disk_space"] = True
    
    def check_dependencies(self):
        """Check required Python packages."""
        self.log("Checking dependencies...")
        
        required_packages = [
            ("torch", "PyTorch"),
            ("numpy", "NumPy"),
            ("tqdm", "tqdm"),
            ("psutil", "psutil"),
            ("scipy", "SciPy")
        ]
        
        missing_packages = []
        
        for package, name in required_packages:
            try:
                __import__(package)
                self.log(f"{name} available", "success")
            except ImportError:
                missing_packages.append(name)
                self.log(f"{name} missing", "error")
        
        if missing_packages:
            self.errors.append(f"Missing packages: {', '.join(missing_packages)}")
            self.checks["dependencies"] = False
        else:
            self.checks["dependencies"] = True
    
    def check_graphverse_installation(self):
        """Check GraphVerse modules are available."""
        self.log("Checking GraphVerse installation...")
        
        if not GRAPHVERSE_AVAILABLE:
            self.errors.append(f"GraphVerse import failed: {IMPORT_ERROR}")
            self.log("GraphVerse modules not available", "error")
            self.checks["graphverse"] = False
            return
        
        # Test key imports
        try:
            from graphverse.graph.base import Graph
            from graphverse.graph.rules import RepeaterRule
            from graphverse.llm.model import WalkTransformer
            from graphverse.llm.large_scale_evaluation import LargeScaleEvaluator
            from graphverse.utils.memory_monitor import MemoryMonitor
            
            self.log("GraphVerse modules available", "success")
            self.checks["graphverse"] = True
            
        except ImportError as e:
            self.errors.append(f"GraphVerse module import failed: {e}")
            self.log(f"GraphVerse import error: {e}", "error")
            self.checks["graphverse"] = False
    
    def check_gpu_availability(self):
        """Check GPU availability for accelerated training."""
        self.log("Checking GPU availability...")
        
        if not torch:
            self.log("PyTorch not available, skipping GPU check", "warning")
            self.checks["gpu"] = False
            return
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
            
            self.log(f"GPU available: {gpu_name} ({gpu_memory:.1f}GB)", "success")
            self.checks["gpu"] = True
            self.checks["gpu_memory_gb"] = gpu_memory
            self.checks["gpu_count"] = gpu_count
        else:
            self.log("No GPU available (will use CPU)", "warning")
            self.checks["gpu"] = False
    
    def check_large_scale_config(self):
        """Check large-scale configuration validity."""
        if not GRAPHVERSE_AVAILABLE:
            return
        
        self.log("Checking large-scale configuration...")
        
        try:
            validation = validate_large_scale_config(LARGE_SCALE_CONFIG)
            memory_est = estimate_memory_requirements(LARGE_SCALE_CONFIG)
            
            if validation["valid"]:
                self.log("Large-scale configuration valid", "success")
                self.checks["config_valid"] = True
                
                # Check memory requirements vs available
                estimated_gb = memory_est["total_estimated"]
                available_gb = self.checks.get("memory_gb", 0)
                
                if estimated_gb > available_gb * 0.8:  # Use 80% of available memory
                    self.warnings.append(f"High memory usage: {estimated_gb:.1f}GB estimated vs {available_gb:.1f}GB available")
                    self.log(f"Memory usage: {estimated_gb:.1f}GB estimated vs {available_gb:.1f}GB available", "warning")
                else:
                    self.log(f"Memory requirements: {estimated_gb:.1f}GB (within limits)", "success")
                
                self.checks["estimated_memory_gb"] = estimated_gb
                self.checks["estimated_runtime_hours"] = validation["estimated_runtime_hours"]
                
            else:
                self.errors.append("Large-scale configuration validation failed")
                self.log("Configuration validation failed", "error")
                self.checks["config_valid"] = False
                
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    self.warnings.append(f"Config: {warning}")
                    self.log(f"Config warning: {warning}", "warning")
                    
        except Exception as e:
            self.errors.append(f"Configuration check failed: {e}")
            self.log(f"Configuration check error: {e}", "error")
            self.checks["config_valid"] = False
    
    def check_file_paths(self, graph_path=None, models_dir=None):
        """Check if required files exist."""
        self.log("Checking file paths...")
        
        paths_to_check = []
        
        if graph_path:
            paths_to_check.extend([
                (f"{graph_path}.npy", "Graph adjacency matrix"),
                (f"{graph_path}_attrs.json", "Graph attributes"),
                (f"{graph_path}_rules.json", "Graph rules")
            ])
        
        if models_dir and os.path.exists(models_dir):
            context_windows = LARGE_SCALE_CONFIG.get("context_windows", []) if GRAPHVERSE_AVAILABLE else []
            for ctx in context_windows:
                paths_to_check.extend([
                    (os.path.join(models_dir, f"model_ctx_{ctx}.pt"), f"Model for context {ctx}"),
                    (os.path.join(models_dir, f"vocab_ctx_{ctx}.pkl"), f"Vocab for context {ctx}")
                ])
        
        missing_files = []
        existing_files = []
        
        for path, description in paths_to_check:
            if os.path.exists(path):
                existing_files.append(description)
                self.log(f"{description}: found", "success")
            else:
                missing_files.append(description)
                self.log(f"{description}: missing", "warning")
        
        self.checks["existing_files"] = existing_files
        self.checks["missing_files"] = missing_files
        
        if missing_files:
            self.warnings.append(f"Missing files: {len(missing_files)} files not found")
    
    def run_all_checks(self, graph_path=None, models_dir=None):
        """Run all prerequisite checks."""
        if self.verbose:
            print("=" * 80)
            print("GRAPHVERSE LARGE-SCALE EXPERIMENT PREREQUISITES CHECK")
            print("=" * 80)
        
        self.check_python_version()
        self.check_system_resources()
        self.check_dependencies()
        self.check_graphverse_installation()
        self.check_gpu_availability()
        self.check_large_scale_config()
        self.check_file_paths(graph_path, models_dir)
        
        return self.get_summary()
    
    def get_summary(self):
        """Get summary of all checks."""
        summary = {
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "overall_status": "ready" if not self.errors else "blocked",
            "can_proceed": len(self.errors) == 0
        }
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("PREREQUISITES SUMMARY")
            print("=" * 80)
            
            if summary["can_proceed"]:
                print("🎉 SYSTEM READY for large-scale experiments!")
            else:
                print("❌ SYSTEM NOT READY - please fix errors below")
            
            if self.errors:
                print(f"\n❌ ERRORS ({len(self.errors)}):")
                for error in self.errors:
                    print(f"   • {error}")
            
            if self.warnings:
                print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
                for warning in self.warnings:
                    print(f"   • {warning}")
            
            print(f"\nSystem Specs:")
            print(f"   Memory: {self.checks.get('memory_gb', 0):.1f}GB")
            print(f"   CPU cores: {self.checks.get('cpu_cores', 0)}")
            print(f"   GPU: {'Yes' if self.checks.get('gpu', False) else 'No'}")
            
            if GRAPHVERSE_AVAILABLE and self.checks.get('config_valid', False):
                print(f"\nExperiment Estimates:")
                print(f"   Memory needed: {self.checks.get('estimated_memory_gb', 0):.1f}GB")
                print(f"   Runtime: {self.checks.get('estimated_runtime_hours', 0):.1f} hours")
            
            existing_files = self.checks.get('existing_files', [])
            missing_files = self.checks.get('missing_files', [])
            
            if existing_files or missing_files:
                print(f"\nFiles:")
                print(f"   Found: {len(existing_files)}")
                print(f"   Missing: {len(missing_files)}")
        
        return summary


def main():
    """Main prerequisites checking function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check prerequisites for large-scale GraphVerse experiments")
    parser.add_argument("--graph", "-g", 
                       help="Path to graph files to check (without extension)")
    parser.add_argument("--models", "-m",
                       help="Directory containing trained models to check")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Run prerequisites check
    checker = PrerequisiteChecker(verbose=verbose)
    summary = checker.run_all_checks(graph_path=args.graph, models_dir=args.models)
    
    # Output results
    if args.json:
        import json
        print(json.dumps(summary, indent=2, default=str))
    
    # Exit with appropriate code
    if summary["can_proceed"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()