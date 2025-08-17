"""
Memory monitoring utilities for large-scale GraphVerse experiments.
Provides real-time memory tracking, alerts, and optimization suggestions.
"""

import os
import gc
import time
import psutil
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
import numpy as np


class MemoryMonitor:
    """
    Real-time memory monitoring with alerts and automatic cleanup.
    """
    
    def __init__(self, 
                 memory_limit_gb: float = 32.0,
                 alert_threshold: float = 0.8,
                 critical_threshold: float = 0.95,
                 check_interval: float = 10.0):
        """
        Initialize memory monitor.
        
        Args:
            memory_limit_gb: Memory limit in GB
            alert_threshold: Fraction of limit to trigger alerts (0.8 = 80%)
            critical_threshold: Fraction of limit to trigger critical actions
            check_interval: Seconds between memory checks
        """
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        
        # State tracking
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history = []
        self.alert_callbacks = []
        self.critical_callbacks = []
        
        # Statistics
        self.peak_memory = 0
        self.alert_count = 0
        self.critical_count = 0
        self.start_time = None
        
    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """Add callback function to call when memory alert is triggered."""
        self.alert_callbacks.append(callback)
    
    def add_critical_callback(self, callback: Callable[[Dict], None]):
        """Add callback function to call when critical memory threshold is reached."""
        self.critical_callbacks.append(callback)
    
    def start_monitoring(self, verbose: bool = True):
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(verbose,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        if verbose:
            print(f"Memory monitoring started:")
            print(f"  Limit: {self.memory_limit_bytes/1024**3:.1f} GB")
            print(f"  Alert threshold: {self.alert_threshold:.0%}")
            print(f"  Critical threshold: {self.critical_threshold:.0%}")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self, verbose: bool):
        """Main monitoring loop (runs in background thread)."""
        while self.monitoring:
            try:
                memory_info = self.get_memory_info()
                self.memory_history.append(memory_info)
                
                # Keep only recent history (last hour)
                max_history = int(3600 / self.check_interval)
                if len(self.memory_history) > max_history:
                    self.memory_history = self.memory_history[-max_history:]
                
                # Check thresholds
                usage_fraction = memory_info['used_memory'] / self.memory_limit_bytes
                
                if usage_fraction >= self.critical_threshold:
                    self.critical_count += 1
                    if verbose:
                        print(f"⚠️ CRITICAL MEMORY: {usage_fraction:.1%} of limit ({memory_info['used_memory']/1024**3:.1f} GB)")
                    
                    # Trigger critical callbacks
                    for callback in self.critical_callbacks:
                        try:
                            callback(memory_info)
                        except Exception as e:
                            print(f"Error in critical callback: {e}")
                
                elif usage_fraction >= self.alert_threshold:
                    self.alert_count += 1
                    if verbose and self.alert_count % 6 == 1:  # Print every minute at 10s intervals
                        print(f"⚠️ Memory alert: {usage_fraction:.1%} of limit ({memory_info['used_memory']/1024**3:.1f} GB)")
                    
                    # Trigger alert callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(memory_info)
                        except Exception as e:
                            print(f"Error in alert callback: {e}")
                
                # Update peak memory
                self.peak_memory = max(self.peak_memory, memory_info['used_memory'])
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                if verbose:
                    print(f"Error in memory monitoring: {e}")
                time.sleep(self.check_interval)
    
    def get_memory_info(self) -> Dict:
        """Get current memory information."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        return {
            'timestamp': time.time(),
            'used_memory': memory_info.rss,  # Resident Set Size
            'virtual_memory': memory_info.vms,  # Virtual Memory Size
            'system_total': system_memory.total,
            'system_available': system_memory.available,
            'system_percent': system_memory.percent,
            'memory_fraction': memory_info.rss / self.memory_limit_bytes
        }
    
    def get_memory_summary(self) -> Dict:
        """Get summary of memory usage during monitoring."""
        if not self.memory_history:
            return {}
        
        current = self.get_memory_info()
        runtime = time.time() - self.start_time if self.start_time else 0
        
        memory_values = [m['used_memory'] for m in self.memory_history]
        
        return {
            'current_memory_gb': current['used_memory'] / 1024**3,
            'peak_memory_gb': self.peak_memory / 1024**3,
            'memory_limit_gb': self.memory_limit_bytes / 1024**3,
            'peak_fraction': self.peak_memory / self.memory_limit_bytes,
            'alert_count': self.alert_count,
            'critical_count': self.critical_count,
            'runtime_hours': runtime / 3600,
            'avg_memory_gb': np.mean(memory_values) / 1024**3,
            'memory_trend': self._calculate_memory_trend()
        }
    
    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend."""
        if len(self.memory_history) < 10:
            return "insufficient_data"
        
        recent = self.memory_history[-10:]
        early = self.memory_history[:10]
        
        recent_avg = np.mean([m['used_memory'] for m in recent])
        early_avg = np.mean([m['used_memory'] for m in early])
        
        change = (recent_avg - early_avg) / early_avg
        
        if change > 0.1:
            return "increasing"
        elif change < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def force_garbage_collection(self):
        """Force garbage collection and return memory freed."""
        before = self.get_memory_info()['used_memory']
        gc.collect()
        after = self.get_memory_info()['used_memory']
        freed = before - after
        return freed


class MemoryOptimizer:
    """
    Provides memory optimization strategies for large-scale experiments.
    """
    
    @staticmethod
    def estimate_trajectory_memory(num_walks: int, vocab_size: int, avg_steps: int, sample_rate: float = 1.0) -> Dict:
        """
        Estimate memory usage for trajectory storage.
        
        Args:
            num_walks: Number of walks
            vocab_size: Size of vocabulary
            avg_steps: Average steps per walk
            sample_rate: Sampling rate for full trajectories
            
        Returns:
            Memory estimates in bytes
        """
        # Each probability distribution: vocab_size * 4 bytes (float32)
        prob_dist_size = vocab_size * 4
        
        # Full trajectory memory
        full_walks = int(num_walks * sample_rate)
        full_trajectory_memory = full_walks * avg_steps * prob_dist_size
        
        # Summary memory (all walks)
        summary_memory_per_walk = 1024  # ~1KB for summary statistics
        summary_memory = num_walks * summary_memory_per_walk
        
        # Additional metadata
        metadata_memory = num_walks * 512  # ~0.5KB per walk for basic metadata
        
        total_memory = full_trajectory_memory + summary_memory + metadata_memory
        
        return {
            'full_trajectory_memory': full_trajectory_memory,
            'summary_memory': summary_memory,
            'metadata_memory': metadata_memory,
            'total_memory': total_memory,
            'memory_per_walk': total_memory / num_walks,
            'full_walks': full_walks,
            'sample_rate': sample_rate
        }
    
    @staticmethod
    def recommend_sample_rate(num_walks: int, memory_limit_gb: float, vocab_size: int = 1000, avg_steps: int = 20) -> Dict:
        """
        Recommend optimal sampling rate for given memory constraints.
        
        Args:
            num_walks: Number of walks
            memory_limit_gb: Available memory in GB
            vocab_size: Size of vocabulary
            avg_steps: Average steps per walk
            
        Returns:
            Recommendations
        """
        memory_limit_bytes = memory_limit_gb * 1024**3
        
        # Try different sample rates
        sample_rates = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        recommendations = []
        
        for rate in sample_rates:
            estimate = MemoryOptimizer.estimate_trajectory_memory(num_walks, vocab_size, avg_steps, rate)
            
            # Add safety factor for other memory usage
            total_with_safety = estimate['total_memory'] * 1.5
            
            if total_with_safety <= memory_limit_bytes:
                recommendations.append({
                    'sample_rate': rate,
                    'estimated_memory_gb': estimate['total_memory'] / 1024**3,
                    'memory_with_safety_gb': total_with_safety / 1024**3,
                    'full_trajectories': estimate['full_walks'],
                    'fits_in_memory': True
                })
        
        if not recommendations:
            # Even the smallest rate doesn't fit
            smallest_rate = sample_rates[-1]
            estimate = MemoryOptimizer.estimate_trajectory_memory(num_walks, vocab_size, avg_steps, smallest_rate)
            recommendations.append({
                'sample_rate': smallest_rate,
                'estimated_memory_gb': estimate['total_memory'] / 1024**3,
                'memory_with_safety_gb': estimate['total_memory'] * 1.5 / 1024**3,
                'full_trajectories': estimate['full_walks'],
                'fits_in_memory': False,
                'warning': 'May require batch processing or reduced walks'
            })
        
        return {
            'memory_limit_gb': memory_limit_gb,
            'num_walks': num_walks,
            'recommendations': recommendations,
            'best_rate': recommendations[0]['sample_rate'] if recommendations else 0.001
        }
    
    @staticmethod
    def suggest_batch_size(num_walks: int, memory_limit_gb: float, sample_rate: float = 0.02) -> Dict:
        """
        Suggest optimal batch size for processing.
        
        Args:
            num_walks: Total number of walks
            memory_limit_gb: Available memory in GB
            sample_rate: Trajectory sampling rate
            
        Returns:
            Batch size recommendations
        """
        memory_limit_bytes = memory_limit_gb * 1024**3
        
        # Estimate memory per walk (including model and processing overhead)
        base_memory_per_walk = 50 * 1024  # 50KB base processing
        trajectory_memory_per_walk = 20000 * 4 * sample_rate  # Assuming avg 20 steps, vocab 1000
        total_memory_per_walk = base_memory_per_walk + trajectory_memory_per_walk
        
        # Calculate batch size with safety factor
        safe_memory = memory_limit_bytes * 0.7  # Use 70% of limit
        batch_size = int(safe_memory / total_memory_per_walk)
        
        # Round to reasonable values
        if batch_size >= 100000:
            batch_size = 100000
        elif batch_size >= 50000:
            batch_size = 50000
        elif batch_size >= 10000:
            batch_size = 10000
        elif batch_size >= 1000:
            batch_size = 1000
        else:
            batch_size = max(100, batch_size)
        
        num_batches = (num_walks + batch_size - 1) // batch_size
        
        return {
            'recommended_batch_size': batch_size,
            'num_batches': num_batches,
            'memory_per_batch_gb': (batch_size * total_memory_per_walk) / 1024**3,
            'estimated_peak_memory_gb': (batch_size * total_memory_per_walk + 2*1024**3) / 1024**3,  # +2GB for model/overhead
            'fits_in_memory': (batch_size * total_memory_per_walk + 2*1024**3) <= memory_limit_bytes
        }


def create_memory_efficient_callbacks():
    """Create standard memory management callbacks."""
    
    def alert_callback(memory_info: Dict):
        """Standard alert callback - trigger garbage collection."""
        freed = gc.collect()
        if freed > 0:
            print(f"Garbage collection freed {freed} objects")
    
    def critical_callback(memory_info: Dict):
        """Critical callback - aggressive cleanup."""
        print("CRITICAL MEMORY: Triggering aggressive cleanup")
        
        # Force garbage collection multiple times
        for i in range(3):
            freed = gc.collect()
            if freed > 0:
                print(f"GC round {i+1}: freed {freed} objects")
        
        # Suggest actions
        print("Consider:")
        print("  - Reducing batch size")
        print("  - Lowering trajectory sampling rate")
        print("  - Saving checkpoint and restarting")
    
    return alert_callback, critical_callback


def monitor_large_scale_experiment(
    experiment_function: Callable,
    memory_limit_gb: float = 32.0,
    *args, **kwargs
):
    """
    Wrapper to monitor memory usage during large-scale experiments.
    
    Args:
        experiment_function: Function to run with monitoring
        memory_limit_gb: Memory limit in GB
        *args, **kwargs: Arguments to pass to experiment function
        
    Returns:
        Result of experiment function and memory summary
    """
    # Set up monitoring
    monitor = MemoryMonitor(memory_limit_gb=memory_limit_gb)
    alert_callback, critical_callback = create_memory_efficient_callbacks()
    monitor.add_alert_callback(alert_callback)
    monitor.add_critical_callback(critical_callback)
    
    # Start monitoring
    monitor.start_monitoring(verbose=True)
    
    try:
        # Run experiment
        print(f"Starting monitored experiment...")
        result = experiment_function(*args, **kwargs)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Get summary
        memory_summary = monitor.get_memory_summary()
        print(f"\nMemory usage summary:")
        print(f"  Peak memory: {memory_summary['peak_memory_gb']:.1f} GB")
        print(f"  Memory utilization: {memory_summary['peak_fraction']:.1%}")
        print(f"  Alerts triggered: {memory_summary['alert_count']}")
        print(f"  Critical events: {memory_summary['critical_count']}")
        
        return result, memory_summary
        
    except Exception as e:
        monitor.stop_monitoring()
        print(f"Experiment failed: {e}")
        memory_summary = monitor.get_memory_summary()
        raise e
    
    finally:
        # Ensure monitoring is stopped
        monitor.stop_monitoring()


# Example usage and utilities
if __name__ == "__main__":
    # Example: Analyze memory requirements for large experiment
    num_walks = 1000000
    memory_limit = 32  # GB
    
    print("Memory Analysis for Large-Scale Experiment")
    print("=" * 50)
    
    # Get recommendations
    recommendations = MemoryOptimizer.recommend_sample_rate(
        num_walks=num_walks,
        memory_limit_gb=memory_limit
    )
    
    print(f"For {num_walks:,} walks with {memory_limit}GB memory limit:")
    print(f"Recommended sampling rate: {recommendations['best_rate']:.1%}")
    
    for rec in recommendations['recommendations'][:3]:
        print(f"  Rate {rec['sample_rate']:.1%}: {rec['estimated_memory_gb']:.1f}GB, {rec['full_trajectories']:,} full trajectories")
    
    # Batch size recommendation
    batch_rec = MemoryOptimizer.suggest_batch_size(
        num_walks=num_walks,
        memory_limit_gb=memory_limit,
        sample_rate=recommendations['best_rate']
    )
    
    print(f"\nBatch processing recommendation:")
    print(f"  Batch size: {batch_rec['recommended_batch_size']:,}")
    print(f"  Number of batches: {batch_rec['num_batches']}")
    print(f"  Peak memory per batch: {batch_rec['estimated_peak_memory_gb']:.1f}GB")