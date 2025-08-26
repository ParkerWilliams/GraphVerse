"""
Parallel walk generation utilities for high-performance random walk sampling.

This module provides CPU multiprocessing and GPU-accelerated implementations
for generating large numbers of random walks efficiently.
"""

import random
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import numpy as np

from .walk import generate_valid_walk, check_rule_compliance
from ..utils.hardware_detection import detect_hardware_capabilities, get_optimal_parallelization_config


def generate_walks_worker(args: Tuple) -> Tuple[List[List[int]], int, int]:
    """
    Worker function for multiprocessing walk generation.
    
    Args:
        args: Tuple containing (graph, target_walks, min_length, max_length, 
                               rules, worker_id, random_seed)
    
    Returns:
        tuple: (generated_walks, successful_walks, failed_attempts)
    """
    graph, target_walks, min_length, max_length, rules, worker_id, random_seed = args
    
    # Set unique random seed for this worker
    random.seed(random_seed + worker_id)
    np.random.seed(random_seed + worker_id)
    
    walks = []
    failed_attempts = 0
    max_attempts_per_walk = 10
    
    attempts = 0
    while len(walks) < target_walks:
        start_vertex = random.choice(list(range(graph.n)))
        walk = generate_valid_walk(
            graph, start_vertex, min_length, max_length, 
            rules, max_attempts_per_walk, verbose=False
        )
        
        if walk:
            walks.append(walk)
            attempts = 0
        else:
            failed_attempts += 1
            attempts += 1
            
            # Prevent infinite loops in problematic configurations
            if attempts >= max_attempts_per_walk:
                attempts = 0
                # Try a different starting vertex
                continue
    
    return walks, len(walks), failed_attempts


def generate_multiple_walks_parallel(
    graph, 
    num_walks: int, 
    min_length: int, 
    max_length: int, 
    rules, 
    verbose: bool = False,
    n_workers: Optional[int] = None,
    strategy: str = "auto",
    random_seed: int = 42,
    device: Optional[str] = None
) -> List[List[int]]:
    """
    Generate multiple walks using parallel processing.
    
    Args:
        graph: Graph object to walk on
        num_walks: Total number of walks to generate
        min_length: Minimum walk length
        max_length: Maximum walk length  
        rules: Rules to follow during walk generation
        verbose: Whether to show progress
        n_workers: Number of worker processes (None = auto-detect)
        strategy: Parallelization strategy ("auto", "cpu_only", "gpu_preferred")
        random_seed: Random seed for reproducibility
        device: GPU device for GPU strategies (None = auto-detect)
        
    Returns:
        List of generated walks
    """
    if strategy == "auto":
        # Auto-detect optimal strategy
        config = get_optimal_parallelization_config(num_walks, graph.n)
        strategy = config["strategy"] 
        if n_workers is None:
            n_workers = config["cpu_workers"]
    
    # Try GPU strategies first if requested
    if strategy.startswith("gpu_") or strategy == "gpu_preferred":
        try:
            from .gpu_walk import generate_walks_gpu_accelerated
            
            if verbose:
                print(f"  Attempting GPU-accelerated walk generation...")
            
            return generate_walks_gpu_accelerated(
                graph, num_walks, min_length, max_length, rules,
                device=device, verbose=verbose
            )
            
        except ImportError:
            if verbose:
                print(f"  ⚠ PyTorch not available, falling back to CPU parallel processing")
        except Exception as e:
            if verbose:
                print(f"  ⚠ GPU generation failed ({e}), falling back to CPU parallel processing")
        
        # Fallback to CPU parallel
        strategy = "cpu_standard_parallel"
    
    return _generate_walks_cpu_parallel(
        graph, num_walks, min_length, max_length, rules,
        verbose, n_workers, random_seed
    )


def _generate_walks_cpu_parallel(
    graph, 
    num_walks: int, 
    min_length: int, 
    max_length: int, 
    rules,
    verbose: bool,
    n_workers: Optional[int],
    random_seed: int
) -> List[List[int]]:
    """CPU multiprocessing implementation."""
    
    # Determine number of workers
    if n_workers is None:
        hw_info = detect_hardware_capabilities()
        n_workers = max(1, hw_info.cpu_cores - 2)  # Leave 2 cores for system
    
    n_workers = min(n_workers, num_walks)  # Don't use more workers than walks
    
    if verbose:
        print(f"\n  Generating {num_walks} walks using {n_workers} CPU workers...")
        print(f"  Walk length: {min_length}-{max_length}")
    
    # For small workloads, use sequential processing
    if num_walks < 100 or n_workers == 1:
        if verbose:
            print("  Using sequential processing for small workload")
        from .walk import generate_multiple_walks
        return generate_multiple_walks(graph, num_walks, min_length, max_length, rules, verbose)
    
    # Distribute work across workers
    walks_per_worker = num_walks // n_workers
    remaining_walks = num_walks % n_workers
    
    # Create work assignments
    work_args = []
    for worker_id in range(n_workers):
        worker_walks = walks_per_worker + (1 if worker_id < remaining_walks else 0)
        if worker_walks > 0:
            work_args.append((
                graph, worker_walks, min_length, max_length, 
                rules, worker_id, random_seed
            ))
    
    if verbose:
        print(f"  Work distribution: {walks_per_worker} walks per worker (+{remaining_walks} extra)")
        print(f"  Starting {len(work_args)} parallel workers...")
    
    # Execute parallel processing with progress tracking
    all_walks = []
    total_failed = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_worker = {executor.submit(generate_walks_worker, args): i 
                           for i, args in enumerate(work_args)}
        
        if verbose:
            progress_bar = tqdm(total=num_walks, desc="Generating walks", unit="walk")
        
        # Collect results as they complete
        for future in as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                walks, successful, failed = future.result()
                all_walks.extend(walks)
                total_failed += failed
                
                if verbose:
                    progress_bar.update(successful)
                    success_rate = len(all_walks) / (len(all_walks) + total_failed) if total_failed > 0 else 1.0
                    progress_bar.set_postfix({
                        "workers_done": f"{len([f for f in future_to_worker if f.done()])}/{len(work_args)}", 
                        "success_rate": f"{success_rate:.1%}"
                    })
                    
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Worker {worker_id} failed: {e}")
                # Continue with other workers
    
    if verbose:
        progress_bar.close()
        
    duration = time.time() - start_time
    
    if verbose:
        print(f"  ✓ {len(all_walks)} walks generated successfully in {duration:.1f} seconds")
        if total_failed > 0:
            print(f"  ⚠ {total_failed} failed attempts ({total_failed/(len(all_walks)+total_failed):.1%} failure rate)")
        
        walks_per_second = len(all_walks) / duration
        print(f"  📊 Performance: {walks_per_second:.1f} walks/second")
        
        # Show performance comparison
        sequential_estimate = len(all_walks) / (walks_per_second / n_workers)
        speedup = sequential_estimate / duration
        print(f"  🚀 Estimated speedup: {speedup:.1f}x vs sequential")
    
    return all_walks


def generate_per_node_walks_parallel(
    graph, 
    min_length: int, 
    max_length: int, 
    rules,
    verbose: bool = False,
    n_workers: Optional[int] = None
) -> List[List[int]]:
    """
    Generate one walk starting from each node in parallel.
    
    Args:
        graph: Graph object
        min_length: Minimum walk length
        max_length: Maximum walk length
        rules: Rules to follow
        verbose: Show progress
        n_workers: Number of workers (None = auto-detect)
        
    Returns:
        List of walks (one per node that succeeded)
    """
    if n_workers is None:
        hw_info = detect_hardware_capabilities()
        n_workers = max(1, hw_info.cpu_cores - 1)
    
    n_workers = min(n_workers, graph.n)  # Don't use more workers than nodes
    
    if verbose:
        print(f"\n  Generating per-node walks for {graph.n} nodes using {n_workers} workers...")
    
    # For small graphs, use sequential processing
    if graph.n < 50 or n_workers == 1:
        if verbose:
            print("  Using sequential processing for small graph")
        walks = []
        node_iterator = tqdm(range(graph.n), desc="Per-node walks", unit="node") if verbose else range(graph.n)
        
        for node in node_iterator:
            walk = generate_valid_walk(graph, node, min_length, max_length, rules, verbose=False)
            if walk:
                walks.append(walk)
        
        return walks
    
    # Distribute nodes across workers
    nodes_per_worker = graph.n // n_workers
    remaining_nodes = graph.n % n_workers
    
    work_args = []
    start_node = 0
    for worker_id in range(n_workers):
        worker_nodes = nodes_per_worker + (1 if worker_id < remaining_nodes else 0)
        if worker_nodes > 0:
            node_range = list(range(start_node, start_node + worker_nodes))
            work_args.append((graph, node_range, min_length, max_length, rules, worker_id))
            start_node += worker_nodes
    
    # Execute parallel processing
    all_walks = []
    successful_nodes = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_worker = {executor.submit(_generate_per_node_walks_worker, args): i 
                           for i, args in enumerate(work_args)}
        
        if verbose:
            progress_bar = tqdm(total=graph.n, desc="Per-node walks", unit="node")
        
        for future in as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                walks, nodes_processed = future.result()
                all_walks.extend(walks)
                successful_nodes += len(walks)
                
                if verbose:
                    progress_bar.update(nodes_processed)
                    progress_bar.set_postfix({
                        "success_rate": f"{successful_nodes}/{progress_bar.n:.0f}" if progress_bar.n > 0 else "0"
                    })
                    
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Worker {worker_id} failed: {e}")
    
    if verbose:
        progress_bar.close()
        print(f"  ✓ {len(all_walks)} successful walks from {graph.n} nodes")
    
    return all_walks


def _generate_per_node_walks_worker(args: Tuple) -> Tuple[List[List[int]], int]:
    """Worker for per-node walk generation."""
    graph, node_range, min_length, max_length, rules, worker_id = args
    
    walks = []
    for node in node_range:
        walk = generate_valid_walk(graph, node, min_length, max_length, rules, verbose=False)
        if walk:
            walks.append(walk)
    
    return walks, len(node_range)


# Compatibility function - drop-in replacement for original generate_multiple_walks
def generate_multiple_walks_auto(
    graph, num_walks, min_length, max_length, rules, verbose=False
):
    """
    Auto-selecting version of generate_multiple_walks.
    
    This function automatically chooses between sequential and parallel processing
    based on workload size and available hardware.
    """
    # For very small workloads, use original sequential implementation
    if num_walks < 100:
        from .walk import generate_multiple_walks
        return generate_multiple_walks(graph, num_walks, min_length, max_length, rules, verbose)
    
    # For larger workloads, use parallel implementation
    return generate_multiple_walks_parallel(
        graph, num_walks, min_length, max_length, rules, verbose
    )