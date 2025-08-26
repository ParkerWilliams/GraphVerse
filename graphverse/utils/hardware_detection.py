"""
Hardware detection utilities for optimal parallelization strategy selection.

This module detects available hardware resources (CPU cores, GPU, memory) and 
recommends the best parallelization approach for walk generation.
"""

import multiprocessing
import sys
import subprocess
import platform
from typing import Dict, Any, Optional


class HardwareInfo:
    """Container for hardware information and capabilities."""
    
    def __init__(self):
        self.cpu_cores = multiprocessing.cpu_count()
        self.platform = platform.system()
        self.gpu_available = False
        self.gpu_type = None
        self.gpu_memory_gb = 0
        self.torch_available = False
        self.cuda_available = False
        self.metal_available = False
        self.recommended_strategy = "cpu_standard_parallel"
        
    def __str__(self):
        return f"HardwareInfo(cores={self.cpu_cores}, gpu={self.gpu_type}, strategy={self.recommended_strategy})"


def detect_pytorch_gpu_support() -> tuple[bool, bool, bool]:
    """
    Detect PyTorch availability and GPU support.
    
    Returns:
        tuple: (torch_available, cuda_available, metal_available)
    """
    try:
        import torch
        torch_available = True
        
        # Check CUDA support
        cuda_available = torch.cuda.is_available()
        
        # Check Metal support (macOS)
        metal_available = False
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            metal_available = True
            
        return torch_available, cuda_available, metal_available
        
    except ImportError:
        return False, False, False


def get_gpu_memory_info() -> tuple[Optional[str], float]:
    """
    Get GPU type and memory information.
    
    Returns:
        tuple: (gpu_type, memory_gb)
    """
    # Try NVIDIA first
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                name, memory_mb = lines[0].split(', ')
                return f"NVIDIA {name.strip()}", float(memory_mb) / 1024
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    
    # Try Metal GPU info on macOS
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and "Metal Support: Metal" in result.stdout:
                # Extract GPU name and estimate memory (Metal GPUs typically share system RAM)
                if "Apple" in result.stdout:
                    return "Apple Metal GPU", 8.0  # Conservative estimate
                else:
                    return "Metal GPU", 4.0
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    return None, 0.0


def detect_hardware_capabilities() -> HardwareInfo:
    """
    Comprehensive hardware detection for parallelization strategy selection.
    
    Returns:
        HardwareInfo: Complete hardware information and recommendations
    """
    hw_info = HardwareInfo()
    
    # PyTorch and GPU detection
    hw_info.torch_available, hw_info.cuda_available, hw_info.metal_available = detect_pytorch_gpu_support()
    
    # GPU information
    hw_info.gpu_type, hw_info.gpu_memory_gb = get_gpu_memory_info()
    hw_info.gpu_available = hw_info.gpu_type is not None
    
    # Strategy recommendation based on capabilities
    hw_info.recommended_strategy = _recommend_strategy(hw_info)
    
    return hw_info


def _recommend_strategy(hw_info: HardwareInfo) -> str:
    """
    Recommend the optimal parallelization strategy based on hardware.
    
    Args:
        hw_info: Hardware information
        
    Returns:
        str: Recommended strategy name
    """
    # GPU strategies (if PyTorch available and sufficient memory)
    if hw_info.torch_available:
        if hw_info.cuda_available and hw_info.gpu_memory_gb >= 8:
            return "gpu_cuda_accelerated"
        elif hw_info.metal_available and hw_info.gpu_memory_gb >= 4:
            return "gpu_metal_accelerated" 
        elif hw_info.cuda_available and hw_info.gpu_memory_gb >= 4:
            return "gpu_cuda_limited"  # Use smaller batches
    
    # CPU strategies based on core count
    if hw_info.cpu_cores >= 16:
        return "cpu_high_parallel"
    elif hw_info.cpu_cores >= 8:
        return "cpu_standard_parallel"
    elif hw_info.cpu_cores >= 4:
        return "cpu_medium_parallel"
    else:
        return "cpu_sequential"  # Fallback to original implementation


def get_optimal_parallelization_config(target_walks: int, graph_size: int) -> Dict[str, Any]:
    """
    Get optimal parallelization configuration for given workload.
    
    Args:
        target_walks: Number of walks to generate
        graph_size: Number of nodes in the graph
        
    Returns:
        dict: Parallelization configuration
    """
    hw_info = detect_hardware_capabilities()
    
    config = {
        "strategy": hw_info.recommended_strategy,
        "hardware_info": hw_info,
        "cpu_workers": _calculate_optimal_cpu_workers(hw_info, target_walks),
        "batch_size": _calculate_optimal_batch_size(hw_info, target_walks, graph_size),
        "memory_limit_gb": _estimate_memory_limit(hw_info),
        "use_gpu": hw_info.recommended_strategy.startswith("gpu_"),
        "fallback_enabled": True
    }
    
    return config


def _calculate_optimal_cpu_workers(hw_info: HardwareInfo, target_walks: int) -> int:
    """Calculate optimal number of CPU workers."""
    # Leave 1-2 cores for system and main thread
    available_cores = max(1, hw_info.cpu_cores - 2)
    
    # For small workloads, don't use excessive parallelization
    if target_walks < 1000:
        return min(available_cores, 2)
    elif target_walks < 10000:
        return min(available_cores, 4)
    else:
        return available_cores


def _calculate_optimal_batch_size(hw_info: HardwareInfo, target_walks: int, graph_size: int) -> int:
    """Calculate optimal batch size for processing."""
    if hw_info.recommended_strategy.startswith("gpu_"):
        # GPU batch sizes depend on memory
        if hw_info.gpu_memory_gb >= 8:
            return min(2048, max(256, target_walks // 100))
        else:
            return min(512, max(128, target_walks // 200))
    else:
        # CPU batch sizes
        base_batch = max(100, target_walks // (hw_info.cpu_cores * 10))
        return min(5000, base_batch)


def _estimate_memory_limit(hw_info: HardwareInfo) -> int:
    """Estimate reasonable memory limit in GB."""
    # Conservative estimates based on typical systems
    if hw_info.gpu_available and hw_info.gpu_memory_gb >= 8:
        return 16  # Can handle larger datasets with GPU acceleration
    elif hw_info.cpu_cores >= 16:
        return 32  # High-end CPU systems
    elif hw_info.cpu_cores >= 8:
        return 16  # Standard systems
    else:
        return 8   # Conservative for smaller systems


def print_hardware_summary(hw_info: Optional[HardwareInfo] = None, verbose: bool = True):
    """Print a summary of detected hardware and recommendations."""
    if hw_info is None:
        hw_info = detect_hardware_capabilities()
    
    if not verbose:
        print(f"Hardware: {hw_info.cpu_cores} CPU cores, {hw_info.gpu_type or 'No GPU'}")
        print(f"Strategy: {hw_info.recommended_strategy}")
        return
    
    print("=" * 60)
    print("HARDWARE DETECTION SUMMARY")
    print("=" * 60)
    
    print(f"Platform: {hw_info.platform}")
    print(f"CPU Cores: {hw_info.cpu_cores}")
    
    if hw_info.gpu_available:
        print(f"GPU: {hw_info.gpu_type}")
        print(f"GPU Memory: {hw_info.gpu_memory_gb:.1f} GB")
    else:
        print("GPU: Not available")
    
    print(f"PyTorch Available: {hw_info.torch_available}")
    if hw_info.torch_available:
        print(f"  CUDA Support: {hw_info.cuda_available}")
        print(f"  Metal Support: {hw_info.metal_available}")
    
    print(f"\nRecommended Strategy: {hw_info.recommended_strategy}")
    
    # Explain the strategy
    strategy_explanations = {
        "gpu_cuda_accelerated": "High-performance GPU acceleration with CUDA",
        "gpu_metal_accelerated": "GPU acceleration with Apple Metal",
        "gpu_cuda_limited": "GPU acceleration with memory constraints",
        "cpu_high_parallel": "High-performance CPU multiprocessing",
        "cpu_standard_parallel": "Standard CPU multiprocessing",
        "cpu_medium_parallel": "Medium CPU parallelization",
        "cpu_sequential": "Sequential processing (fallback)"
    }
    
    explanation = strategy_explanations.get(hw_info.recommended_strategy, "Unknown strategy")
    print(f"Strategy Description: {explanation}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Demo the hardware detection
    hw_info = detect_hardware_capabilities()
    print_hardware_summary(hw_info, verbose=True)
    
    # Show optimal config for different workload sizes
    print("\nOPTIMAL CONFIGURATIONS FOR DIFFERENT WORKLOADS:")
    test_scenarios = [
        (1000, 100, "Small test (1K walks, 100 nodes)"),
        (100000, 1000, "Medium scale (100K walks, 1K nodes)"),
        (1000000, 10000, "Large scale (1M walks, 10K nodes)")
    ]
    
    for walks, nodes, description in test_scenarios:
        config = get_optimal_parallelization_config(walks, nodes)
        print(f"\n{description}:")
        print(f"  Strategy: {config['strategy']}")
        print(f"  CPU Workers: {config['cpu_workers']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Memory Limit: {config['memory_limit_gb']} GB")