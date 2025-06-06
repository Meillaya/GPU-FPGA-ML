"""Base Kernel Interface for ML Operations"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import time
from dataclasses import dataclass


@dataclass
class KernelMetrics:
    """Performance metrics for kernel execution"""
    execution_time: float
    memory_used: int
    peak_memory: int
    flops: Optional[int] = None
    memory_bandwidth: Optional[float] = None
    
    
class BaseKernel(ABC):
    """Abstract base class for all ML kernels"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.metrics: Optional[KernelMetrics] = None
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the kernel"""
        pass
    
    @abstractmethod
    def get_memory_footprint(self, *args, **kwargs) -> int:
        """Calculate memory footprint for given inputs"""
        pass
    
    @abstractmethod
    def get_flop_count(self, *args, **kwargs) -> int:
        """Calculate theoretical FLOP count"""
        pass
    
    def benchmark(self, *args, warmup_runs: int = 10, 
                  benchmark_runs: int = 100, **kwargs) -> KernelMetrics:
        """Benchmark the kernel performance"""
        # Warmup
        for _ in range(warmup_runs):
            _ = self.forward(*args, **kwargs)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_memory = torch.cuda.memory_allocated()
        peak_memory = start_memory
        
        start_time = time.perf_counter()
        
        for _ in range(benchmark_runs):
            _ = self.forward(*args, **kwargs)
            current_memory = torch.cuda.memory_allocated()
            peak_memory = max(peak_memory, current_memory)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) / benchmark_runs
        memory_used = torch.cuda.memory_allocated() - start_memory
        flops = self.get_flop_count(*args, **kwargs)
        
        self.metrics = KernelMetrics(
            execution_time=execution_time,
            memory_used=memory_used,
            peak_memory=peak_memory,
            flops=flops,
            memory_bandwidth=memory_used / execution_time if execution_time > 0 else 0
        )
        
        return self.metrics
    
    def profile_memory_access(self, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory access patterns"""
        return {
            "global_loads": 0,
            "global_stores": 0, 
            "shared_loads": 0,
            "shared_stores": 0,
            "cache_hit_rate": 0.0
        }
    
    def get_kernel_info(self) -> Dict[str, Any]:
        """Get kernel implementation details"""
        return {
            "name": self.__class__.__name__,
            "device": str(self.device),
            "implementation": "base",
            "memory_footprint": 0,
            "compute_intensity": 0.0
        } 