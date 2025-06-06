"""GPU and Memory Profiling Tools

Advanced profiling capabilities for hardware-specific debugging
and performance optimization of ML kernels.
"""

import torch
import torch.profiler
import time
import psutil
import GPUtil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from contextlib import contextmanager

try:
    import py3nvml.py3nvml as nvml
    nvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, Exception):
    NVML_AVAILABLE = False
    nvml = None


@dataclass
class ProfileResult:
    """Container for profiling results"""
    kernel_name: str
    execution_time: float
    memory_used: int
    peak_memory: int
    gpu_utilization: float
    memory_bandwidth: float
    compute_utilization: float
    cache_hit_rate: Optional[float] = None
    flops: Optional[int] = None
    energy_consumption: Optional[float] = None


@dataclass
class MemoryProfile:
    """Memory usage profile"""
    allocated: int
    reserved: int
    free: int
    total: int
    fragmentation: float
    
    @property
    def utilization(self) -> float:
        return self.allocated / self.total if self.total > 0 else 0.0


class GPUProfiler:
    """Comprehensive GPU profiling for ML kernels"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.device_id = int(device.split(":")[-1]) if ":" in device else 0
        
        # Initialize CUDA events for precise timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
        # GPU info
        self.gpu_info = self._get_gpu_info()
        
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU hardware information"""
        gpu_info = {
            "name": torch.cuda.get_device_name(self.device_id),
            "compute_capability": torch.cuda.get_device_capability(self.device_id),
            "total_memory": torch.cuda.get_device_properties(self.device_id).total_memory,
            "multiprocessor_count": torch.cuda.get_device_properties(self.device_id).multi_processor_count,
        }
        
        if NVML_AVAILABLE:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(self.device_id)
                gpu_info.update({
                    "driver_version": nvml.nvmlSystemGetDriverVersion(),
                    "memory_bus_width": nvml.nvmlDeviceGetMemoryBusWidth(handle),
                    "memory_clock": nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_MEM),
                    "sm_clock": nvml.nvmlDeviceGetMaxClockInfo(handle, nvml.NVML_CLOCK_SM),
                })
            except Exception as e:
                print(f"Warning: Could not get detailed GPU info: {e}")
        
        return gpu_info
    
    def profile_kernel(self, kernel_func: Callable, *args, 
                      warmup_runs: int = 10, benchmark_runs: int = 100,
                      **kwargs) -> ProfileResult:
        """Profile a kernel function with comprehensive metrics"""
        
        # Warmup
        for _ in range(warmup_runs):
            kernel_func(*args, **kwargs)
        torch.cuda.synchronize()
        
        # Reset memory statistics
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Start profiling
        start_memory = torch.cuda.memory_allocated(self.device)
        
        with self._gpu_utilization_monitor() as gpu_monitor:
            # Precise timing with CUDA events
            self.start_event.record()
            
            for _ in range(benchmark_runs):
                result = kernel_func(*args, **kwargs)
            
            self.end_event.record()
            torch.cuda.synchronize()
            
            execution_time = self.start_event.elapsed_time(self.end_event) / 1000.0  # Convert to seconds
            execution_time /= benchmark_runs
        
        # Memory statistics
        end_memory = torch.cuda.memory_allocated(self.device)
        peak_memory = torch.cuda.max_memory_allocated(self.device)
        memory_used = end_memory - start_memory
        
        # GPU utilization
        gpu_util = gpu_monitor.get_average_utilization()
        
        # Calculate memory bandwidth
        memory_bandwidth = self._calculate_memory_bandwidth(
            memory_used, execution_time
        )
        
        return ProfileResult(
            kernel_name=kernel_func.__name__,
            execution_time=execution_time,
            memory_used=memory_used,
            peak_memory=peak_memory,
            gpu_utilization=gpu_util["gpu"],
            memory_bandwidth=memory_bandwidth,
            compute_utilization=gpu_util["memory"],
        )
    
    @contextmanager
    def _gpu_utilization_monitor(self):
        """Context manager for monitoring GPU utilization"""
        monitor = GPUUtilizationMonitor(self.device_id)
        monitor.start()
        try:
            yield monitor
        finally:
            monitor.stop()
    
    def _calculate_memory_bandwidth(self, memory_used: int, 
                                  execution_time: float) -> float:
        """Calculate effective memory bandwidth in GB/s"""
        if execution_time <= 0:
            return 0.0
        
        # Assume we read and write the data (factor of 2)
        total_bytes = memory_used * 2
        bandwidth_gbps = (total_bytes / (1024**3)) / execution_time
        
        return bandwidth_gbps
    
    def benchmark_attention(self, attention_impl, seq_lengths: List[int],
                          d_model: int = 512, batch_size: int = 8) -> Dict[str, List]:
        """Benchmark attention implementation across different sequence lengths"""
        results = {
            "seq_lengths": seq_lengths,
            "execution_times": [],
            "memory_usage": [],
            "throughput": [],
            "memory_bandwidth": []
        }
        
        for seq_len in seq_lengths:
            # Create test data
            q = torch.randn(batch_size, seq_len, d_model, device=self.device)
            k = torch.randn(batch_size, seq_len, d_model, device=self.device)
            v = torch.randn(batch_size, seq_len, d_model, device=self.device)
            
            # Profile the attention implementation
            profile_result = self.profile_kernel(
                attention_impl.forward, q, k, v
            )
            
            # Calculate throughput (tokens/second)
            total_tokens = batch_size * seq_len
            throughput = total_tokens / profile_result.execution_time
            
            results["execution_times"].append(profile_result.execution_time)
            results["memory_usage"].append(profile_result.memory_used)
            results["throughput"].append(throughput)
            results["memory_bandwidth"].append(profile_result.memory_bandwidth)
        
        return results
    
    def compare_implementations(self, implementations: Dict[str, Callable],
                              test_data: tuple) -> Dict[str, ProfileResult]:
        """Compare multiple kernel implementations"""
        results = {}
        
        for name, impl in implementations.items():
            results[name] = self.profile_kernel(impl, *test_data)
        
        return results
    
    def plot_performance_comparison(self, results: Dict[str, ProfileResult],
                                  save_path: Optional[str] = None):
        """Plot performance comparison between implementations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        names = list(results.keys())
        exec_times = [r.execution_time for r in results.values()]
        memory_usage = [r.memory_used / (1024**2) for r in results.values()]  # MB
        gpu_util = [r.gpu_utilization for r in results.values()]
        bandwidth = [r.memory_bandwidth for r in results.values()]
        
        # Execution time
        axes[0, 0].bar(names, exec_times)
        axes[0, 0].set_title("Execution Time (seconds)")
        axes[0, 0].set_ylabel("Time (s)")
        
        # Memory usage
        axes[0, 1].bar(names, memory_usage)
        axes[0, 1].set_title("Memory Usage (MB)")
        axes[0, 1].set_ylabel("Memory (MB)")
        
        # GPU utilization
        axes[1, 0].bar(names, gpu_util)
        axes[1, 0].set_title("GPU Utilization (%)")
        axes[1, 0].set_ylabel("Utilization (%)")
        
        # Memory bandwidth
        axes[1, 1].bar(names, bandwidth)
        axes[1, 1].set_title("Memory Bandwidth (GB/s)")
        axes[1, 1].set_ylabel("Bandwidth (GB/s)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class MemoryProfiler:
    """Detailed memory profiling and analysis"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.device_id = int(device.split(":")[-1]) if ":" in device else 0
    
    def get_memory_profile(self) -> MemoryProfile:
        """Get current memory profile"""
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        
        if NVML_AVAILABLE:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(self.device_id)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                total = mem_info.total
                free = mem_info.free
            except:
                # Fallback to torch info
                props = torch.cuda.get_device_properties(self.device_id)
                total = props.total_memory
                free = total - reserved
        else:
            props = torch.cuda.get_device_properties(self.device_id)
            total = props.total_memory
            free = total - reserved
        
        # Calculate fragmentation (simplified)
        fragmentation = (reserved - allocated) / total if total > 0 else 0.0
        
        return MemoryProfile(
            allocated=allocated,
            reserved=reserved,
            free=free,
            total=total,
            fragmentation=fragmentation
        )
    
    def monitor_memory_usage(self, kernel_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Monitor memory usage during kernel execution"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        initial_profile = self.get_memory_profile()
        
        # Execute kernel
        result = kernel_func(*args, **kwargs)
        torch.cuda.synchronize()
        
        final_profile = self.get_memory_profile()
        peak_memory = torch.cuda.max_memory_allocated(self.device)
        
        return {
            "initial_memory": asdict(initial_profile),
            "final_memory": asdict(final_profile),
            "peak_memory": peak_memory,
            "memory_delta": final_profile.allocated - initial_profile.allocated,
            "peak_utilization": peak_memory / final_profile.total,
            "result": result
        }
    
    def analyze_memory_fragmentation(self) -> Dict[str, float]:
        """Analyze memory fragmentation patterns"""
        profile = self.get_memory_profile()
        
        return {
            "fragmentation_ratio": profile.fragmentation,
            "utilization": profile.utilization,
            "waste_ratio": (profile.reserved - profile.allocated) / profile.total,
            "free_ratio": profile.free / profile.total
        }


class KernelProfiler:
    """Detailed kernel-level profiling with PyTorch profiler"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
    
    def profile_with_trace(self, kernel_func: Callable, *args, 
                          trace_path: str = "kernel_trace", **kwargs):
        """Profile kernel and generate trace file"""
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for _ in range(10):
                kernel_func(*args, **kwargs)
                prof.step()
        
        return prof
    
    def get_kernel_statistics(self, prof: torch.profiler.profile) -> Dict[str, Any]:
        """Extract detailed kernel statistics from profiler"""
        
        # Get CUDA kernel statistics
        cuda_events = prof.events().filter(lambda x: x.device_type == torch.profiler.DeviceType.CUDA)
        
        kernel_stats = {}
        for event in cuda_events:
            if event.name not in kernel_stats:
                kernel_stats[event.name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0
                }
            
            stats = kernel_stats[event.name]
            stats["count"] += 1
            stats["total_time"] += event.cuda_time_total
            stats["min_time"] = min(stats["min_time"], event.cuda_time)
            stats["max_time"] = max(stats["max_time"], event.cuda_time)
        
        # Calculate averages
        for stats in kernel_stats.values():
            if stats["count"] > 0:
                stats["avg_time"] = stats["total_time"] / stats["count"]
        
        return kernel_stats


class GPUUtilizationMonitor:
    """Monitor GPU utilization in real-time"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.utilization_history = []
        self.monitoring = False
    
    def start(self):
        """Start monitoring GPU utilization"""
        self.monitoring = True
        self.utilization_history = []
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
    
    def get_current_utilization(self) -> Dict[str, float]:
        """Get current GPU utilization"""
        if NVML_AVAILABLE:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(self.device_id)
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                return {"gpu": util.gpu, "memory": util.memory}
            except:
                pass
        
        # Fallback using GPUtil
        try:
            gpus = GPUtil.getGPUs()
            if self.device_id < len(gpus):
                gpu = gpus[self.device_id]
                return {"gpu": gpu.load * 100, "memory": gpu.memoryUtil * 100}
        except:
            pass
        
        return {"gpu": 0.0, "memory": 0.0}
    
    def get_average_utilization(self) -> Dict[str, float]:
        """Get average utilization during monitoring period"""
        if not self.utilization_history:
            # Get a single sample if no history
            return self.get_current_utilization()
        
        avg_gpu = sum(u["gpu"] for u in self.utilization_history) / len(self.utilization_history)
        avg_memory = sum(u["memory"] for u in self.utilization_history) / len(self.utilization_history)
        
        return {"gpu": avg_gpu, "memory": avg_memory}
    
    def sample_utilization(self):
        """Sample current utilization (call periodically)"""
        if self.monitoring:
            util = self.get_current_utilization()
            self.utilization_history.append(util) 