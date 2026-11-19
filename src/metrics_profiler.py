"""
src/metrics_profiler.py
=======================
Unified Metrics Profiler for 8GB VRAM Constraint & RAG Benchmarking.
Monitors:
- Peak CUDA Allocated Memory (torch.cuda.max_memory_allocated)
- Peak CUDA Reserved Memory (torch.cuda.max_memory_reserved)
- System/NVML Process VRAM Peak (via NVML or nvidia-smi fallback)
- Wall-clock latencies (End-to-End, Retrieval, Generation)
- LLM Forward Pass / Generation Call Counter per sample
"""

import time
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import torch

try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


def get_nvidia_smi_process_memory_mb() -> Optional[float]:
    """Query nvidia-smi for current process memory usage if NVML is not installed."""
    try:
        pid = os.getpid()
        cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0] == str(pid):
                    return float(parts[1])
    except Exception as e:
        # Fallback failure handled cleanly
        return None
    return None


@dataclass
class SampleProfileRecord:
    sample_id: str
    wall_latency_sec: float
    retrieval_latency_sec: float = 0.0
    generation_latency_sec: float = 0.0
    llm_forward_calls: int = 0
    cuda_allocated_peak_mb: float = 0.0
    cuda_reserved_peak_mb: float = 0.0
    nvml_process_peak_mb: Optional[float] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


class MetricsProfiler:
    """
    Context manager & tracker for benchmarking RAG pipelines on an 8GB VRAM budget.
    """
    def __init__(self, track_cuda: bool = True, track_nvml: bool = True):
        self.track_cuda = track_cuda and torch.cuda.is_available()
        self.track_nvml = track_nvml
        self._nvml_initialized = False
        self.diagnostics: List[str] = []

        if self.track_nvml and _HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                self._nvml_initialized = False
                self.diagnostics.append(f"NVML init failed: {e}")

        self.records: List[SampleProfileRecord] = []
        self._total_forward_calls: int = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def reset_cuda_stats(self) -> None:
        if self.track_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def get_current_cuda_mem_mb(self) -> Dict[str, float]:
        if not self.track_cuda:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0, "max_reserved_mb": 0.0}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
        }

    def get_process_vram_mb(self) -> Optional[float]:
        """Fetch current process GPU memory via NVML or nvidia-smi."""
        if self._nvml_initialized:
            try:
                pid = os.getpid()
                device_count = pynvml.nvmlDeviceGetCount()
                total_used = 0.0
                found = False
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for p in procs:
                        if p.pid == pid:
                            total_used += (p.usedGpuMemory / (1024 ** 2))
                            found = True
                if found:
                    return round(total_used, 2)
            except Exception as nvml_err:
                self.diagnostics.append(f"NVML memory query failed: {nvml_err}")
        
        # Fallback to nvidia-smi
        smi_mem = get_nvidia_smi_process_memory_mb()
        if smi_mem is not None:
            return round(smi_mem, 2)
        return None

    def start_pipeline(self) -> None:
        self.reset_cuda_stats()
        self._start_time = time.perf_counter()
        self._total_forward_calls = 0
        self.records.clear()

    def stop_pipeline(self) -> None:
        self._end_time = time.perf_counter()

    def record_sample(
        self,
        sample_id: str,
        wall_latency_sec: float,
        retrieval_latency_sec: float = 0.0,
        generation_latency_sec: float = 0.0,
        llm_forward_calls: int = 1,
        extra_metrics: Optional[Dict[str, Any]] = None
    ) -> SampleProfileRecord:
        cuda_stats = self.get_current_cuda_mem_mb()
        nvml_vram = self.get_process_vram_mb()

        rec = SampleProfileRecord(
            sample_id=sample_id,
            wall_latency_sec=round(wall_latency_sec, 4),
            retrieval_latency_sec=round(retrieval_latency_sec, 4),
            generation_latency_sec=round(generation_latency_sec, 4),
            llm_forward_calls=llm_forward_calls,
            cuda_allocated_peak_mb=cuda_stats["max_allocated_mb"],
            cuda_reserved_peak_mb=cuda_stats["max_reserved_mb"],
            nvml_process_peak_mb=nvml_vram,
            extra_metrics=extra_metrics or {}
        )
        self.records.append(rec)
        self._total_forward_calls += llm_forward_calls
        return rec

    def summary(self) -> Dict[str, Any]:
        n = len(self.records)
        total_wall = (self._end_time - self._start_time) if (self._start_time and self._end_time) else 0.0
        
        if n == 0:
            return {
                "sample_count": 0,
                "total_wall_sec": round(total_wall, 4),
                "total_forward_calls": self._total_forward_calls,
                "avg_forward_calls_per_sample": 0.0,
                "peak_cuda_allocated_mb": 0.0,
                "peak_cuda_reserved_mb": 0.0,
                "diagnostics": self.diagnostics
            }

        avg_lat = sum(r.wall_latency_sec for r in self.records) / n
        avg_ret_lat = sum(r.retrieval_latency_sec for r in self.records) / n
        avg_gen_lat = sum(r.generation_latency_sec for r in self.records) / n
        avg_forwards = self._total_forward_calls / n
        
        max_alloc = max((r.cuda_allocated_peak_mb for r in self.records), default=0.0)
        max_res = max((r.cuda_reserved_peak_mb for r in self.records), default=0.0)
        
        valid_nvml = [r.nvml_process_peak_mb for r in self.records if r.nvml_process_peak_mb is not None]
        max_nvml = max(valid_nvml) if valid_nvml else None

        return {
            "sample_count": n,
            "total_wall_sec": round(total_wall, 4),
            "total_forward_calls": self._total_forward_calls,
            "avg_forward_calls_per_sample": round(avg_forwards, 2),
            "avg_latency_sec": round(avg_lat, 4),
            "avg_retrieval_latency_sec": round(avg_ret_lat, 4),
            "avg_generation_latency_sec": round(avg_gen_lat, 4),
            "peak_cuda_allocated_mb": round(max_alloc, 2),
            "peak_cuda_reserved_mb": round(max_res, 2),
            "peak_nvml_process_mb": round(max_nvml, 2) if max_nvml is not None else "N/A",
            "vram_headroom_8gb_mb": round(8192.0 - max_res, 2) if max_res > 0 else 8192.0,
            "diagnostics": self.diagnostics
        }

    def __del__(self):
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                self.diagnostics.append(f"NVML shutdown failed: {e}")


def self_test():
    print("Testing MetricsProfiler on current machine...")
    profiler = MetricsProfiler()
    profiler.start_pipeline()
    
    # Simulate sample 1
    t0 = time.perf_counter()
    time.sleep(0.05)
    t_ret = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    if torch.cuda.is_available():
        dummy_tensor = torch.randn((1024, 1024, 8), device="cuda")
        torch.cuda.synchronize()
        _ = dummy_tensor * 2.0
        del dummy_tensor
        torch.cuda.synchronize()
    time.sleep(0.05)
    t_gen = time.perf_counter() - t1
    
    total_t = t_ret + t_gen
    rec = profiler.record_sample("test_sample_0", total_t, t_ret, t_gen, llm_forward_calls=1)
    
    profiler.stop_pipeline()
    res = profiler.summary()
    print("Self-test Summary:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    assert res["sample_count"] == 1
    assert res["total_forward_calls"] == 1
    print("Self-test PASS!")


if __name__ == "__main__":
    self_test()
