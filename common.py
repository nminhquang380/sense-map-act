"""Shared timing / reporting utilities for the GTC benchmark kit.

Timing rules (state these on the poster under "Benchmarking methodology"):
  * warm-up iterations are discarded (CUDA context, cuDNN autotune, allocator);
  * torch.cuda.synchronize() is called before reading the clock, because CUDA
    kernels are launched asynchronously;
  * we report the MEDIAN over repeated runs (robust to OS jitter) plus p10/p90;
  * the CPU baseline uses all physical cores (torch.set_num_threads).
"""
import json
import os
import platform
import statistics
import subprocess
import time
from contextlib import contextmanager

import torch


def sync(device):
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_fn(fn, device, warmup=5, repeat=20):
    """Run fn() `warmup` times (discarded) then `repeat` times; return stats in ms."""
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(repeat):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    n = len(samples)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": samples[max(0, int(0.1 * n) - 1)],
        "p90_ms": samples[min(n - 1, int(0.9 * n))],
        "n": n,
    }


def hardware_info():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads_used": torch.get_num_threads(),
        "gpus": [],
    }
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "name": p.name,
                "mem_GB": round(p.total_memory / 1e9, 1),
                "sm_count": p.multi_processor_count,
                "capability": f"{p.major}.{p.minor}",
            })
        try:
            info["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        except Exception:
            pass
    return info


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {path}")


def set_cpu_threads(n=None):
    """Use all physical cores for the CPU baseline (fair, best-effort baseline)."""
    n = n or os.cpu_count()
    torch.set_num_threads(n)
    return n


@contextmanager
def tf32(enabled):
    old_mm = torch.backends.cuda.matmul.allow_tf32
    old_cd = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32 = old_cd


def fmt_duration(seconds):
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"
