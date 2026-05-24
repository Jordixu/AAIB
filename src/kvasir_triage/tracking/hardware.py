"""Snapshot hardware info at run start and time training phases."""
from __future__ import annotations

import platform
import socket
import time
from contextlib import contextmanager
from typing import Any

import torch


def snapshot_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / 2**30, 2)
        info["cuda_version"] = torch.version.cuda

    try:
        import psutil
        mem = psutil.virtual_memory()
        info["system_ram_gb"] = round(mem.total / 2**30, 2)
        info["cpu_cores"] = psutil.cpu_count(logical=False)
        info["cpu_model"] = platform.processor()
    except ImportError:
        pass

    return info


def gpu_memory_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
        "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
    }


@contextmanager
def Stopwatch(label: str, log_fn=None):
    """Context manager that times a block and optionally logs it."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if log_fn:
        log_fn(f"{label}: {elapsed:.2f}s")
    return elapsed
