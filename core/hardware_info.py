"""
CrossGuard Hardware Telemetry & Probe
=====================================
Performs genuine system hardware and acceleration framework diagnostics.
Does not fabricate mock GPU metrics or benchmark numbers.
"""

import platform
import sys
from typing import Any, Dict

import cv2
import psutil
import torch


def get_hardware_telemetry() -> Dict[str, Any]:
    """
    Probes host system hardware, CPU topology, system RAM, PyTorch build,
    and CUDA runtime acceleration capabilities.

    Returns:
        Structured dictionary of real system telemetry.
    """
    # System & CPU
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu_processor = platform.processor() or "x86_64 Compatible"
    cpu_cores_physical = psutil.cpu_count(logical=False) or 1
    cpu_cores_logical = psutil.cpu_count(logical=True) or 1

    # RAM
    vmem = psutil.virtual_memory()
    ram_total_gb = vmem.total / (1024**3)
    ram_available_gb = vmem.available / (1024**3)
    ram_used_percent = vmem.percent

    # PyTorch & Acceleration Check
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0

    if cuda_available and cuda_device_count > 0:
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_vram_str = f"{cuda_vram_total:.1f} GB"
        acceleration_mode = f"CUDA Active ({cuda_device_name})"
    else:
        cuda_device_name = "None Detected (CPU Host Mode)"
        cuda_vram_str = "N/A"
        acceleration_mode = "CPU Only (PyTorch CPU Build)"

    return {
        "os_name": os_name,
        "cpu_processor": cpu_processor,
        "cpu_cores_physical": cpu_cores_physical,
        "cpu_cores_logical": cpu_cores_logical,
        "ram_total_gb": round(ram_total_gb, 1),
        "ram_available_gb": round(ram_available_gb, 1),
        "ram_used_percent": round(ram_used_percent, 1),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "opencv_version": cv2.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "cuda_vram_str": cuda_vram_str,
        "acceleration_mode": acceleration_mode,
        "benchmark_status": "Stage 4 Benchmark Harness Pending",
    }
