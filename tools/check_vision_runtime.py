"""Perception runtime readiness check utility for CrossGuard (Step 9).

Inspects Python environment, dependency availability (Ultralytics, Torch),
hardware device support (CUDA / CPU), and YOLOv8n weights status.

Usage:
  python tools/check_vision_runtime.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def find_yolov8n_weights() -> Tuple[bool, Optional[str]]:
    """Check whether yolov8n.pt weights are present locally."""
    # 1. Environment variable override
    env_path = os.environ.get("CROSSGUARD_YOLO_WEIGHTS")
    if env_path and os.path.exists(env_path):
        return True, env_path

    # 2. Local candidate locations relative to project root or current working dir
    candidates = [
        "yolov8n.pt",
        os.path.join(PROJECT_ROOT, "yolov8n.pt"),
        os.path.join(PROJECT_ROOT, "models", "yolov8n.pt"),
        os.path.join(PROJECT_ROOT, "vision", "yolov8n.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return True, os.path.abspath(c)

    return False, None


def get_vision_runtime_info() -> Dict[str, Any]:
    """Inspect and collect structured runtime readiness information."""
    python_ver = sys.version.split()[0]
    python_exe = sys.executable

    # Ultralytics check
    ultralytics_spec = importlib.util.find_spec("ultralytics")
    ultralytics_installed = ultralytics_spec is not None
    ultralytics_version = None
    if ultralytics_installed:
        try:
            import ultralytics  # type: ignore
            ultralytics_version = getattr(ultralytics, "__version__", "unknown")
        except Exception:
            ultralytics_version = "error loading"

    # PyTorch check
    torch_spec = importlib.util.find_spec("torch")
    torch_installed = torch_spec is not None
    torch_version = None
    cuda_available = False
    if torch_installed:
        try:
            import torch  # type: ignore
            torch_version = getattr(torch, "__version__", "unknown")
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            torch_version = "error loading"

    # Weights check
    weights_available, weights_path = find_yolov8n_weights()

    # YOLOBackend import & status check
    yolo_importable = False
    yolo_status = "UNAVAILABLE"
    yolo_error: Optional[str] = None
    effective_device = "cpu"

    try:
        from vision.backend import YOLOBackend
        from vision.models import resolve_device
        yolo_importable = True
        backend = YOLOBackend()
        yolo_status = backend.status
        yolo_error = backend.load_error
        effective_device = backend.device
    except Exception as e:
        yolo_importable = False
        yolo_status = "ERROR"
        yolo_error = str(e)

    perception_ready = (yolo_status == "ACTIVE")

    return {
        "python_version": python_ver,
        "python_executable": python_exe,
        "ultralytics_installed": ultralytics_installed,
        "ultralytics_version": ultralytics_version,
        "torch_installed": torch_installed,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "yolov8n_weights_available": weights_available,
        "weights_path": weights_path,
        "yolo_backend_importable": yolo_importable,
        "yolo_backend_status": yolo_status,
        "yolo_backend_load_error": yolo_error,
        "effective_device": effective_device,
        "perception_ready": perception_ready,
    }


def print_report(info: Optional[Dict[str, Any]] = None) -> None:
    """Print human-readable runtime readiness report."""
    if info is None:
        info = get_vision_runtime_info()

    print("=" * 60)
    print("CROSSGUARD PERCEPTION RUNTIME READINESS CHECK")
    print("=" * 60)
    print(f"Python version:          {info['python_version']}")
    print(f"Python interpreter:      {info['python_executable']}")
    print(f"Ultralytics installed:   {'yes (' + str(info['ultralytics_version']) + ')' if info['ultralytics_installed'] else 'no'}")
    print(f"PyTorch installed:       {'yes (' + str(info['torch_version']) + ')' if info['torch_installed'] else 'no'}")
    print(f"CUDA available:          {'yes' if info['cuda_available'] else 'no (CPU only)'}")
    print(f"YOLOv8n weights:         {'yes (' + str(info['weights_path']) + ')' if info['yolov8n_weights_available'] else 'no (not found locally)'}")
    print(f"YOLO backend importable: {'yes' if info['yolo_backend_importable'] else 'no'}")
    print(f"YOLO backend status:     {info['yolo_backend_status']}")
    if info.get("yolo_backend_load_error"):
        print(f"Backend reason/error:    {info['yolo_backend_load_error']}")
    print(f"Effective device:        {info['effective_device']}")
    print("-" * 60)
    print(f"PERCEPTION STATUS:       {'READY' if info['perception_ready'] else 'BLOCKED'}")
    if not info["perception_ready"]:
        print("Reason: Ultralytics/PyTorch or model weights not ready in current environment.")
        print("Note:   Pipeline operates safely with MockBackend or zero-detection fallback.")
    print("=" * 60)


if __name__ == "__main__":
    print_report()
