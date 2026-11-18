"""Unit tests for the Perception Runtime Readiness & Validation Tooling (Step 9)."""
from __future__ import annotations

import os
import pytest

from core.pipeline import AdaptivePipeline
from tools.check_vision_runtime import find_yolov8n_weights, get_vision_runtime_info
from vision.backend import MockBackend, YOLOBackend
from vision.vision_pipeline import VisionPipeline


# -----------------------------------------------------------------------------
# Test 1 — Runtime Readiness Info Structure
# -----------------------------------------------------------------------------
def test_get_vision_runtime_info_structure():
    """Verify runtime check collects and returns all expected readiness keys."""
    info = get_vision_runtime_info()

    assert isinstance(info, dict)
    required_keys = [
        "python_version",
        "python_executable",
        "ultralytics_installed",
        "ultralytics_version",
        "torch_installed",
        "torch_version",
        "cuda_available",
        "yolov8n_weights_available",
        "weights_path",
        "yolo_backend_importable",
        "yolo_backend_status",
        "yolo_backend_load_error",
        "effective_device",
        "perception_ready",
    ]
    for key in required_keys:
        assert key in info, f"Missing key: {key}"

    assert isinstance(info["python_version"], str)
    assert isinstance(info["ultralytics_installed"], bool)
    assert isinstance(info["torch_installed"], bool)
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["yolov8n_weights_available"], bool)
    assert info["yolo_backend_status"] in ("ACTIVE", "UNAVAILABLE", "ERROR")
    assert info["effective_device"] in ("cpu", "cuda")
    assert isinstance(info["perception_ready"], bool)


# -----------------------------------------------------------------------------
# Test 2 — YOLO Backend Status & Fallback Safety
# -----------------------------------------------------------------------------
def test_yolo_backend_status_and_fallback():
    """YOLOBackend never crashes when dependencies/weights are missing, and tracks status."""
    backend = YOLOBackend()

    assert backend.status in ("ACTIVE", "UNAVAILABLE")
    if backend.status == "UNAVAILABLE":
        assert backend.is_loaded is False
        assert backend.load_error is not None
        assert len(backend.load_error) > 0
        # Must return empty list cleanly without raising
        dets = backend.detect("dummy_frame")
        assert dets == []
    else:
        assert backend.is_loaded is True
        assert backend.load_error is None


# -----------------------------------------------------------------------------
# Test 3 — Mock Backend Status
# -----------------------------------------------------------------------------
def test_mock_backend_status():
    """MockBackend explicitly reports status as MOCK."""
    mock = MockBackend()
    assert mock.status == "MOCK"
    assert mock.is_loaded is True
    assert mock.load_error is None


# -----------------------------------------------------------------------------
# Test 4 — Vision Pipeline Status Delegation
# -----------------------------------------------------------------------------
def test_vision_pipeline_status_delegation():
    """VisionPipeline delegates status and load_error from its underlying backend."""
    mock = MockBackend()
    vp_mock = VisionPipeline(backend=mock)
    assert vp_mock.status == "MOCK"
    assert vp_mock.load_error is None

    vp_default = VisionPipeline()
    assert vp_default.status in ("ACTIVE", "UNAVAILABLE")
    assert vp_default.status == vp_default.backend.status
    assert vp_default.load_error == vp_default.backend.load_error


# -----------------------------------------------------------------------------
# Test 5 — Adaptive Pipeline Vision Status Delegation
# -----------------------------------------------------------------------------
def test_adaptive_pipeline_vision_status_delegation():
    """AdaptivePipeline correctly exposes vision_status and vision_load_error."""
    pipe = AdaptivePipeline()
    assert pipe.vision_status in ("ACTIVE", "UNAVAILABLE", "MOCK")
    assert pipe.vision_status == pipe.vision.status
    assert pipe.vision_load_error == pipe.vision.load_error


# -----------------------------------------------------------------------------
# Test 6 — Model Weights Environment Variable Handling
# -----------------------------------------------------------------------------
def test_weights_env_override(tmp_path, monkeypatch):
    """CROSSGUARD_YOLO_WEIGHTS env var is recognized when pointing to existing file."""
    fake_weights = tmp_path / "custom_yolov8n.pt"
    fake_weights.write_text("fake weight contents")

    monkeypatch.setenv("CROSSGUARD_YOLO_WEIGHTS", str(fake_weights))
    found, path = find_yolov8n_weights()

    assert found is True
    assert path == str(fake_weights)

    # Resolution inside YOLOBackend
    backend = YOLOBackend()
    assert backend._resolve_model_path("yolov8n.pt") == str(fake_weights)


# -----------------------------------------------------------------------------
# Test 7 — Real YOLOv8n Inference Execution
# -----------------------------------------------------------------------------
def test_yolo_real_inference_execution():
    """Verify real YOLOBackend performs inference on a frame without crashing."""
    import numpy as np

    backend = YOLOBackend()
    assert backend.status == "ACTIVE"
    assert backend.is_loaded is True
    assert backend.device == "cpu"

    # Synthetic black frame
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    dets = backend.detect(frame)

    assert isinstance(dets, list)
