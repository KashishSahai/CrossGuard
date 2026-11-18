"""Unit and integration tests for explicit CPU/GPU processing selection.

Verifies Auto, CPU, and GPU (CUDA) device resolution, strict error raising on unavailable
CUDA (no silent fallback), and runtime device propagation through AdaptivePipeline.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from core.analytics import AnalyticsSession
from core.pipeline import AdaptivePipeline
from vision.backend import MockBackend, YOLOBackend
from vision.models import DeviceType, is_cuda_available, resolve_device
from vision.vision_pipeline import VisionPipeline, VisionPipelineConfig


# -----------------------------------------------------------------------------
# Test 1: Auto with CUDA available -> CUDA
# -----------------------------------------------------------------------------
def test_auto_with_cuda_available():
    """Verify Auto mode selects CUDA when CUDA is detected."""
    with patch("vision.models.is_cuda_available", return_value=True):
        assert resolve_device("auto") == "cuda"
        assert resolve_device(DeviceType.AUTO) == "cuda"

        cfg = VisionPipelineConfig(device=DeviceType.AUTO)
        assert cfg.get_effective_device() == "cuda"

        backend = MockBackend(device=DeviceType.AUTO)
        assert backend.device == "cuda"
        assert backend.requested_device == "auto"


# -----------------------------------------------------------------------------
# Test 2: Auto without CUDA -> CPU
# -----------------------------------------------------------------------------
def test_auto_without_cuda():
    """Verify Auto mode safely falls back to CPU when CUDA is unavailable."""
    with patch("vision.models.is_cuda_available", return_value=False):
        assert resolve_device("auto") == "cpu"
        assert resolve_device(DeviceType.AUTO) == "cpu"

        cfg = VisionPipelineConfig(device=DeviceType.AUTO)
        assert cfg.get_effective_device() == "cpu"

        backend = MockBackend(device=DeviceType.AUTO)
        assert backend.device == "cpu"
        assert backend.requested_device == "auto"


# -----------------------------------------------------------------------------
# Test 3: Explicit CPU -> CPU
# -----------------------------------------------------------------------------
def test_explicit_cpu():
    """Verify explicit CPU selection forces CPU even if CUDA is available."""
    # When CUDA is available
    with patch("vision.models.is_cuda_available", return_value=True):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device(DeviceType.CPU) == "cpu"
        backend = MockBackend(device="cpu")
        assert backend.device == "cpu"
        assert backend.requested_device == "cpu"

    # When CUDA is unavailable
    with patch("vision.models.is_cuda_available", return_value=False):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device(DeviceType.CPU) == "cpu"
        backend = MockBackend(device=DeviceType.CPU)
        assert backend.device == "cpu"


# -----------------------------------------------------------------------------
# Test 4: Explicit GPU with CUDA available -> CUDA
# -----------------------------------------------------------------------------
def test_explicit_gpu_with_cuda_available():
    """Verify explicit GPU selection resolves to CUDA when CUDA is available."""
    with patch("vision.models.is_cuda_available", return_value=True):
        assert resolve_device("cuda") == "cuda"
        assert resolve_device(DeviceType.CUDA) == "cuda"
        assert resolve_device("gpu") == "cuda"

        backend = MockBackend(device="cuda")
        assert backend.device == "cuda"
        assert backend.requested_device == "cuda"


# -----------------------------------------------------------------------------
# Test 5: Explicit GPU without CUDA -> clear failure (NO silent fallback)
# -----------------------------------------------------------------------------
def test_explicit_gpu_without_cuda_raises_error():
    """Verify explicit GPU selection raises RuntimeError rather than silently falling back to CPU."""
    with patch("vision.models.is_cuda_available", return_value=False), \
         patch("core.gpu_config.find_env_file", return_value=None), \
         patch.dict(os.environ, {}, clear=True):
        # 1. resolve_device must fail
        with pytest.raises(RuntimeError) as exc_info:
            resolve_device("cuda")
        assert "CUDA device requested but CUDA is not available" in str(exc_info.value)

        with pytest.raises(RuntimeError):
            resolve_device(DeviceType.CUDA)

        with pytest.raises(RuntimeError):
            resolve_device("gpu")

        # 2. Backend instantiation with CUDA must fail
        with pytest.raises(RuntimeError):
            MockBackend(device="cuda")

        with pytest.raises(RuntimeError):
            YOLOBackend(device="cuda")

        # 3. Pipeline instantiation with CUDA must fail when no worker or local CUDA is available
        with pytest.raises(RuntimeError):
            AdaptivePipeline(device=DeviceType.CUDA)


# -----------------------------------------------------------------------------
# Test 6: Pipeline device selection reaches the existing pipeline
# -----------------------------------------------------------------------------
def test_pipeline_device_selection_propagation():
    """Verify device selection passed to AdaptivePipeline properly reaches VisionPipeline."""
    # CPU
    pipe_cpu = AdaptivePipeline(
        device="cpu",
        vision_pipeline=VisionPipeline(
            config=VisionPipelineConfig(device="cpu"),
            backend=MockBackend(device="cpu"),
        ),
    )
    assert pipe_cpu.device == "cpu"
    assert pipe_cpu.requested_device == "cpu"

    # Auto when CUDA available
    with patch("vision.models.is_cuda_available", return_value=True):
        pipe_auto = AdaptivePipeline(
            device="auto",
            vision_pipeline=VisionPipeline(
                config=VisionPipelineConfig(device="auto"),
                backend=MockBackend(device="auto"),
            ),
        )
        assert pipe_auto.device == "cuda"
        assert pipe_auto.requested_device == "auto"


# -----------------------------------------------------------------------------
# Test 7: Actual runtime device is exposed correctly
# -----------------------------------------------------------------------------
def test_actual_runtime_device_exposed():
    """Verify runtime device is accurately reported and reflected in AnalyticsSession."""
    backend_cpu = MockBackend(device="cpu")
    pipeline = AdaptivePipeline(
        vision_pipeline=VisionPipeline(backend=backend_cpu),
    )
    assert pipeline.device == "cpu"

    # AnalyticsSession receives the uppercase actual device string
    session = AnalyticsSession(device=pipeline.device.upper())
    assert session.device == "CPU"
    summary = session.to_executive_summary()
    assert summary["device"] == "CPU"


# -----------------------------------------------------------------------------
# Test 8: Existing reset behavior still works
# -----------------------------------------------------------------------------
def test_reset_preserves_device_configuration():
    """Verify pipeline reset preserves configured compute device and tracker state."""
    pipeline = AdaptivePipeline(
        device="cpu",
        vision_pipeline=VisionPipeline(backend=MockBackend(device="cpu")),
    )
    assert pipeline.device == "cpu"

    # Step clock and advance time
    pipeline.current_time = 15.0
    pipeline.reset()

    # Time and controller reset, but device configuration remains
    assert pipeline.current_time == 0.0
    assert pipeline.device == "cpu"
    assert pipeline.requested_device == "cpu"


# -----------------------------------------------------------------------------
# Test 9: Existing video switching still works
# -----------------------------------------------------------------------------
def test_video_switching_device_preservation():
    """Verify that switching video sources allows clean re-initialization with device settings."""
    # Video A on CPU
    pipe_a = AdaptivePipeline(device=DeviceType.CPU, vision_pipeline=MockBackend(device="cpu"))
    session_a = AnalyticsSession(video_name="video_a.mp4", device=pipe_a.device.upper())
    assert session_a.video_name == "video_a.mp4"
    assert session_a.device == "CPU"

    # Video B on Auto
    with patch("vision.models.is_cuda_available", return_value=False):
        pipe_b = AdaptivePipeline(device=DeviceType.AUTO, vision_pipeline=MockBackend(device="auto"))
        session_b = AnalyticsSession(video_name="video_b.mp4", device=pipe_b.device.upper())
        assert session_b.video_name == "video_b.mp4"
        assert session_b.device == "CPU"
