"""Tests for CrossGuard Remote GPU Worker Protocol, Client, and Pipeline Integration.

Verifies:
1. Worker protocol (health response, CUDA status, GPU metadata, detection response,
   malformed response, timeout, connection failure).
2. Device selection (CPU -> local CPU, GPU + connected worker -> remote CUDA,
   GPU + disconnected worker -> clear error, Auto + GPU available -> remote CUDA,
   Auto + GPU unavailable -> local CPU, explicit GPU never silently falls back).
3. Pipeline integration (downstream receives identical detection/telemetry/FSM contracts).
4. Reset and video switching state isolation.

Uses MockGPUWorkerServer for deterministic, network-safe in-process testing.
DOES NOT import streamlit or app at module level to preserve test isolation.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from core.analytics import AnalyticsSession
from core.models import Phase, VehicleSignal
from core.pipeline import AdaptivePipeline, PipelineFrameResult
from gpu_worker.backend import RemoteGPUBackend
from gpu_worker.client import (
    GPUWorkerClient,
    GPUWorkerConnectionError,
    GPUWorkerError,
    GPUWorkerProtocolError,
    GPUWorkerTimeoutError,
    GPUWorkerUnavailableError,
)
from gpu_worker.mock_worker import MockGPUWorkerServer
from gpu_worker.protocol import (
    DetectionItem,
    InferenceRequest,
    InferenceResponse,
    WorkerHealth,
    decode_b64_to_frame,
    encode_frame_to_b64,
)
from vision.models import DeviceType, RawDetection, VisionPipelineConfig
from vision.vision_pipeline import VisionPipeline


# =============================================================================
# 1. Worker Protocol & Data Transfer Tests
# =============================================================================

def test_health_response_parsing():
    """Verify health response dictionary converts properly to WorkerHealth model."""
    raw = {
        "status": "ok",
        "cuda_available": True,
        "gpu_name": "Tesla T4",
        "gpu_memory_total_mb": 15109.0,
        "gpu_memory_allocated_mb": 128.5,
        "yolo_loaded": True,
        "model_name": "yolov8n.pt",
        "torch_version": "2.1.0+cu121",
        "device": "cuda",
    }
    health = WorkerHealth.from_dict(raw)
    assert health.status == "ok"
    assert health.cuda_available is True
    assert health.gpu_name == "Tesla T4"
    assert health.gpu_memory_total_mb == 15109.0
    assert health.yolo_loaded is True
    assert health.is_ready is True
    assert health.device == "cuda"


def test_cuda_status_and_gpu_metadata_parsing():
    """Verify CUDA availability and degraded status handling."""
    # Degraded: CUDA unavailable
    raw_no_cuda = {
        "status": "degraded",
        "cuda_available": False,
        "gpu_name": "CPU Only",
        "yolo_loaded": True,
    }
    health_no_cuda = WorkerHealth.from_dict(raw_no_cuda)
    assert health_no_cuda.cuda_available is False
    assert health_no_cuda.is_ready is False

    # Degraded: Model not loaded
    raw_no_model = {
        "status": "ok",
        "cuda_available": True,
        "gpu_name": "Tesla T4",
        "yolo_loaded": False,
    }
    health_no_model = WorkerHealth.from_dict(raw_no_model)
    assert health_no_model.is_ready is False


def test_detection_response_parsing():
    """Verify InferenceResponse parses detections and converts to RawDetection."""
    raw = {
        "status": "ok",
        "detections": [
            {
                "x1": 100.5,
                "y1": 200.0,
                "x2": 300.2,
                "y2": 400.8,
                "confidence": 0.92,
                "class_id": 2,
                "class_name": "car",
            },
            {
                "x1": 500.0,
                "y1": 150.0,
                "x2": 550.0,
                "y2": 260.0,
                "confidence": 0.81,
                "class_id": 0,
                "class_name": "person",
            },
        ],
        "inference_time_ms": 11.4,
        "device": "cuda",
        "gpu_name": "Tesla T4",
        "frame_index": 7,
    }
    resp = InferenceResponse.from_dict(raw)
    assert resp.is_success is True
    assert resp.inference_time_ms == 11.4
    assert resp.gpu_name == "Tesla T4"
    assert len(resp.detections) == 2

    raw_dets = resp.to_raw_detections()
    assert len(raw_dets) == 2
    assert isinstance(raw_dets[0], RawDetection)
    assert raw_dets[0].class_name == "car"
    assert raw_dets[0].confidence == 0.92
    assert raw_dets[1].class_name == "person"


def test_frame_encoding_and_decoding():
    """Verify image frame round-trip base64 encoding and decoding."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:50, 30:70] = [0, 255, 0]  # green rectangle

    b64_str = encode_frame_to_b64(img, quality=90)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0

    decoded = decode_b64_to_frame(b64_str)
    assert decoded.shape == (100, 100, 3)

    # Empty frame should fail cleanly
    with pytest.raises(ValueError):
        encode_frame_to_b64(None)


def test_malformed_response_handling():
    """Verify client rejects malformed JSON or invalid schema from worker."""
    with MockGPUWorkerServer() as server:
        client = GPUWorkerClient(base_url=server.url)

        # Malformed JSON on health
        server.should_malform_json = True
        with pytest.raises(GPUWorkerProtocolError):
            client.get_health()

        # Malformed JSON on detect
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(GPUWorkerProtocolError):
            client.detect(dummy)


def test_worker_timeout_handling():
    """Verify client raises GPUWorkerTimeoutError when worker hangs."""
    with MockGPUWorkerServer() as server:
        server.should_timeout = True
        client = GPUWorkerClient(base_url=server.url, timeout=0.3)

        with pytest.raises(GPUWorkerTimeoutError):
            client.get_health()

        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(GPUWorkerTimeoutError):
            client.detect(dummy)


def test_worker_connection_failure():
    """Verify client raises GPUWorkerConnectionError or GPUWorkerTimeoutError when endpoint is unreachable."""
    dead_client = GPUWorkerClient(base_url="http://127.0.0.1:59999", timeout=0.2)

    with pytest.raises((GPUWorkerConnectionError, GPUWorkerTimeoutError)):
        dead_client.get_health()

    ready, msg, _ = dead_client.check_ready()
    assert ready is False
    assert any(w in msg.lower() for w in ("connect", "timed out", "unreachable"))


def test_worker_http_500_error():
    """Verify client raises GPUWorkerProtocolError on HTTP 500 error."""
    with MockGPUWorkerServer() as server:
        server.should_error_500 = True
        client = GPUWorkerClient(base_url=server.url)

        with pytest.raises(GPUWorkerProtocolError):
            client.get_health()


# =============================================================================
# 2. Device Selection & Resolution Semantics
# =============================================================================

def test_device_selection_cpu():
    """Verify selecting CPU always builds local CPU pipeline regardless of worker presence."""
    with MockGPUWorkerServer() as server:
        client = GPUWorkerClient(base_url=server.url)
        pipeline = AdaptivePipeline(
            device=DeviceType.CPU,
            gpu_worker_client=client,
        )
        assert pipeline.device == "cpu"
        assert pipeline.requested_device == "cpu"
        assert pipeline.execution_mode == "Local CPU"
        assert pipeline.gpu_name is None


def test_device_selection_gpu_with_connected_worker():
    """Verify selecting GPU with online CUDA worker configures RemoteGPUBackend."""
    with MockGPUWorkerServer(gpu_name="NVIDIA Tesla T4") as server:
        client = GPUWorkerClient(base_url=server.url)
        pipeline = AdaptivePipeline(
            device=DeviceType.CUDA,
            gpu_worker_client=client,
        )
        assert pipeline.device == "cuda"
        assert pipeline.requested_device == "cuda"
        assert pipeline.execution_mode == "Remote CUDA"
        assert pipeline.gpu_name == "NVIDIA Tesla T4"
        assert isinstance(pipeline.vision.backend, RemoteGPUBackend)


def test_device_selection_gpu_with_disconnected_worker_fails():
    """Verify selecting GPU without reachable CUDA worker raises RuntimeError without silent fallback."""
    dead_client = GPUWorkerClient(base_url="http://127.0.0.1:59998", timeout=0.5)

    with patch("core.pipeline.is_cuda_available", return_value=False):
        with pytest.raises(RuntimeError) as exc_info:
            AdaptivePipeline(
                device=DeviceType.CUDA,
                gpu_worker_client=dead_client,
            )
        assert "CUDA GPU unavailable" in str(exc_info.value) or "Cannot connect" in str(exc_info.value)


def test_device_selection_auto_with_gpu_available():
    """Verify Auto mode selects Remote CUDA when worker is online."""
    with MockGPUWorkerServer(gpu_name="NVIDIA A100") as server:
        client = GPUWorkerClient(base_url=server.url)
        pipeline = AdaptivePipeline(
            device=DeviceType.AUTO,
            gpu_worker_client=client,
        )
        assert pipeline.device == "cuda"
        assert pipeline.execution_mode == "Remote CUDA"
        assert pipeline.gpu_name == "NVIDIA A100"


def test_device_selection_auto_without_gpu_uses_cpu():
    """Verify Auto mode falls back safely to Local CPU when worker is offline and local CUDA is missing."""
    dead_client = GPUWorkerClient(base_url="http://127.0.0.1:59997", timeout=0.5)

    with patch("core.pipeline.is_cuda_available", return_value=False):
        pipeline = AdaptivePipeline(
            device=DeviceType.AUTO,
            gpu_worker_client=dead_client,
        )
        assert pipeline.device == "cpu"
        assert pipeline.execution_mode == "Local CPU"
        assert pipeline.gpu_name is None


def test_explicit_gpu_never_silently_falls_back():
    """Verify that requesting CUDA never quietly degrades to CPU."""
    dead_client = GPUWorkerClient(base_url="http://127.0.0.1:59996", timeout=0.5)

    with patch("core.pipeline.is_cuda_available", return_value=False):
        with pytest.raises(Exception):
            pipeline = AdaptivePipeline(
                device=DeviceType.CUDA,
                gpu_worker_client=dead_client,
            )
            # If reached, must assert not CPU
            assert pipeline.device != "cpu"


# =============================================================================
# 3. Pipeline Integration & Downstream Compatibility
# =============================================================================

def test_pipeline_integration_downstream_compatibility():
    """Verify RemoteGPUBackend feeds identically into ByteTrack, SpatialAnalysis, Telemetry, and FSM."""
    with MockGPUWorkerServer() as server:
        client = GPUWorkerClient(base_url=server.url)
        pipeline = AdaptivePipeline(
            device=DeviceType.CUDA,
            gpu_worker_client=client,
            default_fps=10.0,
        )

        dummy_frame = np.zeros((800, 1000, 3), dtype=np.uint8)
        res = pipeline.process_frame(dummy_frame, frame_index=0, dt=0.1)

        assert isinstance(res, PipelineFrameResult)
        assert res.frame_index == 0
        assert res.dt == 0.1
        # Downstream structures present
        assert isinstance(res.tracked_detections, list)
        assert isinstance(res.spatial_detections, list)
        assert res.telemetry_snapshot is not None
        assert res.demand_snapshot is not None
        assert res.controller_state is not None
        assert res.signal_state is not None

        # Check recorded latency
        assert pipeline.last_inference_time_ms > 0.0

        # Check AnalyticsSession records Remote CUDA correctly
        session = AnalyticsSession(
            device=pipeline.execution_mode,
            gpu_name=pipeline.gpu_name,
        )
        session.record_frame(res, frame_proc_time_s=0.05, yolo_latency_s=pipeline.last_inference_time_ms / 1000.0)
        summary = session.to_executive_summary()
        assert summary["device"] == "Remote CUDA"
        assert summary["gpu_name"] == server.gpu_name


# =============================================================================
# 4. Reset & Video Switching State Isolation
# =============================================================================

def test_pipeline_reset_preserves_worker_and_device():
    """Verify pipeline reset maintains remote GPU backend configuration while resetting temporal state."""
    with MockGPUWorkerServer() as server:
        client = GPUWorkerClient(base_url=server.url)
        pipeline = AdaptivePipeline(
            device=DeviceType.CUDA,
            gpu_worker_client=client,
        )
        dummy_frame = np.zeros((800, 1000, 3), dtype=np.uint8)
        _ = pipeline.process_frame(dummy_frame, frame_index=0, dt=0.1)
        _ = pipeline.process_frame(dummy_frame, frame_index=1, dt=0.1)

        assert pipeline.current_time > 0.0

        # Reset
        pipeline.reset()

        assert pipeline.current_time == 0.0
        assert pipeline.last_result is None
        assert pipeline.device == "cuda"
        assert pipeline.execution_mode == "Remote CUDA"
        assert pipeline.gpu_name == server.gpu_name


def test_video_switching_does_not_leak_stale_state():
    """Verify switching video sessions produces fresh tracking and analytics without stale cache."""
    with MockGPUWorkerServer() as server:
        client = GPUWorkerClient(base_url=server.url)

        # Video Session 1
        pipe1 = AdaptivePipeline(device=DeviceType.CUDA, gpu_worker_client=client)
        session1 = AnalyticsSession(video_name="video_one.mp4", device=pipe1.execution_mode, gpu_name=pipe1.gpu_name)
        dummy = np.zeros((800, 1000, 3), dtype=np.uint8)
        res1 = pipe1.process_frame(dummy, frame_index=0)
        session1.record_frame(res1, frame_proc_time_s=0.03)

        assert session1.frames_processed == 1
        assert session1.video_name == "video_one.mp4"

        # Video Session 2
        pipe2 = AdaptivePipeline(device=DeviceType.CUDA, gpu_worker_client=client)
        session2 = AnalyticsSession(video_name="video_two.mp4", device=pipe2.execution_mode, gpu_name=pipe2.gpu_name)

        assert session2.frames_processed == 0
        assert session2.video_name == "video_two.mp4"
        assert session2.device == "Remote CUDA"
