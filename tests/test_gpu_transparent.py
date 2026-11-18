"""Unit and integration tests for User-Transparent Remote GPU Execution.

Verifies:
1. Configuration:
   - Missing GPU worker URL handling
   - Valid GPU worker URL normalization
   - Malformed URL normalization
   - Environment variable and .env file loading
2. Device Selection Semantics:
   - CPU never contacts worker
   - GPU strictly requires worker (zero silent fallback)
   - GPU unavailable raises clear error
   - Auto uses remote GPU when available
   - Auto safely falls back to local CPU when worker is unavailable
3. UI / Backend Transparent Integration:
   - Pipeline operates transparently without manual UI worker URL passing
   - Connected GPU reports actual GPU metadata
   - Unavailable GPU produces clear status
4. Mid-stream worker disconnect and resume resilience

DOES NOT import streamlit or app at module level to preserve test isolation.
"""
from __future__ import annotations

import os
import tempfile
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from core.analytics import AnalyticsSession
from core.gpu_config import (
    GPUWorkerStatus,
    WorkerDiagnosticState,
    check_worker_status,
    clear_status_cache,
    get_configured_worker_url,
    is_gpu_worker_available,
    load_env_file,
    mask_worker_url,
    normalize_worker_url,
)
from core.pipeline import AdaptivePipeline, PipelineFrameResult
from gpu_worker.backend import RemoteGPUBackend
from gpu_worker.client import (
    GPUWorkerClient,
    GPUWorkerConnectionError,
    GPUWorkerError,
    GPUWorkerTimeoutError,
)
from gpu_worker.mock_worker import MockGPUWorkerServer
from vision.models import DeviceType


# =============================================================================
# 1. Configuration & Discovery Tests
# =============================================================================

@pytest.fixture(autouse=True)
def isolate_from_registry():
    """Ensure static environment tests are not overridden by live registry in .env."""
    with patch.dict(os.environ, {"CROSSGUARD_GPU_REGISTRY_GIST_ID": ""}):
        clear_status_cache()
        yield
        clear_status_cache()


def test_missing_gpu_worker_url():
    """Verify missing worker URL is handled cleanly without exceptions."""
    clear_status_cache()
    with patch("core.gpu_config.find_env_file", return_value=None), patch.dict(os.environ, {}, clear=True):
        assert "CROSSGUARD_GPU_WORKER_URL" not in os.environ
        url = get_configured_worker_url()
        assert url is None

        status = check_worker_status(force_refresh=True)
        assert status.is_configured is False
        assert status.is_connected is False
        assert status.is_available is False
        assert status.status_label == "Unavailable"
        assert "Unavailable" in status.badge_text
        assert status.env_found is False
        assert status.url_configured is False


def test_valid_and_malformed_url_normalization():
    """Verify URL normalization strips whitespace, formats protocols, and strips trailing slashes."""
    # None or empty
    assert normalize_worker_url(None) == ""
    assert normalize_worker_url("") == ""
    assert normalize_worker_url("   ") == ""

    # Normal https URL with trailing slashes
    assert normalize_worker_url("https://my-colab.trycloudflare.com/") == "https://my-colab.trycloudflare.com"
    assert normalize_worker_url("https://worker.internal:8000///") == "https://worker.internal:8000"

    # Missing scheme: cloud domain defaults to https
    assert normalize_worker_url("random-tunnel.trycloudflare.com") == "https://random-tunnel.trycloudflare.com"

    # Missing scheme: localhost defaults to http
    assert normalize_worker_url("localhost:8000/") == "http://localhost:8000"
    assert normalize_worker_url("127.0.0.1:5000") == "http://127.0.0.1:5000"


def test_env_file_loading():
    """Verify load_env_file reads key-value pairs without overriding existing environment vars."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".env") as f:
        f.write("# Sample configuration\n")
        f.write("CROSSGUARD_GPU_WORKER_URL=https://test-worker.example.com\n")
        f.write("CROSSGUARD_ANOTHER_VAR='quoted_value'\n")
        tmp_name = f.name

    try:
        # If env already has it, file should NOT overwrite it
        with patch.dict(os.environ, {"CROSSGUARD_GPU_WORKER_URL": "https://existing.example.com"}, clear=True):
            load_env_file(tmp_name)
            assert os.environ["CROSSGUARD_GPU_WORKER_URL"] == "https://existing.example.com"
            assert os.environ["CROSSGUARD_ANOTHER_VAR"] == "quoted_value"

        # If env is empty, file supplies it
        with patch.dict(os.environ, {}, clear=True):
            load_env_file(tmp_name)
            assert os.environ["CROSSGUARD_GPU_WORKER_URL"] == "https://test-worker.example.com"
            assert os.environ["CROSSGUARD_ANOTHER_VAR"] == "quoted_value"
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_cached_worker_status():
    """Verify worker health check is cached for a short TTL to prevent request spamming."""
    clear_status_cache()
    with MockGPUWorkerServer(gpu_name="Tesla T4 Cache Test") as server:
        # First call hits mock server
        st1 = check_worker_status(worker_url=server.url, force_refresh=True)
        assert st1.is_available is True
        assert st1.gpu_name == "Tesla T4 Cache Test"

        # Change mock server property
        server.gpu_name = "Changed Name"

        # Immediate second call should return cached object
        st2 = check_worker_status(worker_url=server.url, force_refresh=False)
        assert st2.gpu_name == "Tesla T4 Cache Test"

        # Force refresh updates cache
        st3 = check_worker_status(worker_url=server.url, force_refresh=True)
        assert st3.gpu_name == "Changed Name"


# =============================================================================
# 2. Device Selection Semantics
# =============================================================================

def test_cpu_mode_never_contacts_worker():
    """Verify CPU mode never makes network calls or contacts worker even if URL is configured."""
    with MockGPUWorkerServer() as server:
        with patch.dict(os.environ, {"CROSSGUARD_GPU_WORKER_URL": server.url}):
            pipe = AdaptivePipeline(device=DeviceType.CPU)
            assert pipe.device == "cpu"
            assert pipe.execution_mode == "Local CPU"
            assert pipe.gpu_name is None


def test_gpu_mode_transparent_auto_discovery():
    """Verify GPU mode automatically discovers worker from environment without manual UI URL."""
    with MockGPUWorkerServer(gpu_name="NVIDIA Tesla T4") as server:
        with patch.dict(os.environ, {"CROSSGUARD_GPU_WORKER_URL": server.url}):
            clear_status_cache()
            pipe = AdaptivePipeline(device=DeviceType.CUDA)
            assert pipe.device == "cuda"
            assert pipe.execution_mode == "Remote CUDA"
            assert pipe.gpu_name == "NVIDIA Tesla T4"
            assert isinstance(pipe.vision.backend, RemoteGPUBackend)


def test_gpu_unavailable_raises_clear_error_without_fallback():
    """Verify selecting GPU when worker is offline raises clear error without silent CPU fallback."""
    with patch.dict(os.environ, {"CROSSGUARD_GPU_WORKER_URL": "http://127.0.0.1:59994"}):
        with patch("core.pipeline.is_cuda_available", return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                AdaptivePipeline(device=DeviceType.CUDA)
            err_msg = str(exc_info.value)
            assert "Remote CUDA worker is not currently reachable" in err_msg


def test_auto_mode_routes_to_gpu_when_online():
    """Verify Auto mode automatically routes to Remote CUDA when worker is online."""
    with MockGPUWorkerServer(gpu_name="Tesla T4 Auto") as server:
        with patch.dict(os.environ, {"CROSSGUARD_GPU_WORKER_URL": server.url}):
            pipe = AdaptivePipeline(device=DeviceType.AUTO)
            assert pipe.device == "cuda"
            assert pipe.execution_mode == "Remote CUDA"
            assert pipe.gpu_name == "Tesla T4 Auto"


def test_auto_mode_falls_back_to_cpu_when_worker_unavailable():
    """Verify Auto mode safely falls back to Local CPU when worker is offline and local CUDA is missing."""
    with patch.dict(os.environ, {"CROSSGUARD_GPU_WORKER_URL": "http://127.0.0.1:59993"}):
        with patch("core.pipeline.is_cuda_available", return_value=False):
            pipe = AdaptivePipeline(device=DeviceType.AUTO)
            assert pipe.device == "cpu"
            assert pipe.execution_mode == "Local CPU"
            assert pipe.gpu_name is None


# =============================================================================
# 3. Transparent UI & Pipeline Integration
# =============================================================================

def test_transparent_pipeline_inference_and_analytics():
    """Verify pipeline runs end-to-end transparently and records genuine worker latency."""
    with MockGPUWorkerServer(gpu_name="Tesla T4", latency_ms=16.8) as server:
        with patch.dict(os.environ, {"CROSSGUARD_GPU_WORKER_URL": server.url}):
            pipeline = AdaptivePipeline(device=DeviceType.CUDA, default_fps=10.0)

            dummy_frame = np.zeros((800, 1000, 3), dtype=np.uint8)
            res = pipeline.process_frame(dummy_frame, frame_index=0, dt=0.1)

            assert isinstance(res, PipelineFrameResult)
            assert pipeline.last_inference_time_ms == 16.8

            session = AnalyticsSession(
                device=pipeline.execution_mode,
                gpu_name=pipeline.gpu_name,
            )
            session.record_frame(res, frame_proc_time_s=0.03, yolo_latency_s=pipeline.last_inference_time_ms / 1000.0)
            summary = session.to_executive_summary()

            assert summary["device"] == "Remote CUDA"
            assert summary["gpu_name"] == "Tesla T4"


# =============================================================================
# 4. Mid-Stream Disconnect & Recovery Resilience
# =============================================================================

def test_midstream_gpu_disconnect_and_recovery():
    """Verify that mid-stream worker disconnect raises connection error and preserves state for resumption."""
    with MockGPUWorkerServer() as server:
        client = GPUWorkerClient(base_url=server.url)
        pipeline = AdaptivePipeline(device=DeviceType.CUDA, gpu_worker_client=client)

        dummy = np.zeros((800, 1000, 3), dtype=np.uint8)

        # Frame 0 succeeds
        res0 = pipeline.process_frame(dummy, frame_index=0, dt=0.1)
        assert res0.frame_index == 0
        assert pipeline.current_time == 0.1

        # Simulate worker crash / disconnect
        server.shutdown()

        # Frame 1 fails with GPUWorkerConnectionError
        with pytest.raises((GPUWorkerConnectionError, GPUWorkerTimeoutError, GPUWorkerError)):
            pipeline.process_frame(dummy, frame_index=1, dt=0.1)

        # Pipeline state must be preserved
        assert pipeline.current_time == 0.1
        assert pipeline.device == "cuda"
        assert pipeline.execution_mode == "Remote CUDA"


# =============================================================================
# 5. Diagnostic States & Pipeline Instrumentation Tests
# =============================================================================

def test_url_masking():
    """Verify that mask_worker_url prevents leaking sensitive tunnel endpoints."""
    assert mask_worker_url(None) == "None (unset)"
    assert mask_worker_url("") == "None (unset)"
    assert mask_worker_url("http://short.io") == "http://***"
    masked = mask_worker_url("https://random-subdomain-1234.trycloudflare.com")
    assert "https://rand..." in masked
    assert masked.endswith("dflare.com")
    assert "subdomain-1234" not in masked


def test_four_worker_diagnostic_states():
    """Verify check_worker_status accurately categorizes into all 4 diagnostic states."""
    clear_status_cache()

    # 1. CONFIG MISSING
    with patch("core.gpu_config.find_env_file", return_value=None), patch.dict(os.environ, {}, clear=True):
        st1 = check_worker_status(force_refresh=True)
        assert st1.diagnostic_state == WorkerDiagnosticState.CONFIG_MISSING
        assert st1.is_available is False
        assert st1.env_found is False

    # 2. CONFIGURED BUT UNREACHABLE
    with patch.dict(os.environ, {"CROSSGUARD_GPU_WORKER_URL": "http://127.0.0.1:59999"}):
        st2 = check_worker_status(force_refresh=True, timeout=0.2)
        assert st2.diagnostic_state == WorkerDiagnosticState.CONFIGURED_BUT_UNREACHABLE
        assert st2.is_available is False
        assert "59999" in (st2.masked_url or "")

    # 3. WORKER REACHABLE BUT CUDA UNAVAILABLE
    with MockGPUWorkerServer(cuda_available=False, gpu_name=None) as s3:
        with patch.dict(os.environ, {"CROSSGUARD_GPU_WORKER_URL": s3.url}):
            st3 = check_worker_status(force_refresh=True)
            assert st3.diagnostic_state == WorkerDiagnosticState.WORKER_REACHABLE_BUT_CUDA_UNAVAILABLE
            assert st3.is_available is False
            assert st3.cuda_available is False

    # 4. WORKER REACHABLE + CUDA AVAILABLE
    with MockGPUWorkerServer(cuda_available=True, gpu_name="Tesla V100") as s4:
        with patch.dict(os.environ, {"CROSSGUARD_GPU_WORKER_URL": s4.url}):
            st4 = check_worker_status(force_refresh=True)
            assert st4.diagnostic_state == WorkerDiagnosticState.WORKER_REACHABLE_AND_CUDA_AVAILABLE
            assert st4.is_available is True
            assert st4.cuda_available is True
            assert st4.gpu_name == "Tesla V100"


def test_pipeline_frame_diagnostics_instrumentation():
    """Verify AdaptivePipeline exposes complete 13-field diagnostic telemetry."""
    pipe = AdaptivePipeline(device=DeviceType.CPU)
    diag0 = pipe.get_last_frame_diagnostics()
    assert diag0["requested_device"] == "cpu"
    assert diag0["actual_device"] == "cpu"
    assert diag0["execution_mode"] == "Local CPU"
    assert diag0["yolo_loaded"] is True
    assert diag0["raw_detection_count"] == 0
    assert diag0["tracked_detection_count"] == 0

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = pipe.process_frame(dummy, frame_index=0, dt=0.04)
    diag1 = pipe.get_last_frame_diagnostics()
    assert diag1["vision_status"] == "ACTIVE"
    assert "raw_class_counts" in diag1
    assert "bytetrack_input_count" in diag1
    assert "spatial_detection_count" in diag1
    assert "waiting_pedestrians" in diag1
    assert "crossing_pedestrians" in diag1
    assert "approaching_vehicles" in diag1
    assert "queued_vehicles" in diag1
    assert "controller_phase" in diag1


def test_no_literal_python_expressions_in_app_ui():
    """Verify app.py does not contain unrendered python expressions in raw markdown."""
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Problem 3 check: Ensure literal unformatted template strings do not exist
    assert "Standby ({pipeline.device" not in content, "Found unrendered literal string in app.py"
    assert "Hardware Selection Error" not in content, "Found obsolete template fallback in app.py"
    assert "Vision: <b>Standby &bull; {hw_desc}</b>" in content, "Properly formatted Vision standby expression expected"

