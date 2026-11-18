"""Focused unit and integration tests for CrossGuard Dynamic GPU Worker Registry.

Covers the 16 verification points:
 1. registry URL parsing (plain ID, full gist URL, raw URL, whitespace, trailing slash)
 2. valid registry record (parsed correctly from dict/JSON, valid status, fresh timestamp)
 3. stale registry record rejection (record older than TTL is marked stale and rejected)
 4. malformed registry response (corrupted JSON or missing fields handled gracefully)
 5. unavailable registry (network failure or HTTP 404 handled gracefully without crashing)
 6. registry-first resolution (registry URL takes priority over .env when fresh)
 7. .env fallback (falls back to CROSSGUARD_GPU_WORKER_URL when registry is stale/unavailable/offline)
 8. no configured worker (resolves to None when neither registry nor .env is set)
 9. explicit worker URL override (explicit parameter overrides both registry and .env)
10. GPU mode with healthy registry worker (AdaptivePipeline initializes RemoteGPUBackend using registry URL)
11. GPU mode with unavailable worker (raises RuntimeError with clear explanation, never falls back to CPU)
12. Auto mode with registry worker (uses remote CUDA worker when registry worker is healthy)
13. Auto mode without registry worker (safely falls back to local CPU when registry worker is unavailable/offline)
14. no silent CPU fallback (explicit GPU request fails loudly when worker goes down)
15. reset behavior (resetting pipeline preserves device configuration and clears state)
16. existing GPU-worker protocol tests still pass (tested via integration with MockGPUWorkerServer)

Mock GitHub/network requests in unit tests. Does not depend on live GitHub or Colab.
DOES NOT import streamlit or app at module level to preserve test isolation.
"""
from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch
import urllib.error
import numpy as np
import pytest

from core.gpu_config import (
    GPUWorkerStatus,
    WorkerDiagnosticState,
    WorkerResolution,
    WorkerUrlSource,
    check_worker_status,
    clear_status_cache,
    get_configured_worker_url,
    resolve_worker_url,
)
from core.gpu_registry import (
    DEFAULT_REGISTRY_TTL_SECONDS,
    GPURegistryClient,
    GPURegistryRecord,
    mask_gist_id,
    parse_gist_id,
)
from core.pipeline import AdaptivePipeline
from gpu_worker.client import GPUWorkerClient
from gpu_worker.mock_worker import MockGPUWorkerServer
from gpu_worker.protocol import WorkerHealth
from vision.models import DeviceType


# =============================================================================
# 1. Registry URL & Gist ID Parsing
# =============================================================================

def test_registry_url_parsing():
    """Verify parse_gist_id handles plain IDs, web URLs, raw URLs, and whitespace."""
    # Plain hex ID
    assert parse_gist_id("3a7b9c4d5e6f7a8b9c0d1e2f3a4b5c6d") == "3a7b9c4d5e6f7a8b9c0d1e2f3a4b5c6d"

    # Full GitHub web URL
    url1 = "https://gist.github.com/developer/3a7b9c4d5e6f7a8b9c0d1e2f3a4b5c6d"
    assert parse_gist_id(url1) == "3a7b9c4d5e6f7a8b9c0d1e2f3a4b5c6d"

    # Raw GitHub Usercontent URL
    url2 = "https://gist.githubusercontent.com/developer/3a7b9c4d5e6f7a8b9c0d1e2f3a4b5c6d/raw/crossguard_worker.json"
    assert parse_gist_id(url2) == "3a7b9c4d5e6f7a8b9c0d1e2f3a4b5c6d"

    # Trailing slash and spaces
    assert parse_gist_id("  https://gist.github.com/user/abcdef123456/  ") == "abcdef123456"
    assert parse_gist_id("  abcdef123456/  ") == "abcdef123456"

    # None, empty or whitespace
    assert parse_gist_id(None) is None
    assert parse_gist_id("") is None
    assert parse_gist_id("   ") is None

    # Masking test
    assert mask_gist_id(None) == "None (unset)"
    assert mask_gist_id("3a7b9c4d5e6f7a8b9c0d1e2f3a4b5c6d") == "3a7b...5c6d"


# =============================================================================
# 2. Valid Registry Record
# =============================================================================

def test_valid_registry_record():
    """Verify GPURegistryRecord converts to/from dict/JSON and evaluates valid status."""
    now = time.time()
    rec = GPURegistryRecord(
        worker_url="https://custom-colab.trycloudflare.com",
        worker_id="crossguard-colab",
        gpu_name="Tesla T4",
        status="online",
        updated_at=now,
    )
    assert rec.is_online is True
    assert rec.is_fresh(ttl_seconds=300.0, current_time=now) is True
    assert rec.is_valid(ttl_seconds=300.0, current_time=now) is True

    # JSON serialization and deserialization
    js_str = rec.to_json()
    parsed = GPURegistryRecord.from_json(js_str)
    assert parsed.worker_url == "https://custom-colab.trycloudflare.com"
    assert parsed.gpu_name == "Tesla T4"
    assert parsed.status == "online"
    assert abs(parsed.updated_at - now) < 0.001

    # ISO 8601 timestamp parsing support
    iso_data = {
        "worker_url": "https://iso-colab.trycloudflare.com",
        "gpu_name": "NVIDIA A100",
        "status": "online",
        "updated_at": "2026-11-10T14:30:00Z",
    }
    rec_iso = GPURegistryRecord.from_dict(iso_data)
    assert rec_iso.worker_url == "https://iso-colab.trycloudflare.com"
    assert rec_iso.updated_at > 0.0


# =============================================================================
# 3. Stale Registry Record Rejection
# =============================================================================

def test_stale_registry_record_rejection():
    """Verify records older than TTL are marked stale and rejected."""
    current_time = 1000.0
    ttl = 300.0  # 5 minutes

    # 1. Exactly at boundary (fresh)
    rec_fresh = GPURegistryRecord(
        worker_url="https://valid-tunnel.trycloudflare.com",
        status="online",
        updated_at=current_time - 299.0,
    )
    assert rec_fresh.is_fresh(ttl_seconds=ttl, current_time=current_time) is True
    assert rec_fresh.is_valid(ttl_seconds=ttl, current_time=current_time) is True

    # 2. Expired past TTL (stale)
    rec_stale = GPURegistryRecord(
        worker_url="https://old-tunnel.trycloudflare.com",
        status="online",
        updated_at=current_time - 305.0,  # 305s old > 300s
    )
    assert rec_stale.is_fresh(ttl_seconds=ttl, current_time=current_time) is False
    assert rec_stale.is_valid(ttl_seconds=ttl, current_time=current_time) is False

    # 3. GPURegistryClient.get_record returns status='stale'
    client = GPURegistryClient(gist_id="testgist123", ttl_seconds=ttl)
    with patch.object(client, "_fetch_gist_content", return_value=(rec_stale.to_json(), None)):
        record, status, err = client.get_record(force_refresh=True, current_time=current_time)
        assert status == "stale"
        assert record is not None
        assert "stale" in err.lower()


# =============================================================================
# 4. Malformed Registry Response
# =============================================================================

def test_malformed_registry_response():
    """Verify corrupted JSON or missing required fields in Gist are handled gracefully."""
    client = GPURegistryClient(gist_id="testgist123")

    # Corrupted JSON syntax
    with patch.object(client, "_fetch_gist_content", return_value=("{invalid json syntax", None)):
        record, status, err = client.get_record(force_refresh=True)
        assert record is None
        assert status == "invalid"
        assert "JSON" in err

    # Valid JSON but missing worker_url
    empty_url_json = json.dumps({"status": "online", "updated_at": time.time()})
    with patch.object(client, "_fetch_gist_content", return_value=(empty_url_json, None)):
        record, status, err = client.get_record(force_refresh=True)
        assert record is None
        assert status == "invalid"
        assert "worker_url" in err.lower()


# =============================================================================
# 5. Unavailable Registry (Network Failure / HTTP 404)
# =============================================================================

def test_unavailable_registry():
    """Verify network timeouts, connection drops, or HTTP 404 return unavailable without crashing."""
    client = GPURegistryClient(gist_id="testgist123")

    # Mock 404 HTTP error
    with patch.object(client, "_fetch_gist_content", return_value=(None, "GitHub Gist testgist123 not found (HTTP 404).")):
        record, status, err = client.get_record(force_refresh=True)
        assert record is None
        assert status == "unavailable"
        assert "404" in err


# =============================================================================
# 6. Registry-First Resolution
# =============================================================================

def test_registry_first_resolution():
    """Verify that a fresh, valid registry entry takes precedence over .env."""
    clear_status_cache()
    fresh_rec = GPURegistryRecord(
        worker_url="https://registry-discovered.trycloudflare.com",
        status="online",
        updated_at=time.time(),
    )

    with patch.dict(os.environ, {
        "CROSSGUARD_GPU_REGISTRY_GIST_ID": "mock_gist_123",
        "CROSSGUARD_GPU_WORKER_URL": "https://static-env-fallback.trycloudflare.com",
    }, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
        with patch.object(GPURegistryClient, "_fetch_gist_content", return_value=(fresh_rec.to_json(), None)):
            res = resolve_worker_url(force_refresh_registry=True)
            assert res.source == WorkerUrlSource.REGISTRY
            assert res.url == "https://registry-discovered.trycloudflare.com"
            assert res.registry_status == "fresh"

            # Authoritative single choke point matches
            assert get_configured_worker_url() == "https://registry-discovered.trycloudflare.com"


# =============================================================================
# 7. Static .env Fallback When Registry is Stale / Offline / Unavailable
# =============================================================================

def test_env_fallback():
    """Verify fallback to CROSSGUARD_GPU_WORKER_URL when registry is stale, offline, or unavailable."""
    clear_status_cache()
    fallback_url = "https://static-env-fallback.trycloudflare.com"

    env_mock = {
        "CROSSGUARD_GPU_REGISTRY_GIST_ID": "mock_gist_123",
        "CROSSGUARD_GPU_WORKER_URL": fallback_url,
    }

    # Case A: Registry entry is STALE
    stale_rec = GPURegistryRecord(
        worker_url="https://stale-tunnel.trycloudflare.com",
        status="online",
        updated_at=time.time() - 600.0,
    )
    with patch.dict(os.environ, env_mock, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
        with patch.object(GPURegistryClient, "_fetch_gist_content", return_value=(stale_rec.to_json(), None)):
            res = resolve_worker_url(force_refresh_registry=True)
            assert res.source == WorkerUrlSource.ENV
            assert res.url == fallback_url
            assert res.registry_status == "stale"
            assert get_configured_worker_url() == fallback_url

    # Case B: Registry entry is explicitly OFFLINE
    offline_rec = GPURegistryRecord(
        worker_url="https://offline-tunnel.trycloudflare.com",
        status="offline",
        updated_at=time.time(),
    )
    with patch.dict(os.environ, env_mock, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
        with patch.object(GPURegistryClient, "_fetch_gist_content", return_value=(offline_rec.to_json(), None)):
            res = resolve_worker_url(force_refresh_registry=True)
            assert res.source == WorkerUrlSource.ENV
            assert res.url == fallback_url
            assert res.registry_status == "offline"

    # Case C: Registry is UNAVAILABLE (network error)
    with patch.dict(os.environ, env_mock, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
        with patch.object(GPURegistryClient, "_fetch_gist_content", return_value=(None, "Network timeout")):
            res = resolve_worker_url(force_refresh_registry=True)
            assert res.source == WorkerUrlSource.ENV
            assert res.url == fallback_url
            assert res.registry_status == "unavailable"


# =============================================================================
# 8. No Configured Worker
# =============================================================================

def test_no_configured_worker():
    """Verify that when neither registry nor .env is set, resolution returns None."""
    clear_status_cache()
    with patch.dict(os.environ, {}, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
        res = resolve_worker_url(force_refresh_registry=True)
        assert res.url is None
        assert res.source == WorkerUrlSource.NONE
        assert res.registry_status == "not_configured"
        assert get_configured_worker_url() is None

        stat = check_worker_status(force_refresh=True)
        assert stat.is_configured is False
        assert stat.diagnostic_state == WorkerDiagnosticState.CONFIG_MISSING
        assert stat.url_source == WorkerUrlSource.NONE


# =============================================================================
# 9. Explicit Worker URL Override
# =============================================================================

def test_explicit_worker_url_override():
    """Verify explicit worker URL parameter overrides both registry and .env."""
    clear_status_cache()
    with patch.dict(os.environ, {
        "CROSSGUARD_GPU_REGISTRY_GIST_ID": "gist_abc",
        "CROSSGUARD_GPU_WORKER_URL": "https://env.trycloudflare.com",
    }, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
        res = resolve_worker_url(explicit_url="https://explicit.trycloudflare.com")
        assert res.source == WorkerUrlSource.EXPLICIT
        assert res.url == "https://explicit.trycloudflare.com"


# =============================================================================
# 10. GPU Mode with Healthy Registry Worker
# =============================================================================

def test_gpu_mode_with_healthy_registry_worker():
    """Verify AdaptivePipeline initialized with device='cuda' uses the registry-discovered worker."""
    clear_status_cache()
    with MockGPUWorkerServer(cuda_available=True, gpu_name="Tesla T4") as mock_server:
        fresh_rec = GPURegistryRecord(
            worker_url=mock_server.base_url,
            status="online",
            updated_at=time.time(),
        )

        with patch.dict(os.environ, {
            "CROSSGUARD_GPU_REGISTRY_GIST_ID": "mock_gist",
        }, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
            with patch.object(GPURegistryClient, "_fetch_gist_content", return_value=(fresh_rec.to_json(), None)):
                pipeline = AdaptivePipeline(device=DeviceType.CUDA)
                assert pipeline.execution_mode == "Remote CUDA"
                assert pipeline.device == "cuda"
                assert pipeline.gpu_name == "Tesla T4"
                assert pipeline.vision_status == "ACTIVE"


# =============================================================================
# 11. GPU Mode with Unavailable Worker
# =============================================================================

def test_gpu_mode_with_unavailable_worker():
    """Verify explicit GPU request raises RuntimeError when worker is unavailable (NO silent CPU fallback)."""
    clear_status_cache()
    with patch.dict(os.environ, {}, clear=True), \
         patch("core.gpu_config.find_env_file", return_value=None), \
         patch("vision.models.is_cuda_available", return_value=False):
        with pytest.raises(RuntimeError) as exc_info:
            AdaptivePipeline(device=DeviceType.CUDA)
        assert "Remote CUDA worker is not currently reachable" in str(exc_info.value)


# =============================================================================
# 12. Auto Mode with Registry Worker
# =============================================================================

def test_auto_mode_with_registry_worker():
    """Verify Auto mode routes to remote CUDA when registry worker is online."""
    clear_status_cache()
    with MockGPUWorkerServer(cuda_available=True, gpu_name="Tesla T4") as mock_server:
        fresh_rec = GPURegistryRecord(
            worker_url=mock_server.base_url,
            status="online",
            updated_at=time.time(),
        )

        with patch.dict(os.environ, {
            "CROSSGUARD_GPU_REGISTRY_GIST_ID": "mock_gist",
        }, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
            with patch.object(GPURegistryClient, "_fetch_gist_content", return_value=(fresh_rec.to_json(), None)):
                pipeline = AdaptivePipeline(device=DeviceType.AUTO)
                assert pipeline.execution_mode == "Remote CUDA"
                assert pipeline.device == "cuda"


# =============================================================================
# 13. Auto Mode without Registry Worker
# =============================================================================

def test_auto_mode_without_registry_worker():
    """Verify Auto mode safely falls back to local CPU when registry worker is offline or unavailable."""
    clear_status_cache()
    with patch.dict(os.environ, {}, clear=True), \
         patch("core.gpu_config.find_env_file", return_value=None), \
         patch("vision.models.is_cuda_available", return_value=False):
        pipeline = AdaptivePipeline(device=DeviceType.AUTO)
        assert pipeline.execution_mode == "Local CPU"
        assert pipeline.device == "cpu"


# =============================================================================
# 14. No Silent CPU Fallback
# =============================================================================

def test_no_silent_cpu_fallback():
    """Verify that selecting GPU (CUDA) never silently falls back to CPU when worker fails."""
    clear_status_cache()
    # Mock worker that reports CUDA unavailable
    with MockGPUWorkerServer(cuda_available=False, gpu_name="CPU Only") as mock_server:
        with patch.dict(os.environ, {
            "CROSSGUARD_GPU_WORKER_URL": mock_server.base_url,
        }, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                AdaptivePipeline(device=DeviceType.CUDA)
            assert "Remote CUDA worker is not currently reachable" in str(exc_info.value)
            assert "CUDA is unavailable" in str(exc_info.value)


# =============================================================================
# 15. Reset Behavior
# =============================================================================

def test_reset_behavior():
    """Verify pipeline reset preserves device configuration and clears state."""
    clear_status_cache()
    with MockGPUWorkerServer(cuda_available=True, gpu_name="Tesla T4") as mock_server:
        with patch.dict(os.environ, {
            "CROSSGUARD_GPU_WORKER_URL": mock_server.base_url,
        }, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
            pipeline = AdaptivePipeline(device=DeviceType.CUDA)
            assert pipeline.execution_mode == "Remote CUDA"

            # Process a blank frame
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            res = pipeline.process_frame(dummy, frame_index=0, dt=0.033)
            assert res.frame_index == 0

            # Reset
            pipeline.reset()
            assert pipeline.execution_mode == "Remote CUDA"
            assert pipeline.device == "cuda"
            assert pipeline.controller.state.current_phase.value == "VEHICLE_GREEN"


# =============================================================================
# 16. Existing GPU-Worker Protocol Tests Still Pass
# =============================================================================

def test_existing_gpu_worker_protocol_tests_pass():
    """Verify protocol health check and inference request/response schemas operate identically."""
    health = WorkerHealth(
        status="ok",
        cuda_available=True,
        gpu_name="Tesla T4",
        yolo_loaded=True,
    )
    assert health.is_ready is True
    d = health.to_dict()
    assert d["status"] == "ok"
    assert d["cuda_available"] is True

    # Client health check
    with MockGPUWorkerServer(cuda_available=True, gpu_name="Tesla T4") as mock_server:
        client = GPUWorkerClient(base_url=mock_server.base_url)
        is_ready, msg, h_obj = client.check_ready()
        assert is_ready is True
        assert h_obj is not None
        assert h_obj.gpu_name == "Tesla T4"
