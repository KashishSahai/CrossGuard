"""The exact GPU-configuration resolution path: .env -> Gist ID -> registry fetch -> worker URL -> UI state.

Regression for: a valid CROSSGUARD_GPU_REGISTRY_GIST_ID in .env with no currently registered worker was
reported as "CONFIG MISSING". It must be "CONFIGURED BUT UNREACHABLE" (with the specific reason), and
CONFIG MISSING must be reserved for "no registry and no static URL".
Never touches the network (urllib is mocked). Registry precedence itself is covered in test_gpu_registry.py.
"""
from __future__ import annotations

import functools
import json
import os
import time
import urllib.error
from unittest.mock import patch

import pytest

from core.gpu_config import (WorkerDiagnosticState, WorkerUrlSource, check_worker_status,
                             clear_status_cache, get_env_diagnostic_summary, resolve_worker_url)
from core.pipeline import AdaptivePipeline
from vision.models import DeviceType

def _clean(fn):
    """Clear the cached worker status/registry client before AND after, so no test leaks state."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        clear_status_cache()
        try:
            return fn(*a, **k)
        finally:
            clear_status_cache()
    return wrapper


GIST = "3bf1ca01d7d46d5af6ef5a099feb9cf5"
WORKER = "https://reg-worker.trycloudflare.com"


class _Resp:
    status = 200

    def __init__(self, body: bytes):
        self._b = body

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _gist(files: dict) -> _Resp:
    return _Resp(json.dumps({"files": files}).encode())


def _record(**kw) -> dict:
    d = dict(worker_url=WORKER, worker_id="crossguard-colab", gpu_name="Tesla T4", status="online",
             updated_at=time.time(), protocol_version="1", heartbeat_interval_s=60)
    d.update(kw)
    return d


def _with_gist(opener):
    """Environment with ONLY the Gist ID (no token, no static URL), network replaced by `opener`."""
    clear_status_cache()
    return (patch.dict(os.environ, {"CROSSGUARD_GPU_REGISTRY_GIST_ID": GIST}, clear=True),
            patch("core.gpu_config.find_env_file", return_value=None),
            patch("urllib.request.urlopen", opener))


def _status(opener):
    a, b, c = _with_gist(opener)
    with a, b, c:
        return check_worker_status(force_refresh=True)


# ------------------------------------------------------------------ 1-2: .env is loaded and read
@_clean
def test_env_file_is_loaded_before_gpu_configuration_is_evaluated(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"CROSSGUARD_GPU_REGISTRY_GIST_ID={GIST}\n", encoding="utf-8")
    clear_status_cache()
    with patch.dict(os.environ, {}, clear=True), patch("core.gpu_config.find_env_file", return_value=env_file):
        assert "CROSSGUARD_GPU_REGISTRY_GIST_ID" not in os.environ
        diag = get_env_diagnostic_summary()
        assert os.environ["CROSSGUARD_GPU_REGISTRY_GIST_ID"] == GIST
        assert diag["registry_configured"] is True and GIST not in str(diag["registry_gist_id"])  # masked


@_clean
def test_resolution_loads_the_env_file_itself_without_a_prior_call(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f'CROSSGUARD_GPU_REGISTRY_GIST_ID="{GIST}"\n', encoding="utf-8")   # quoted value
    requested = []

    def opener(req, timeout=None):
        requested.append(req.full_url)
        return _gist({"crossguard_worker.json": {"content": json.dumps(_record())}})

    clear_status_cache()
    with patch.dict(os.environ, {}, clear=True), patch("core.gpu_config.find_env_file", return_value=env_file), \
            patch("urllib.request.urlopen", opener):
        r = resolve_worker_url(force_refresh_registry=True)
    assert requested == [f"https://api.github.com/gists/{GIST}"]
    assert (r.source, r.url) == (WorkerUrlSource.REGISTRY, WORKER)                    # 6: registered URL is effective


# ------------------------------------------------------------------ 4, 7-8: registry read needs no token
@_clean
def test_registry_is_read_without_any_github_token():
    seen = []

    def opener(req, timeout=None):
        seen.append(req.get_header("Authorization"))
        return _gist({"crossguard_worker.json": {"content": json.dumps(_record())}})

    a, b, c = _with_gist(opener)
    with a, b, c:
        assert resolve_worker_url(force_refresh_registry=True).url == WORKER
    assert seen == [None]                                   # no Authorization header was needed or sent


@_clean
def test_env_files_in_the_repo_never_hold_the_github_token():
    for name in (".env", ".env.example"):
        if os.path.exists(name):
            active = [ln for ln in open(name, encoding="utf-8").read().splitlines()
                      if ln.strip() and not ln.lstrip().startswith("#")]
            assert not any(ln.split("=", 1)[0].strip() == "CROSSGUARD_GPU_REGISTRY_GITHUB_TOKEN" for ln in active), name
            assert not any("trycloudflare.com" in ln for ln in active), name              # 9: no hardcoded tunnel URL


# ------------------------------------------------------------------ 5: configured-but-empty is not CONFIG MISSING
def _nf(req, timeout=None):
    raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)


def _down(req, timeout=None):
    raise urllib.error.URLError("network down")


SITUATIONS = {
    "gist has no worker file yet": lambda r, timeout=None: _gist({"README.md": {"content": "x"}}),
    "worker offline": lambda r, timeout=None: _gist({"crossguard_worker.json": {"content": json.dumps(_record(status="offline"))}}),
    "worker stale": lambda r, timeout=None: _gist({"crossguard_worker.json": {"content": json.dumps(_record(updated_at=time.time() - 3600))}}),
    "gist id wrong (404)": _nf,
    "github unreachable": _down,
    "untrusted url": lambda r, timeout=None: _gist({"crossguard_worker.json": {"content": json.dumps(_record(worker_url="https://attacker.example.com"))}}),
}


@_clean
def test_gist_configured_but_no_usable_worker_is_never_config_missing():
    for name, opener in SITUATIONS.items():
        st = _status(opener)
        assert st.diagnostic_state == WorkerDiagnosticState.CONFIGURED_BUT_UNREACHABLE, name
        assert st.diagnostic_state != WorkerDiagnosticState.CONFIG_MISSING, name
        assert st.is_configured is True and not st.is_available and st.target_url is None, name
        assert "Registry (Gist 3bf1...9cf5)" in st.error and GIST not in st.error, name   # specific + masked


@_clean
def test_config_missing_is_reserved_for_no_registry_and_no_static_url():
    clear_status_cache()
    with patch.dict(os.environ, {}, clear=True), patch("core.gpu_config.find_env_file", return_value=None):
        st = check_worker_status(force_refresh=True)
    assert st.diagnostic_state == WorkerDiagnosticState.CONFIG_MISSING and st.is_configured is False
    assert "CROSSGUARD_GPU_REGISTRY_GIST_ID" in st.error


# ------------------------------------------------------------------ 10: explicit GPU never silently uses CPU
@_clean
def test_explicit_gpu_mode_fails_loudly_when_registry_has_no_worker():
    a, b, c = _with_gist(SITUATIONS["worker offline"])
    with a, b, c, patch("vision.models.is_cuda_available", return_value=False):
        with pytest.raises(RuntimeError) as exc_info:
            AdaptivePipeline(device=DeviceType.CUDA)
    assert "Remote CUDA worker is not currently reachable" in str(exc_info.value)
