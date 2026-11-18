"""Hardening tests for the GPU worker registry (review findings).

Covers: lookup caching (and failure caching), stale-if-error, clock-skew tolerance, rejection of untrusted
registry URLs, the removed unauthenticated /register endpoint, and a clean .env.example.
Never touches the network: GPURegistryClient._fetch_gist_content is mocked.
"""
from __future__ import annotations

import functools
import json
import os
import re
import time
from unittest.mock import patch

import pytest

from core.gpu_config import (WorkerUrlSource, check_worker_status, clear_status_cache,
                             get_configured_worker_url, resolve_worker_url)
from core.gpu_registry import GPURegistryClient, GPURegistryRecord, is_trusted_worker_url

def _clean(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        clear_status_cache()
        try:
            return fn(*a, **k)
        finally:
            clear_status_cache()
    return wrapper


GOOD = "https://reg-worker.trycloudflare.com"


def _env(**extra):
    env = {"CROSSGUARD_GPU_REGISTRY_GIST_ID": "hardening_gist_1"}
    env.update(extra)
    return env


def _content(url=GOOD, updated_at=None, status="online") -> str:
    return GPURegistryRecord(worker_url=url, status=status,
                             updated_at=time.time() if updated_at is None else updated_at).to_json()


def _fetching(ret):
    return patch.object(GPURegistryClient, "_fetch_gist_content", return_value=ret)


def _ctx(env):
    clear_status_cache()
    return patch.dict(os.environ, env, clear=True)


# ------------------------------------------------------------------ caching
@_clean
def test_repeated_resolutions_hit_github_once():
    with _ctx(_env()), patch("core.gpu_config.find_env_file", return_value=None), \
            _fetching((_content(), None)) as fetch:
        for _ in range(10):
            assert get_configured_worker_url() == GOOD
        assert fetch.call_count == 1


@_clean
def test_an_unreachable_registry_is_not_hammered():
    with _ctx(_env(CROSSGUARD_GPU_WORKER_URL="https://static.trycloudflare.com")), \
            patch("core.gpu_config.find_env_file", return_value=None), \
            _fetching((None, "net down")) as fetch:
        for _ in range(6):
            r = resolve_worker_url()
            assert r.source == WorkerUrlSource.ENV and r.registry_status == "unavailable"
        assert fetch.call_count == 1


@_clean
def test_clear_status_cache_forces_a_new_lookup():
    with _ctx(_env()), patch("core.gpu_config.find_env_file", return_value=None), \
            _fetching((_content(), None)) as fetch:
        get_configured_worker_url()
        clear_status_cache()
        get_configured_worker_url()
        assert fetch.call_count == 2


@_clean
def test_brief_github_outage_keeps_a_still_fresh_worker():
    with _ctx(_env()), patch("core.gpu_config.find_env_file", return_value=None):
        with _fetching((_content(), None)):
            assert resolve_worker_url(force_refresh_registry=True).source == WorkerUrlSource.REGISTRY
        with _fetching((None, "HTTP 502")):
            r = resolve_worker_url(force_refresh_registry=True)
            assert r.source == WorkerUrlSource.REGISTRY and r.url == GOOD


# ------------------------------------------------------------------ clock skew
@_clean
def test_small_clock_difference_between_colab_and_this_machine_is_not_stale():
    for skew in (0.2, 3.0, 45.0):
        with _ctx(_env()), patch("core.gpu_config.find_env_file", return_value=None), \
                _fetching((_content(updated_at=time.time() + skew), None)):
            r = resolve_worker_url(force_refresh_registry=True)
            assert (r.source, r.registry_status) == (WorkerUrlSource.REGISTRY, "fresh"), skew


@_clean
def test_far_future_or_old_records_are_still_rejected():
    for ts in (time.time() + 3600, time.time() - 3600):
        with _ctx(_env()), patch("core.gpu_config.find_env_file", return_value=None), \
                _fetching((_content(updated_at=ts), None)):
            r = resolve_worker_url(force_refresh_registry=True)
            assert r.source == WorkerUrlSource.NONE and r.registry_status == "stale"


# ------------------------------------------------------------------ untrusted registry URLs
BAD_URLS = [
    "https://attacker.example.com",
    "http://abc.trycloudflare.com",                       # remote workers must be https
    "https://user:pw@abc.trycloudflare.com",              # embedded credentials
    "https://abc.trycloudflare.com.evil.example",         # suffix confusion
    "https://trycloudflare.com.attacker.io",
]
GOOD_URLS = [
    "https://abc-def.trycloudflare.com",
    "https://my-gpu.ngrok-free.app",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]


@_clean
def test_untrusted_registry_urls_are_rejected_and_fall_back_to_env():
    for url in BAD_URLS:
        with _ctx(_env(CROSSGUARD_GPU_WORKER_URL="https://static.trycloudflare.com")), \
                patch("core.gpu_config.find_env_file", return_value=None), \
                _fetching((_content(url=url), None)):
            r = resolve_worker_url(force_refresh_registry=True)
            assert r.registry_status == "untrusted", url
            assert r.source == WorkerUrlSource.ENV and r.url == "https://static.trycloudflare.com", url


@_clean
def test_untrusted_registry_url_with_no_fallback_gives_a_clear_gpu_error():
    with _ctx(_env()), patch("core.gpu_config.find_env_file", return_value=None), \
            _fetching((_content(url="https://attacker.example.com"), None)):
        st = check_worker_status(force_refresh=True)
        assert not st.is_available and "rejected" in (st.error or "")


@_clean
def test_trusted_registry_urls_are_accepted():
    for url in GOOD_URLS:
        with _ctx(_env()), patch("core.gpu_config.find_env_file", return_value=None), \
                _fetching((_content(url=url), None)):
            r = resolve_worker_url(force_refresh_registry=True)
            assert (r.source, r.url) == (WorkerUrlSource.REGISTRY, url), url


@_clean
def test_allowed_hosts_can_be_extended_explicitly():
    url = "https://gpu.corp.example.com"
    assert not is_trusted_worker_url(url, {})[0]
    assert is_trusted_worker_url(url, {"CROSSGUARD_GPU_ALLOWED_HOSTS": "gpu.corp.example.com"})[0]
    assert is_trusted_worker_url(url, {"CROSSGUARD_GPU_ALLOWED_HOSTS": ".corp.example.com"})[0]
    assert not is_trusted_worker_url("", {})[0]


@_clean
def test_static_env_url_is_not_subject_to_the_registry_allow_list():
    """The allow-list guards registry-provided data only; an operator-set .env URL is trusted."""
    with _ctx({"CROSSGUARD_GPU_WORKER_URL": "https://gpu.corp.example.com"}), \
            patch("core.gpu_config.find_env_file", return_value=None):
        assert resolve_worker_url().source == WorkerUrlSource.ENV


# ------------------------------------------------------------------ worker + repo hygiene
@_clean
def test_worker_server_has_no_unauthenticated_register_endpoint():
    src = open("colab/worker_server.py", encoding="utf-8").read()
    assert "/register" not in src and "RegisterRequestModel" not in src
    assert "is_alive()" in src                       # heartbeat start is idempotent
    nb = open("colab/CrossGuard_GPU_Worker.ipynb", encoding="utf-8").read()
    assert "/register" not in nb


@_clean
def test_env_example_has_no_live_url_or_active_placeholder_settings():
    active = [ln.strip() for ln in open(".env.example", encoding="utf-8")
              if ln.strip() and not ln.lstrip().startswith("#")]
    assert active == [], active                      # everything is a commented example


@_clean
def test_no_secrets_in_repo_env_files():
    for name in (".env", ".env.example"):
        if os.path.exists(name):
            text = open(name, encoding="utf-8").read()
            assert not re.search(r"gh[pousr]_[A-Za-z0-9]{36}\b|github_pat_[A-Za-z0-9_]{50,}", text), name
