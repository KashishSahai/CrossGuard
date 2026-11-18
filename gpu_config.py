"""Automatic GPU Worker Discovery & Configuration Layer for CrossGuard.

Manages backend hardware discovery, environment loading (.env / CROSSGUARD_GPU_WORKER_URL),
non-blocking cached health verification, and status reporting.

Treats remote CUDA workers (e.g. Google Colab or cloud GPU instances) as developer/deployment
infrastructure, hiding network endpoints, tunnels, and infrastructure details from the end user.
"""
from __future__ import annotations

from enum import Enum
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

from core.gpu_registry import (
    GPURegistryClient,
    GPURegistryRecord,
    is_trusted_worker_url,
    mask_gist_id,
    parse_gist_id,
)
from gpu_worker.client import (
    GPUWorkerClient,
    GPUWorkerConnectionError,
    GPUWorkerTimeoutError,
)
from gpu_worker.protocol import WorkerHealth


class WorkerUrlSource(str, Enum):
    REGISTRY = "registry"
    ENV = "env"
    EXPLICIT = "explicit"
    NONE = "none"


@dataclass
class WorkerResolution:
    url: Optional[str]
    source: WorkerUrlSource
    registry_record: Optional[GPURegistryRecord] = None
    registry_status: str = "not_configured"
    error: Optional[str] = None


class WorkerDiagnosticState(str, Enum):
    CONFIG_MISSING = "CONFIG MISSING"
    CONFIGURED_BUT_UNREACHABLE = "CONFIGURED BUT UNREACHABLE"
    WORKER_REACHABLE_BUT_CUDA_UNAVAILABLE = "WORKER REACHABLE BUT CUDA UNAVAILABLE"
    WORKER_REACHABLE_AND_CUDA_AVAILABLE = "WORKER REACHABLE + CUDA AVAILABLE"


def mask_worker_url(url: Optional[str]) -> str:
    """Mask sensitive or session-specific tunnel URLs for display."""
    if not url:
        return "None (unset)"
    u = str(url).strip()
    if "://" in u:
        scheme, rest = u.split("://", 1)
        if len(rest) > 14:
            return f"{scheme}://{rest[:4]}...{rest[-10:]}"
        return f"{scheme}://***"
    return "***"


@dataclass
class GPUWorkerStatus:
    """Cached operational status of the configured GPU worker."""
    is_configured: bool
    is_connected: bool
    cuda_available: bool
    gpu_name: Optional[str] = None
    yolo_loaded: bool = False
    model_name: Optional[str] = None
    error: Optional[str] = None
    last_checked_timestamp: float = 0.0
    diagnostic_state: WorkerDiagnosticState = WorkerDiagnosticState.CONFIG_MISSING
    target_url: Optional[str] = None
    masked_url: Optional[str] = None
    url_source: WorkerUrlSource = WorkerUrlSource.NONE
    registry_status: str = "not_configured"
    registry_updated_at: Optional[float] = None
    registry_record: Optional[GPURegistryRecord] = None
    registry_gist_id: Optional[str] = None
    env_found: bool = False
    env_path: Optional[str] = None
    url_configured: bool = False
    url_normalized: bool = False

    @property
    def is_available(self) -> bool:
        """True if worker is online, reachable, has CUDA available, and YOLO loaded."""
        return self.is_configured and self.is_connected and self.cuda_available and self.yolo_loaded

    @property
    def badge_text(self) -> str:
        """Concise summary for UI status badges."""
        if self.is_available:
            name = self.gpu_name or "NVIDIA GPU"
            src_str = f" ({self.url_source.value})" if self.url_source != WorkerUrlSource.NONE else ""
            return f"Connected • {name} • CUDA{src_str}"
        return "Unavailable • CPU available"

    @property
    def status_label(self) -> str:
        return "Connected" if self.is_available else "Unavailable"


# Internal in-process status cache
_STATUS_CACHE: Optional[GPUWorkerStatus] = None
_CACHE_TTL_SECONDS: float = 6.0


def get_project_root() -> Path:
    """Return the repository root directory containing app.py."""
    return Path(__file__).resolve().parent.parent


def find_env_file() -> Optional[Path]:
    """Find the .env file in the project root or working directory."""
    root = get_project_root()
    candidates = [
        root / ".env",
        Path.cwd() / ".env",
        root.parent / ".env",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_env_file(filepath: Optional[str] = None) -> bool:
    """Load key-value pairs from a .env file into os.environ without overwriting existing vars."""
    target_path = Path(filepath) if filepath else find_env_file()
    if target_path is None or not target_path.is_file():
        return False

    if _HAS_DOTENV:
        try:
            load_dotenv(dotenv_path=target_path, override=False)
        except Exception:
            pass

    # Built-in robust fallback parser (handles quotes, spaces, comments)
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
        return True
    except Exception:
        return False


def get_env_diagnostic_summary() -> Dict[str, Any]:
    """Return explicit diagnostic flags for configuration visibility."""
    env_file = find_env_file()
    env_found = env_file is not None and env_file.is_file()
    load_env_file(str(env_file) if env_file else None)

    raw_val = os.environ.get("CROSSGUARD_GPU_WORKER_URL", "").strip()
    url_configured = bool(raw_val)
    normalized = normalize_worker_url(raw_val) if url_configured else ""
    url_normalized = bool(normalized and (normalized.startswith("http://") or normalized.startswith("https://")))
    raw_gist = os.environ.get("CROSSGUARD_GPU_REGISTRY_GIST_ID", "").strip()
    gist_id = parse_gist_id(raw_gist)

    return {
        "env_found": env_found,
        "env_path": str(env_file) if env_file else None,
        "url_configured": url_configured,
        "url_normalized": url_normalized,
        "masked_url": mask_worker_url(normalized),
        "registry_configured": bool(gist_id),
        "registry_gist_id": mask_gist_id(gist_id),
    }


def normalize_worker_url(url: Optional[str]) -> str:
    """Clean and normalize a worker endpoint URL."""
    if not url:
        return ""
    u = str(url).strip()
    if not u:
        return ""
    if not (u.startswith("http://") or u.startswith("https://")):
        if "localhost" in u or "127.0.0.1" in u:
            u = f"http://{u}"
        else:
            u = f"https://{u}"
    return u.rstrip("/")


_REGISTRY_CLIENT: Optional[GPURegistryClient] = None
_REGISTRY_CLIENT_KEY: Optional[tuple] = None
REGISTRY_READ_TIMEOUT_S = 3.0


def _get_registry_client(gist_id: str) -> GPURegistryClient:
    """One shared client per configuration, so its lookup cache survives between calls.

    (A new client per call has an empty cache and would query GitHub every time, blocking the UI and
    exhausting GitHub's 60 requests/hour unauthenticated limit.)
    """
    global _REGISTRY_CLIENT, _REGISTRY_CLIENT_KEY
    key = (gist_id,
           os.environ.get("CROSSGUARD_GPU_REGISTRY_GITHUB_TOKEN", ""),
           os.environ.get("CROSSGUARD_GPU_REGISTRY_TTL", ""))
    if _REGISTRY_CLIENT is None or key != _REGISTRY_CLIENT_KEY:
        _REGISTRY_CLIENT = GPURegistryClient(gist_id=gist_id, timeout=REGISTRY_READ_TIMEOUT_S)
        _REGISTRY_CLIENT_KEY = key
    return _REGISTRY_CLIENT


def resolve_worker_url(
    explicit_url: Optional[str] = None,
    force_refresh_registry: bool = False,
) -> WorkerResolution:
    """Authoritative single choke-point for resolving remote GPU worker URL and source.

    Resolution Order:
        1. Explicit override if provided (source: explicit)
        2. Dynamic Registry (GitHub Gist) if configured, online, and fresh (source: registry)
        3. Static Environment Fallback: CROSSGUARD_GPU_WORKER_URL (source: env)
        4. None: No worker available (source: none)
    """
    load_env_file()

    # 1. Explicit override
    if explicit_url is not None and str(explicit_url).strip():
        norm = normalize_worker_url(explicit_url)
        return WorkerResolution(
            url=norm if norm else None,
            source=WorkerUrlSource.EXPLICIT if norm else WorkerUrlSource.NONE,
        )

    # 2. Dynamic Registry (GitHub Gist)
    raw_gist = os.environ.get("CROSSGUARD_GPU_REGISTRY_GIST_ID", "").strip()
    gist_id = parse_gist_id(raw_gist)
    reg_record = None
    reg_status = "not_configured"
    reg_err = None

    if gist_id:
        reg_client = _get_registry_client(gist_id)
        reg_record, reg_status, reg_err = reg_client.get_record(force_refresh=force_refresh_registry)
        if reg_status == "fresh" and reg_record and reg_record.worker_url:
            norm_reg = normalize_worker_url(reg_record.worker_url)
            trusted, why = is_trusted_worker_url(norm_reg)
            if not trusted:
                # The registry is untrusted input: never route video frames to an unexpected host.
                reg_status, reg_err = "untrusted", f"Registry worker URL rejected: {why}."
            elif norm_reg:
                return WorkerResolution(
                    url=norm_reg,
                    source=WorkerUrlSource.REGISTRY,
                    registry_record=reg_record,
                    registry_status=reg_status,
                )

    # 3. Static Environment Fallback (.env or system env)
    raw_env_url = os.environ.get("CROSSGUARD_GPU_WORKER_URL", "").strip()
    if raw_env_url:
        norm_env = normalize_worker_url(raw_env_url)
        if norm_env:
            return WorkerResolution(
                url=norm_env,
                source=WorkerUrlSource.ENV,
                registry_record=reg_record,
                registry_status=reg_status,
                error=reg_err if gist_id else None,
            )

    # 4. None
    return WorkerResolution(
        url=None,
        source=WorkerUrlSource.NONE,
        registry_record=reg_record,
        registry_status=reg_status,
        error=reg_err,
    )


def get_configured_worker_url(force_refresh_registry: bool = False) -> Optional[str]:
    """Retrieve the authoritative configured worker URL (Registry -> .env -> None)."""
    return resolve_worker_url(force_refresh_registry=force_refresh_registry).url


def check_worker_status(
    worker_url: Optional[str] = None,
    timeout: float = 1.0,
    force_refresh: bool = False,
) -> GPUWorkerStatus:
    """Perform a lightweight, cached health check against the configured worker.

    Guaranteed not to block application startup or UI interactions if the worker is offline.
    """
    global _STATUS_CACHE
    now = time.time()

    if not force_refresh and _STATUS_CACHE is not None:
        if (now - _STATUS_CACHE.last_checked_timestamp) < _CACHE_TTL_SECONDS:
            return _STATUS_CACHE

    env_diag = get_env_diagnostic_summary()
    resolution = resolve_worker_url(
        explicit_url=worker_url,
        force_refresh_registry=force_refresh,
    )
    target_url = resolution.url
    raw_gist = os.environ.get("CROSSGUARD_GPU_REGISTRY_GIST_ID", "").strip()
    gist_id = parse_gist_id(raw_gist)

    if not target_url:
        if resolution.registry_status == "stale":
            err_msg = "Remote GPU worker registry record is stale and no static fallback URL is configured."
        elif resolution.registry_status == "offline":
            err_msg = "Remote GPU worker is marked offline in registry and no static fallback URL is configured."
        elif resolution.registry_status == "untrusted":
            err_msg = f"{resolution.error} No static fallback URL is configured."
        elif resolution.registry_status == "invalid":
            err_msg = f"Remote GPU worker registry record is invalid ({resolution.error}) and no static fallback configured."
        else:
            err_msg = "No remote GPU worker URL configured (Registry not active and CROSSGUARD_GPU_WORKER_URL is unset)."

        _STATUS_CACHE = GPUWorkerStatus(
            is_configured=False,
            is_connected=False,
            cuda_available=False,
            error=err_msg,
            last_checked_timestamp=now,
            diagnostic_state=WorkerDiagnosticState.CONFIG_MISSING,
            target_url=None,
            masked_url="None (unset)",
            url_source=resolution.source,
            registry_status=resolution.registry_status,
            registry_updated_at=resolution.registry_record.updated_at if resolution.registry_record else None,
            registry_record=resolution.registry_record,
            registry_gist_id=gist_id,
            env_found=env_diag["env_found"],
            env_path=env_diag["env_path"],
            url_configured=env_diag["url_configured"],
            url_normalized=env_diag["url_normalized"],
        )
        return _STATUS_CACHE

    masked = mask_worker_url(target_url)
    client = GPUWorkerClient(base_url=target_url, timeout=timeout)
    is_ready, msg, health = client.check_ready()

    base_status_kwargs = dict(
        target_url=target_url,
        masked_url=masked,
        url_source=resolution.source,
        registry_status=resolution.registry_status,
        registry_updated_at=resolution.registry_record.updated_at if resolution.registry_record else None,
        registry_record=resolution.registry_record,
        registry_gist_id=gist_id,
        env_found=env_diag["env_found"],
        env_path=env_diag["env_path"],
        url_configured=env_diag["url_configured"],
        url_normalized=env_diag["url_normalized"],
    )

    if is_ready and health and health.cuda_available:
        _STATUS_CACHE = GPUWorkerStatus(
            is_configured=True,
            is_connected=True,
            cuda_available=True,
            gpu_name=health.gpu_name,
            yolo_loaded=bool(health.yolo_loaded),
            model_name=health.model_name,
            error=None,
            last_checked_timestamp=now,
            diagnostic_state=WorkerDiagnosticState.WORKER_REACHABLE_AND_CUDA_AVAILABLE,
            **base_status_kwargs,
        )
    elif health and not health.cuda_available:
        _STATUS_CACHE = GPUWorkerStatus(
            is_configured=True,
            is_connected=True,
            cuda_available=False,
            gpu_name=health.gpu_name,
            yolo_loaded=bool(health.yolo_loaded),
            model_name=health.model_name,
            error=f"Worker reachable at {masked}, but CUDA is unavailable (worker running on CPU).",
            last_checked_timestamp=now,
            diagnostic_state=WorkerDiagnosticState.WORKER_REACHABLE_BUT_CUDA_UNAVAILABLE,
            **base_status_kwargs,
        )
    elif health:
        _STATUS_CACHE = GPUWorkerStatus(
            is_configured=True,
            is_connected=True,
            cuda_available=bool(health.cuda_available),
            gpu_name=health.gpu_name,
            yolo_loaded=bool(health.yolo_loaded),
            model_name=health.model_name,
            error=f"Worker reachable at {masked}, but model not ready: {msg}",
            last_checked_timestamp=now,
            diagnostic_state=(
                WorkerDiagnosticState.WORKER_REACHABLE_BUT_CUDA_UNAVAILABLE
                if not health.cuda_available
                else WorkerDiagnosticState.CONFIGURED_BUT_UNREACHABLE
            ),
            **base_status_kwargs,
        )
    else:
        _STATUS_CACHE = GPUWorkerStatus(
            is_configured=True,
            is_connected=False,
            cuda_available=False,
            error=f"Remote CUDA worker at {masked} is not reachable: {msg}",
            last_checked_timestamp=now,
            diagnostic_state=WorkerDiagnosticState.CONFIGURED_BUT_UNREACHABLE,
            **base_status_kwargs,
        )

    return _STATUS_CACHE


def is_gpu_worker_available(force_refresh: bool = False) -> bool:
    """Return True if the remote GPU worker is configured, connected, and ready for CUDA."""
    return check_worker_status(force_refresh=force_refresh).is_available


def get_gpu_worker_client(timeout: float = 10.0) -> Optional[GPUWorkerClient]:
    """Return a GPUWorkerClient configured with the active worker URL, if available."""
    url = get_configured_worker_url()
    return GPUWorkerClient(base_url=url, timeout=timeout) if url else None


def clear_status_cache() -> None:
    """Clear internal health check cache (and the registry lookup cache)."""
    global _STATUS_CACHE, _REGISTRY_CLIENT, _REGISTRY_CLIENT_KEY
    _STATUS_CACHE = None
    _REGISTRY_CLIENT = None
    _REGISTRY_CLIENT_KEY = None
