"""Dynamic GPU Worker Registry Client & Publisher for CrossGuard.

Enables remote Google Colab GPU workers to publish their ephemeral Cloudflare
tunnel URLs to a fixed GitHub Gist, allowing CrossGuard to discover active
workers automatically without requiring manual edits to .env on every Colab restart.

Resolution precedence:
    1. Active, fresh registry entry (GitHub Gist)
    2. Static environment fallback (CROSSGUARD_GPU_WORKER_URL)
    3. None (worker unavailable)
"""
from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass
from enum import Enum
import json
import os
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple
import urllib.error
import urllib.request


DEFAULT_REGISTRY_FILENAME = "crossguard_worker.json"
DEFAULT_REGISTRY_TTL_SECONDS = 300.0  # 5 minutes
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60.0  # 1 minute
DEFAULT_REGISTRY_TIMEOUT = 5.0
PROTOCOL_VERSION = "1"

# Lookup caching: a lookup must never make the UI wait repeatedly. GitHub also allows only 60
# unauthenticated API requests per hour, so successes are cached longer than a worker heartbeat tick.
CACHE_TTL_OK_S = 30.0
CACHE_TTL_FAIL_S = 10.0
# The worker (Colab) and this machine have independent clocks; a few seconds of skew must not make a
# live worker look stale.
CLOCK_SKEW_TOLERANCE_S = 120.0

# The registry value is untrusted data. A worker URL is only used if it is https on one of these tunnel
# hosts (or localhost). Extend with CROSSGUARD_GPU_ALLOWED_HOSTS="my-gpu.example.com,.corp.example".
DEFAULT_ALLOWED_HOST_SUFFIXES: Tuple[str, ...] = (
    ".trycloudflare.com", ".ngrok-free.app", ".ngrok-free.dev", ".ngrok.app", ".ngrok.io",
)
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def is_trusted_worker_url(url: Optional[str], env: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """Return (trusted, reason). Applies only to URLs that came from the registry."""
    import urllib.parse
    if not url:
        return False, "empty URL"
    try:
        parsed = urllib.parse.urlparse(str(url).strip())
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False, "malformed URL"
    if not host:
        return False, "URL has no host"
    if parsed.username or parsed.password:
        return False, "URL must not contain credentials"
    if host in LOCAL_HOSTS:
        return (parsed.scheme in ("http", "https")), "localhost"
    if parsed.scheme != "https":
        return False, "remote workers must use https"
    environ = os.environ if env is None else env
    extra = []
    for part in environ.get("CROSSGUARD_GPU_ALLOWED_HOSTS", "").split(","):
        part = part.strip().lower()
        if part:
            extra.append(part if part.startswith(".") else part)
    for suffix in tuple(DEFAULT_ALLOWED_HOST_SUFFIXES) + tuple(extra):
        if (suffix.startswith(".") and host.endswith(suffix)) or host == suffix.lstrip("."):
            return True, "allowed host"
    return False, f"host '{host}' is not an allowed tunnel host (set CROSSGUARD_GPU_ALLOWED_HOSTS to allow it)"


def mask_gist_id(gist_id: Optional[str]) -> str:
    """Mask a GitHub Gist ID for safe logging and UI display."""
    if not gist_id:
        return "None (unset)"
    gid = str(gist_id).strip()
    if len(gid) > 8:
        return f"{gid[:4]}...{gid[-4:]}"
    return "***"


def parse_gist_id(raw_id_or_url: Optional[str]) -> Optional[str]:
    """Extract a bare GitHub Gist ID from a raw ID or full Gist URL."""
    if not raw_id_or_url:
        return None
    val = str(raw_id_or_url).strip()
    if not val:
        return None

    # Handle full web URL: https://gist.github.com/username/gist_id
    # or raw URL: https://gist.githubusercontent.com/username/gist_id/raw/...
    match = re.search(r"gist\.github(?:usercontent)?\.com/(?:[^/]+/)?([a-f0-9]+)", val, re.IGNORECASE)
    if match:
        return match.group(1)

    # Clean query parameters or trailing slashes from raw ID
    val = val.split("?")[0].split("#")[0].rstrip("/")
    # If it's a hex ID (typically 20 or 32 characters, but accept any alphanumeric ID >= 6 chars)
    if re.fullmatch(r"[a-fA-F0-9]{6,}", val):
        return val

    # Return stripped value if non-empty
    return val if val else None


def _parse_timestamp(raw_val: Any) -> float:
    """Parse a timestamp from float/int seconds or an ISO 8601 string."""
    if isinstance(raw_val, (int, float)):
        return float(raw_val)
    if isinstance(raw_val, str):
        val = raw_val.strip()
        try:
            return float(val)
        except ValueError:
            pass
        # Try parsing ISO 8601 format
        try:
            # Handle Z suffix
            clean_iso = val.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(clean_iso)
            return dt.timestamp()
        except Exception:
            pass
    return 0.0


@dataclass
class GPURegistryRecord:
    """Structured worker registration record stored in the GitHub Gist."""
    worker_url: str
    worker_id: str = "crossguard-colab"
    gpu_name: str = "Tesla T4"
    status: str = "online"  # "online" | "offline" | "degraded"
    updated_at: float = 0.0  # Unix timestamp in seconds
    protocol_version: str = PROTOCOL_VERSION
    heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS

    @property
    def is_online(self) -> bool:
        """True if the worker status is reported as online."""
        return str(self.status).strip().lower() == "online"

    def is_fresh(
        self,
        ttl_seconds: float = DEFAULT_REGISTRY_TTL_SECONDS,
        current_time: Optional[float] = None,
    ) -> bool:
        """Check if the record timestamp is within the freshness TTL."""
        if self.updated_at <= 0.0:
            return False
        now = current_time if current_time is not None else time.time()
        age = now - self.updated_at
        return -CLOCK_SKEW_TOLERANCE_S <= age <= max(1.0, float(ttl_seconds))

    def is_valid(
        self,
        ttl_seconds: float = DEFAULT_REGISTRY_TTL_SECONDS,
        current_time: Optional[float] = None,
    ) -> bool:
        """True if worker_url is non-empty, status is online, and record is fresh."""
        if not self.worker_url or not str(self.worker_url).strip():
            return False
        return self.is_online and self.is_fresh(ttl_seconds=ttl_seconds, current_time=current_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GPURegistryRecord:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for GPURegistryRecord, got {type(data).__name__}")
        raw_url = str(data.get("worker_url", "")).strip()
        raw_updated = data.get("updated_at", 0.0)
        return cls(
            worker_url=raw_url,
            worker_id=str(data.get("worker_id", "crossguard-colab")),
            gpu_name=str(data.get("gpu_name", "Unknown GPU")),
            status=str(data.get("status", "unknown")).lower(),
            updated_at=_parse_timestamp(raw_updated),
            protocol_version=str(data.get("protocol_version", PROTOCOL_VERSION)),
            heartbeat_interval_s=float(data.get("heartbeat_interval_s", DEFAULT_HEARTBEAT_INTERVAL_SECONDS)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> GPURegistryRecord:
        s = str(json_str).strip()
        if not s.startswith("{") and "{" in s:
            s = s[s.find("{") :]
        if not s.endswith("}") and "}" in s:
            s = s[: s.rfind("}") + 1]
        data = json.loads(s)
        return cls.from_dict(data)


class GPURegistryClient:
    """Client for reading from and publishing to the GitHub Gist worker registry."""

    def __init__(
        self,
        gist_id: Optional[str] = None,
        github_token: Optional[str] = None,
        filename: str = DEFAULT_REGISTRY_FILENAME,
        ttl_seconds: Optional[float] = None,
        timeout: float = DEFAULT_REGISTRY_TIMEOUT,
    ) -> None:
        raw_id = gist_id or os.environ.get("CROSSGUARD_GPU_REGISTRY_GIST_ID")
        self.gist_id = parse_gist_id(raw_id)
        self.github_token = (github_token or os.environ.get("CROSSGUARD_GPU_REGISTRY_GITHUB_TOKEN", "")).strip()
        self.filename = filename or DEFAULT_REGISTRY_FILENAME
        if ttl_seconds is not None:
            self.ttl_seconds = float(ttl_seconds)
        else:
            env_ttl = os.environ.get("CROSSGUARD_GPU_REGISTRY_TTL")
            self.ttl_seconds = float(env_ttl) if env_ttl else DEFAULT_REGISTRY_TTL_SECONDS
        self.timeout = max(0.5, float(timeout))

        # Lightweight in-memory cache to avoid duplicate network queries
        self._cached_record: Optional[GPURegistryRecord] = None
        self._cached_status: str = "not_configured"
        self._cached_error: Optional[str] = None
        self._cache_timestamp: float = 0.0
        self._cache_ttl: float = CACHE_TTL_OK_S
        self._cache_primed: bool = False                    # True once ANY outcome (incl. failure) is cached
        self._last_good: Optional[GPURegistryRecord] = None  # for stale-if-error

    def clear_cache(self) -> None:
        """Clear local registry lookup cache."""
        self._cached_record = None
        self._cached_status = "not_configured"
        self._cached_error = None
        self._cache_timestamp = 0.0
        self._cache_primed = False
        self._last_good = None

    def get_record(
        self,
        force_refresh: bool = False,
        current_time: Optional[float] = None,
    ) -> Tuple[Optional[GPURegistryRecord], str, Optional[str]]:
        """Fetch and validate the current worker record from the GitHub Gist.

        Returns:
            (record, status_code_string, error_message)
            status_code_string is one of:
                - "fresh"          : Valid record, online, within TTL
                - "stale"          : Valid record, but expired past TTL
                - "offline"        : Valid record, but explicitly marked offline
                - "invalid"        : Malformed JSON or missing required fields
                - "unavailable"    : HTTP error, network failure, or Gist not found
                - "not_configured" : No GIST_ID provided
        """
        now = current_time if current_time is not None else time.time()
        if not force_refresh and self._cache_primed:
            window = self._cache_ttl if self._cached_record is not None else CACHE_TTL_FAIL_S
            if 0.0 <= (now - self._cache_timestamp) < window:
                if self._cached_record is None:          # cached failure: do not hammer an unreachable registry
                    return None, self._cached_status, self._cached_error
                # Re-evaluate freshness on the cached record
                if not self._cached_record.is_online:
                    return self._cached_record, "offline", None
                if not self._cached_record.is_fresh(self.ttl_seconds, current_time=now):
                    return self._cached_record, "stale", f"Registry record expired (> {self.ttl_seconds:.0f}s old)"
                return self._cached_record, "fresh", None

        if not self.gist_id:
            return None, "not_configured", "CROSSGUARD_GPU_REGISTRY_GIST_ID is not configured."

        content, err = self._fetch_gist_content()
        if err or not content:
            self._cache_timestamp = now
            self._cache_primed = True
            # Stale-if-error: a brief GitHub outage must not drop a worker that is still heartbeating.
            if self._last_good is not None and self._last_good.is_online and \
                    self._last_good.is_fresh(self.ttl_seconds, current_time=now):
                self._cached_record, self._cached_status, self._cached_error = self._last_good, "fresh", None
                return self._last_good, "fresh", None
            self._cached_record = None
            self._cached_status = "unavailable"
            self._cached_error = err
            return None, "unavailable", err

        try:
            record = GPURegistryRecord.from_json(content)
        except Exception as e:
            err_msg = f"Failed to parse registry record JSON: {e}"
            self._cached_record = None
            self._cached_status = "invalid"
            self._cached_error = err_msg
            self._cache_timestamp = now
            self._cache_primed = True
            return None, "invalid", err_msg

        if not record.worker_url:
            err_msg = "Registry record has empty worker_url."
            self._cached_record = None
            self._cached_status = "invalid"
            self._cached_error = err_msg
            self._cache_timestamp = now
            self._cache_primed = True
            return None, "invalid", err_msg

        self._cached_record = record
        self._last_good = record
        self._cache_timestamp = now
        self._cache_primed = True

        if not record.is_online:
            self._cached_status = "offline"
            self._cached_error = f"Worker reported status: '{record.status}'"
            return record, "offline", None

        if not record.is_fresh(self.ttl_seconds, current_time=now):
            age = now - record.updated_at
            err_msg = f"Registry record is stale (age={age:.1f}s, TTL={self.ttl_seconds:.0f}s)."
            self._cached_status = "stale"
            self._cached_error = err_msg
            return record, "stale", err_msg

        self._cached_status = "fresh"
        self._cached_error = None
        return record, "fresh", None

    def _fetch_gist_content(self) -> Tuple[Optional[str], Optional[str]]:
        """Fetch raw content of the registry file from GitHub API or raw URL."""
        # 1. Primary: GitHub Gists REST API
        api_url = f"https://api.github.com/gists/{self.gist_id}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "CrossGuard-Registry/1.0",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"

        req = urllib.request.Request(api_url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    return None, f"GitHub Gist API returned status {resp.status}"
                data = json.loads(resp.read().decode("utf-8"))
                files = data.get("files", {})
                if self.filename in files:
                    file_info = files[self.filename]
                    content = file_info.get("content")
                    if content:
                        return content, None
                    # If content is truncated, try raw_url
                    raw_url = file_info.get("raw_url")
                    if raw_url:
                        return self._fetch_raw_url(raw_url)
                # Fallback: if only one file in Gist and ends in .json
                for fname, finfo in files.items():
                    if fname.endswith(".json") and finfo.get("content"):
                        return finfo["content"], None
                # Fallback: check any file containing valid worker_url JSON
                for fname, finfo in files.items():
                    c = finfo.get("content", "")
                    if "{" in c and ("worker_url" in c or "status" in c):
                        return c, None
                return None, f"File '{self.filename}' not found in Gist {mask_gist_id(self.gist_id)}"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, f"GitHub Gist {mask_gist_id(self.gist_id)} not found (HTTP 404)."
            if e.code in (403, 429):
                # Try raw usercontent fallback (not subject to REST API 60 req/hr unauthenticated limit)
                raw_url = f"https://gist.githubusercontent.com/raw/{self.gist_id}/{self.filename}"
                raw_c, raw_err = self._fetch_raw_url(raw_url)
                if raw_c and not raw_err:
                    return raw_c, None
                raw_alt = f"https://gist.githubusercontent.com/raw/{self.gist_id}/gistfile1.txt"
                raw_c2, raw_err2 = self._fetch_raw_url(raw_alt)
                if raw_c2 and not raw_err2:
                    return raw_c2, None
                return None, ("GitHub API rate limit reached (60 requests/hour without a token). "
                              "Set CROSSGUARD_GPU_REGISTRY_GITHUB_TOKEN to raise the limit.")
            return None, f"GitHub Gist HTTP error {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            return None, f"Failed to connect to GitHub Gist: {e.reason}"
        except TimeoutError:
            return None, f"Registry lookup timed out after {self.timeout:.1f}s"
        except Exception as e:
            return None, f"Registry read error: {e}"

    def _fetch_raw_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch raw content directly from GitHub Usercontent URL."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CrossGuard-Registry/1.0"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8"), None
                return None, f"Raw Gist fetch returned status {resp.status}"
        except Exception as e:
            return None, f"Raw Gist fetch failed: {e}"

    def publish_record(
        self,
        worker_url: str,
        gpu_name: str = "Tesla T4",
        status: str = "online",
        worker_id: str = "crossguard-colab",
    ) -> Tuple[bool, str]:
        """Publish or update the worker registration record in the GitHub Gist.

        Requires self.github_token and self.gist_id.
        """
        if not self.gist_id:
            return False, "Cannot publish: Gist ID is not configured."
        if not self.github_token:
            return False, "Cannot publish: CROSSGUARD_GPU_REGISTRY_GITHUB_TOKEN is required to update Gist."

        record = GPURegistryRecord(
            worker_url=worker_url.strip(),
            worker_id=worker_id,
            gpu_name=gpu_name,
            status=status,
            updated_at=time.time(),
            protocol_version=PROTOCOL_VERSION,
            heartbeat_interval_s=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        )

        patch_payload = {
            "description": f"CrossGuard Remote GPU Worker Registry [{status.upper()}]",
            "files": {
                self.filename: {
                    "content": record.to_json(),
                }
            },
        }

        api_url = f"https://api.github.com/gists/{self.gist_id}"
        req_data = json.dumps(patch_payload).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=req_data,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.github_token}",
                "Content-Type": "application/json",
                "User-Agent": "CrossGuard-Colab-Publisher/1.0",
            },
            method="PATCH",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status in (200, 201):
                    self._cached_record = record
                    self._cached_status = "fresh" if status == "online" else "offline"
                    self._cache_timestamp = time.time()
                    self._cache_primed = True
                    return True, "Registry updated successfully"
                return False, f"GitHub Gist update returned HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            return False, f"GitHub Gist update HTTP error {e.code}: {e.reason}"
        except Exception as e:
            return False, f"Failed to update registry: {e}"

    def mark_offline(self, worker_url: str = "", gpu_name: str = "Tesla T4") -> bool:
        """Mark worker status as 'offline' during graceful shutdown."""
        success, _ = self.publish_record(worker_url=worker_url, gpu_name=gpu_name, status="offline")
        return success

    def start_heartbeat(
        self,
        worker_url: str,
        gpu_name: str = "Tesla T4",
        interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ) -> Tuple[threading.Thread, threading.Event]:
        """Start a background daemon thread that periodically refreshes the registration.

        Returns:
            (thread, stop_event)
        """
        stop_event = threading.Event()

        def _heartbeat_worker():
            while not stop_event.wait(interval_s):
                try:
                    self.publish_record(worker_url=worker_url, gpu_name=gpu_name, status="online")
                except Exception:
                    pass

        t = threading.Thread(target=_heartbeat_worker, name="CrossGuard-RegistryHeartbeat", daemon=True)
        t.start()
        return t, stop_event
