"""Client library for communicating with the CrossGuard Remote GPU Worker."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.request

import numpy as np

from vision.models import RawDetection
from .protocol import (
    InferenceRequest,
    InferenceResponse,
    WorkerHealth,
    encode_frame_to_b64,
)


class GPUWorkerError(Exception):
    """Base exception for all GPU worker client errors."""
    pass


class GPUWorkerConnectionError(GPUWorkerError):
    """Raised when the GPU worker endpoint cannot be reached."""
    pass


class GPUWorkerTimeoutError(GPUWorkerError):
    """Raised when a request to the GPU worker times out."""
    pass


class GPUWorkerProtocolError(GPUWorkerError):
    """Raised when the GPU worker returns an invalid, unexpected, or malformed response."""
    pass


class GPUWorkerUnavailableError(GPUWorkerError):
    """Raised when the GPU worker is connected but CUDA or YOLO is unavailable."""
    pass


class GPUWorkerClient:
    """HTTP Client for communicating with the CrossGuard Google Colab GPU Worker."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        raw_url = base_url or os.environ.get("CROSSGUARD_GPU_WORKER_URL", "http://localhost:8000")
        self.base_url = self._normalize_url(raw_url)
        self.timeout = max(0.5, float(timeout))

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = url.strip()
        if not url:
            return "http://localhost:8000"
        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"https://{url}"
        return url.rstrip("/")

    def get_health(self, timeout: Optional[float] = None) -> WorkerHealth:
        """Query worker /health endpoint to check online status and CUDA hardware availability."""
        req_timeout = timeout if timeout is not None else self.timeout
        url = f"{self.base_url}/health"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CrossGuard-Client/1.0", "Accept": "application/json"},
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                status_code = resp.getcode()
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise GPUWorkerProtocolError(f"Worker HTTP error {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            if "timed out" in str(e).lower() or isinstance(e.reason, TimeoutError):
                raise GPUWorkerTimeoutError(f"Health check timed out after {req_timeout:.1f}s") from e
            raise GPUWorkerConnectionError(f"Failed to connect to GPU worker at {self.base_url}: {e.reason}") from e
        except TimeoutError as e:
            raise GPUWorkerTimeoutError(f"Health check timed out after {req_timeout:.1f}s") from e
        except Exception as e:
            raise GPUWorkerConnectionError(f"Unexpected connection error to {self.base_url}: {e}") from e

        if status_code != 200:
            raise GPUWorkerProtocolError(f"Unexpected status code {status_code} from /health")

        try:
            data = json.loads(body)
            return WorkerHealth.from_dict(data)
        except Exception as e:
            raise GPUWorkerProtocolError(f"Failed to parse health response: {e}") from e

    def check_ready(self, timeout: Optional[float] = None) -> Tuple[bool, str, Optional[WorkerHealth]]:
        """Check if worker is online, connected, and ready for CUDA inference.

        Returns:
            (is_ready, description_message, health_obj_or_None)
        """
        try:
            health = self.get_health(timeout=timeout)
        except GPUWorkerTimeoutError:
            return False, f"Worker connection timed out at {self.base_url}", None
        except GPUWorkerConnectionError as e:
            return False, f"Cannot connect to worker ({e})", None
        except GPUWorkerProtocolError as e:
            return False, f"Worker returned invalid protocol response ({e})", None
        except Exception as e:
            return False, f"Worker unreachable ({e})", None

        if not health.cuda_available:
            return False, f"Worker online but CUDA is unavailable (GPU: {health.gpu_name})", health

        if not health.yolo_loaded:
            return False, f"Worker online with CUDA but YOLO model is not loaded", health

        return True, f"Online ({health.gpu_name})", health

    def detect(
        self,
        frame: np.ndarray,
        conf: float = 0.35,
        iou: float = 0.5,
        frame_index: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[List[RawDetection], float, str]:
        """Send a video frame to the remote GPU worker for YOLOv8 CUDA inference.

        Returns:
            (raw_detections, inference_time_ms, gpu_name)
        """
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return [], 0.0, "none"

        req_timeout = timeout if timeout is not None else self.timeout

        try:
            image_b64 = encode_frame_to_b64(frame)
        except Exception as e:
            raise GPUWorkerProtocolError(f"Frame encoding failed: {e}") from e

        inf_req = InferenceRequest(
            image_b64=image_b64,
            confidence_threshold=conf,
            iou_threshold=iou,
            frame_index=frame_index,
        )

        payload_bytes = json.dumps(inf_req.to_dict()).encode("utf-8")
        url = f"{self.base_url}/detect"

        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers={
                "User-Agent": "CrossGuard-Client/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                status_code = resp.getcode()
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise GPUWorkerProtocolError(f"Inference HTTP error {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            if "timed out" in str(e).lower() or isinstance(e.reason, TimeoutError):
                raise GPUWorkerTimeoutError(f"Inference request timed out after {req_timeout:.1f}s") from e
            raise GPUWorkerConnectionError(f"GPU worker connection lost during inference: {e.reason}") from e
        except TimeoutError as e:
            raise GPUWorkerTimeoutError(f"Inference request timed out after {req_timeout:.1f}s") from e
        except Exception as e:
            raise GPUWorkerConnectionError(f"GPU worker error during inference: {e}") from e

        if status_code != 200:
            raise GPUWorkerProtocolError(f"Inference endpoint returned HTTP {status_code}")

        try:
            data = json.loads(body)
            inf_resp = InferenceResponse.from_dict(data)
        except Exception as e:
            raise GPUWorkerProtocolError(f"Failed to parse inference response: {e}") from e

        if not inf_resp.is_success:
            err_msg = inf_resp.error or "Unknown error returned by remote worker"
            raise GPUWorkerProtocolError(f"Remote inference failed: {err_msg}")

        return inf_resp.to_raw_detections(), inf_resp.inference_time_ms, inf_resp.gpu_name
