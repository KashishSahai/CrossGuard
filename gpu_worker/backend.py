"""Remote GPU Detection Backend for CrossGuard Vision Pipeline.

Wraps the GPUWorkerClient into an InferenceBackend implementation so that
remote YOLOv8 CUDA inference on Google Colab integrates cleanly with ByteTrack,
SpatialAnalysis, and the AdaptiveController without altering downstream contracts.
"""
from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np

from vision.backend import InferenceBackend
from vision.models import DeviceType, RawDetection
from .client import (
    GPUWorkerClient,
    GPUWorkerConnectionError,
    GPUWorkerError,
    GPUWorkerTimeoutError,
    GPUWorkerUnavailableError,
)
from .protocol import WorkerHealth


class RemoteGPUBackend(InferenceBackend):
    """Detection backend executing YOLOv8n on a remote CUDA worker (e.g. Google Colab)."""

    def __init__(
        self,
        worker_url: Optional[str] = None,
        client: Optional[GPUWorkerClient] = None,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        timeout: float = 10.0,
        verify_on_init: bool = True,
    ) -> None:
        self.client = client or GPUWorkerClient(base_url=worker_url, timeout=timeout)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._gpu_name: str = "Unknown GPU"
        self._last_inference_time_ms: float = 0.0
        self._load_error: Optional[str] = None
        self._is_loaded: bool = False
        self._health: Optional[WorkerHealth] = None

        if verify_on_init:
            self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify worker is reachable, has CUDA available, and YOLO loaded."""
        try:
            health = self.client.get_health()
            self._health = health
            if not health.cuda_available:
                self._is_loaded = False
                self._load_error = f"Remote worker at {self.client.base_url} is online, but CUDA is unavailable (reported: {health.gpu_name})"
                raise GPUWorkerUnavailableError(self._load_error)

            if not health.yolo_loaded:
                self._is_loaded = False
                self._load_error = f"Remote worker at {self.client.base_url} is online with CUDA, but YOLOv8 model is not loaded"
                raise GPUWorkerUnavailableError(self._load_error)

            self._is_loaded = True
            self._gpu_name = health.gpu_name
            self._load_error = None
        except (GPUWorkerConnectionError, GPUWorkerTimeoutError) as e:
            self._is_loaded = False
            self._load_error = f"Cannot connect to remote GPU worker at {self.client.base_url}: {e}"
            raise GPUWorkerConnectionError(self._load_error) from e
        except GPUWorkerUnavailableError:
            raise
        except Exception as e:
            self._is_loaded = False
            self._load_error = f"Remote GPU worker verification failed: {e}"
            raise GPUWorkerConnectionError(self._load_error) from e

    @property
    def device(self) -> str:
        """Returns 'cuda' since inference executes on the remote CUDA GPU."""
        return "cuda"

    @property
    def requested_device(self) -> str:
        return "cuda"

    @property
    def gpu_name(self) -> str:
        """Name of the remote GPU (e.g. 'NVIDIA T4', 'Tesla T4')."""
        return self._gpu_name

    @property
    def last_inference_time_ms(self) -> float:
        """Measured YOLO inference latency on the remote GPU for the last frame."""
        return self._last_inference_time_ms

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def status(self) -> str:
        return "ACTIVE" if self._is_loaded else "UNAVAILABLE"

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def detect(self, frame: Any) -> List[RawDetection]:
        """Send frame to remote CUDA worker and return raw detections."""
        if not self._is_loaded:
            raise GPUWorkerUnavailableError(
                self._load_error or "Remote GPU worker is not connected or CUDA is unavailable."
            )

        if frame is None:
            return []

        try:
            dets, lat_ms, gpu = self.client.detect(
                frame=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
            )
            self._last_inference_time_ms = lat_ms
            if gpu and gpu.lower() not in ("none", "unknown gpu"):
                self._gpu_name = gpu
            return dets
        except (GPUWorkerConnectionError, GPUWorkerTimeoutError):
            # Re-raise connection error so UI can display worker connection loss without silent fallback
            raise
        except Exception as e:
            raise GPUWorkerError(f"Remote CUDA inference error: {e}") from e
