"""CrossGuard Remote GPU Worker Package.

Enables distributed YOLOv8n CUDA inference on Google Colab or remote GPU instances
while preserving local tracking, spatial analysis, and NEMA FSM adaptive control.
"""
from __future__ import annotations

from .protocol import (
    WorkerHealth,
    InferenceRequest,
    InferenceResponse,
    DetectionItem,
    encode_frame_to_b64,
    decode_b64_to_frame,
)
from .client import (
    GPUWorkerClient,
    GPUWorkerError,
    GPUWorkerConnectionError,
    GPUWorkerTimeoutError,
    GPUWorkerProtocolError,
    GPUWorkerUnavailableError,
)
from .backend import RemoteGPUBackend

__all__ = [
    "WorkerHealth",
    "InferenceRequest",
    "InferenceResponse",
    "DetectionItem",
    "encode_frame_to_b64",
    "decode_b64_to_frame",
    "GPUWorkerClient",
    "GPUWorkerError",
    "GPUWorkerConnectionError",
    "GPUWorkerTimeoutError",
    "GPUWorkerProtocolError",
    "GPUWorkerUnavailableError",
    "RemoteGPUBackend",
]
