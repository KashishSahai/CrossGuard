"""Communication protocol and data transfer models for CrossGuard Remote GPU Worker.

Defines request/response schemas, serialization/deserialization utilities,
and binary frame encoding for high-throughput, low-latency communication between
CrossGuard and the Colab CUDA worker.
"""
from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from vision.models import RawDetection


@dataclass
class WorkerHealth:
    """Health and hardware capabilities reported by the remote GPU worker."""
    status: str = "ok"
    cuda_available: bool = False
    gpu_name: str = "none"
    gpu_memory_total_mb: Optional[float] = None
    gpu_memory_allocated_mb: Optional[float] = None
    yolo_loaded: bool = False
    model_name: str = "yolov8n.pt"
    torch_version: str = ""
    device: str = "cuda"

    @property
    def is_ready(self) -> bool:
        """True if worker is online, CUDA is available, and YOLO model is loaded."""
        return (
            self.status.lower() in ("ok", "ready", "active")
            and bool(self.cuda_available)
            and bool(self.yolo_loaded)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerHealth:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for WorkerHealth, got {type(data).__name__}")
        return cls(
            status=str(data.get("status", "error")),
            cuda_available=bool(data.get("cuda_available", False)),
            gpu_name=str(data.get("gpu_name", "Unknown GPU")),
            gpu_memory_total_mb=data.get("gpu_memory_total_mb"),
            gpu_memory_allocated_mb=data.get("gpu_memory_allocated_mb"),
            yolo_loaded=bool(data.get("yolo_loaded", False)),
            model_name=str(data.get("model_name", "yolov8n.pt")),
            torch_version=str(data.get("torch_version", "")),
            device=str(data.get("device", "cuda")),
        )


@dataclass
class DetectionItem:
    """Individual object detection item returned by remote inference."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_raw_detection(self) -> RawDetection:
        return RawDetection(
            x1=float(self.x1),
            y1=float(self.y1),
            x2=float(self.x2),
            y2=float(self.y2),
            confidence=float(self.confidence),
            class_id=int(self.class_id),
            class_name=str(self.class_name).lower(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DetectionItem:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for DetectionItem, got {type(data).__name__}")
        return cls(
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            x2=float(data["x2"]),
            y2=float(data["y2"]),
            confidence=float(data["confidence"]),
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
        )


@dataclass
class InferenceRequest:
    """Payload sent from CrossGuard to the remote GPU worker for inference."""
    image_b64: str
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    frame_index: Optional[int] = None
    target_classes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InferenceRequest:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for InferenceRequest, got {type(data).__name__}")
        return cls(
            image_b64=str(data["image_b64"]),
            confidence_threshold=float(data.get("confidence_threshold", 0.35)),
            iou_threshold=float(data.get("iou_threshold", 0.5)),
            frame_index=data.get("frame_index"),
            target_classes=data.get("target_classes"),
        )


@dataclass
class InferenceResponse:
    """Inference response received from the remote GPU worker."""
    status: str = "ok"
    detections: List[DetectionItem] = field(default_factory=list)
    inference_time_ms: float = 0.0
    device: str = "cuda"
    gpu_name: str = "NVIDIA CUDA GPU"
    frame_index: Optional[int] = None
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status.lower() in ("ok", "success") and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["detections"] = [det.to_dict() for det in self.detections]
        return d

    def to_raw_detections(self) -> List[RawDetection]:
        return [det.to_raw_detection() for det in self.detections]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InferenceResponse:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for InferenceResponse, got {type(data).__name__}")

        raw_dets = data.get("detections", [])
        if not isinstance(raw_dets, list):
            raise ValueError(f"Expected list for detections, got {type(raw_dets).__name__}")

        parsed_dets = [DetectionItem.from_dict(d) for d in raw_dets]
        return cls(
            status=str(data.get("status", "ok")),
            detections=parsed_dets,
            inference_time_ms=float(data.get("inference_time_ms", 0.0)),
            device=str(data.get("device", "cuda")),
            gpu_name=str(data.get("gpu_name", "NVIDIA CUDA GPU")),
            frame_index=data.get("frame_index"),
            error=data.get("error"),
        )


def encode_frame_to_b64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode an OpenCV BGR frame image to base64 JPEG string."""
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        raise ValueError("Cannot encode empty or invalid frame")

    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        raise RuntimeError("Failed to JPEG-encode image frame")

    return base64.b64encode(buffer).decode("ascii")


def decode_b64_to_frame(b64_str: str) -> np.ndarray:
    """Decode a base64 JPEG string into an OpenCV BGR frame image."""
    if not b64_str:
        raise ValueError("Empty base64 string provided for decoding")

    img_bytes = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode base64 image data into cv2 image")
    return frame
