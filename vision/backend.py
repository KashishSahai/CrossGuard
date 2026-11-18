"""Detection backends for the vision pipeline.

Separates model inference from tracking and spatial logic.
Provides both real YOLOv8 backend and a deterministic MockBackend for unit testing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .models import DeviceType, RawDetection, resolve_device


class InferenceBackend(ABC):
    """Abstract interface for object detection backends."""

    @abstractmethod
    def detect(self, frame: Any) -> List[RawDetection]:
        """Run object detection on a single frame."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Effective hardware device being used ('cpu' or 'cuda')."""
        pass

    @property
    def requested_device(self) -> str:
        """Requested hardware device specification ('auto', 'cpu', or 'cuda')."""
        return "auto"

    @property
    def gpu_name(self) -> Optional[str]:
        """Hardware GPU model name if executing on GPU, else None."""
        return None

    @property
    def last_inference_time_ms(self) -> float:
        """Inference latency in milliseconds for the most recent frame."""
        return 0.0

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model weights are loaded and ready."""
        pass

    @property
    def status(self) -> str:
        """Observable status: 'ACTIVE', 'UNAVAILABLE', or 'MOCK'."""
        return "ACTIVE" if self.is_loaded else "UNAVAILABLE"

    @property
    def load_error(self) -> Optional[str]:
        """Description of load failure, if any."""
        return None


class YOLOBackend(InferenceBackend):
    """Production YOLOv8 detection backend using Ultralytics."""

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        device: Union[str, DeviceType] = DeviceType.AUTO,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self._requested_device = device.value if isinstance(device, DeviceType) else str(device)
        self._device = resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._model: Any = None
        self._loaded: bool = False
        self._load_error: Optional[str] = None
        self._last_inference_time_ms: float = 0.0

        self._load_model()

    def _resolve_model_path(self, path: str) -> str:
        import os
        env_weights = os.environ.get("CROSSGUARD_YOLO_WEIGHTS")
        if env_weights and os.path.exists(env_weights):
            return env_weights
        if path and os.path.exists(path):
            return path
        return path or "yolov8n.pt"

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            resolved_path = self._resolve_model_path(self.model_name_or_path)
            self._model = YOLO(resolved_path)
            self._loaded = True
            self._load_error = None
        except ImportError as e:
            self._model = None
            self._loaded = False
            self._load_error = f"ultralytics package not installed ({e})"
        except Exception as e:
            self._model = None
            self._loaded = False
            self._load_error = f"Failed to load YOLO model: {e}"

    @property
    def device(self) -> str:
        return self._device

    @property
    def requested_device(self) -> str:
        return self._requested_device

    @property
    def gpu_name(self) -> Optional[str]:
        if self._device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return str(torch.cuda.get_device_name(0))
            except Exception:
                pass
            return "NVIDIA CUDA GPU"
        return None

    @property
    def last_inference_time_ms(self) -> float:
        return self._last_inference_time_ms

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> str:
        """Observable status: 'ACTIVE' or 'UNAVAILABLE'."""
        return "ACTIVE" if (self._loaded and self._model is not None) else "UNAVAILABLE"

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def detect(self, frame: Any) -> List[RawDetection]:
        if not self._loaded or self._model is None:
            return []

        if frame is None:
            return []

        try:
            t0 = time.perf_counter()
            results = self._model.predict(
                source=frame,
                device=self._device,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            t1 = time.perf_counter()
            self._last_inference_time_ms = (t1 - t0) * 1000.0
        except Exception:
            return []

        if not results:
            return []

        detections: List[RawDetection] = []
        result = results[0]

        if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy()
        names = result.names if hasattr(result, "names") else {}

        for i in range(len(xyxy)):
            c_id = int(clses[i])
            c_name = names.get(c_id, str(c_id)).lower()
            detections.append(
                RawDetection(
                    x1=float(xyxy[i][0]),
                    y1=float(xyxy[i][1]),
                    x2=float(xyxy[i][2]),
                    y2=float(xyxy[i][3]),
                    confidence=float(confs[i]),
                    class_id=c_id,
                    class_name=c_name,
                )
            )

        return detections


class MockBackend(InferenceBackend):
    """Deterministic mock detection backend for robust unit testing without network/weights."""

    def __init__(
        self,
        device: Union[str, DeviceType] = DeviceType.CPU,
        detections_provider: Optional[Callable[[Any], List[RawDetection]]] = None,
    ) -> None:
        self._requested_device = device.value if isinstance(device, DeviceType) else str(device)
        self._device = resolve_device(device)
        self._provider = detections_provider
        self._fixed_detections: List[RawDetection] = []
        self._last_inference_time_ms: float = 10.0
        self._loaded: bool = True

    @property
    def device(self) -> str:
        return self._device

    @property
    def requested_device(self) -> str:
        return self._requested_device

    @property
    def gpu_name(self) -> Optional[str]:
        if self._device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return str(torch.cuda.get_device_name(0))
            except Exception:
                pass
            return "NVIDIA CUDA GPU"
        return None

    @property
    def last_inference_time_ms(self) -> float:
        return self._last_inference_time_ms

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> str:
        """Observable status for mock backend."""
        return "MOCK"

    @property
    def load_error(self) -> Optional[str]:
        return None

    def set_detections(self, detections: List[RawDetection]) -> None:
        """Set fixed detections to return on subsequent detect() calls."""
        self._fixed_detections = list(detections)

    def detect(self, frame: Any) -> List[RawDetection]:
        if self._provider is not None:
            return self._provider(frame)
        return list(self._fixed_detections)
