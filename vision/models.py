"""Data models and hardware device resolution for the vision layer."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple, Union


class DeviceType(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


def is_cuda_available() -> bool:
    """Check if CUDA is available via torch without crashing if torch is missing."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except (ImportError, Exception):
        return False


def resolve_device(requested: Union[str, DeviceType] = DeviceType.AUTO) -> str:
    """Resolve 'auto', 'cpu', or 'cuda' to an effective hardware device ('cpu' or 'cuda').

    Rules:
      - 'auto': returns 'cuda' if actually available, otherwise 'cpu'.
      - 'cpu': returns 'cpu'.
      - 'cuda': returns 'cuda' if actually available; raises RuntimeError if unavailable.
    """
    req_str = requested.value.lower() if isinstance(requested, DeviceType) else str(requested).strip().lower()

    if req_str == "auto":
        return "cuda" if is_cuda_available() else "cpu"
    elif req_str == "cpu":
        return "cpu"
    elif req_str in ("cuda", "gpu", "cuda:0"):
        if not is_cuda_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available. "
                "An NVIDIA GPU and a CUDA-enabled PyTorch installation are required."
            )
        return "cuda"
    else:
        # Unknown device, default safely to cpu
        return "cpu"


# Standard COCO traffic-relevant classes
DEFAULT_TRAFFIC_CLASSES: Set[str] = {
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
}


@dataclass
class VisionPipelineConfig:
    """Configuration for the vision detection and tracking pipeline."""
    model_name_or_path: str = "yolov8n.pt"
    device: Union[str, DeviceType] = DeviceType.AUTO
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    target_classes: Optional[Set[str]] = None
    filter_classes: bool = True
    tracker_type: str = "bytetrack"
    max_lost_frames: int = 30

    def get_effective_device(self) -> str:
        return resolve_device(self.device)

    def get_target_classes(self) -> Set[str]:
        if self.target_classes is not None:
            return {c.lower() for c in self.target_classes}
        return set(DEFAULT_TRAFFIC_CLASSES)


@dataclass(frozen=True)
class RawDetection:
    """Un-tracked detection output from an object detector."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class TrackedDetection:
    """Deterministic tracked object output contract.

    Guarantees required geometric and tracking attributes for downstream layers.
    """
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    frame_index: int

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return (self.center_x, self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "frame_index": self.frame_index,
        }
