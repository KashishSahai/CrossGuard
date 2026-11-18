"""End-to-end Computer Vision Detection + ByteTrack Tracking Pipeline."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Set

from .backend import InferenceBackend, YOLOBackend
from .models import DeviceType, RawDetection, TrackedDetection, VisionPipelineConfig
from .tracker import ByteTracker


class VisionPipeline:
    """Vision Pipeline coordinating YOLO detection and ByteTrack tracking.

    Receives video frames, runs detection on the configured hardware backend,
    filters traffic-relevant classes, and tracks objects persistently across frames.
    """

    def __init__(
        self,
        config: Optional[VisionPipelineConfig] = None,
        backend: Optional[InferenceBackend] = None,
        tracker: Optional[ByteTracker] = None,
    ) -> None:
        self.config = config or VisionPipelineConfig()

        if backend is not None:
            self.backend = backend
        else:
            self.backend = YOLOBackend(
                model_name_or_path=self.config.model_name_or_path,
                device=self.config.device,
                confidence_threshold=self.config.confidence_threshold,
                iou_threshold=self.config.iou_threshold,
            )

        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = ByteTracker(
                high_threshold=self.config.confidence_threshold,
                low_threshold=max(0.1, self.config.confidence_threshold * 0.4),
                match_iou_threshold=self.config.iou_threshold,
                max_lost_frames=self.config.max_lost_frames,
            )

        self.last_raw_detections: List[RawDetection] = []
        self.last_filtered_detections: List[RawDetection] = []
        self.last_tracked_objects: List[TrackedDetection] = []

    @property
    def device(self) -> str:
        """Effective hardware device ('cpu' or 'cuda')."""
        return self.backend.device

    @property
    def requested_device(self) -> str:
        """Requested hardware device specification ('auto', 'cpu', or 'cuda')."""
        return getattr(self.backend, "requested_device", str(self.config.device))

    @property
    def gpu_name(self) -> Optional[str]:
        """Hardware GPU model name if executing on GPU, else None."""
        return getattr(self.backend, "gpu_name", None)

    @property
    def last_inference_time_ms(self) -> float:
        """Inference latency in milliseconds for the most recent frame."""
        return getattr(self.backend, "last_inference_time_ms", 0.0)

    @property
    def is_loaded(self) -> bool:
        """Whether the detection backend is initialized and loaded."""
        return self.backend.is_loaded

    @property
    def status(self) -> str:
        """Observable detection status: 'ACTIVE', 'UNAVAILABLE', or 'MOCK'."""
        return getattr(self.backend, "status", "UNKNOWN")

    @property
    def load_error(self) -> Optional[str]:
        """Description of backend load failure, if any."""
        return getattr(self.backend, "load_error", None)

    def reset_tracking(self) -> None:
        """Reset internal tracking state for a new video session."""
        self.tracker.reset()
        self.last_raw_detections = []
        self.last_filtered_detections = []
        self.last_tracked_objects = []

    def filter_detections(self, detections: Sequence[RawDetection]) -> List[RawDetection]:
        """Filter detections to retain traffic-relevant classes."""
        if not self.config.filter_classes:
            return list(detections)

        target_classes = self.config.get_target_classes()
        filtered: List[RawDetection] = []
        for det in detections:
            if det.class_name.lower() in target_classes:
                filtered.append(det)
        return filtered

    def process_frame(self, frame: Any, frame_index: int = 0) -> List[TrackedDetection]:
        """Process a single video frame and return tracked objects.

        Lifecycle:
          Frame -> Detection -> Class Filtering -> ByteTrack Update -> Tracked Detections
        """
        if frame is None:
            # Empty / missing frame returns empty cleanly
            self.last_raw_detections = []
            self.last_filtered_detections = []
            self.last_tracked_objects = []
            return self.tracker.update([], frame_index)

        # 1. Detection
        raw_detections = self.backend.detect(frame)
        self.last_raw_detections = list(raw_detections)

        # 2. Filtering
        filtered_detections = self.filter_detections(raw_detections)
        self.last_filtered_detections = list(filtered_detections)

        # 3. Tracking
        tracked_objects = self.tracker.update(filtered_detections, frame_index)
        self.last_tracked_objects = list(tracked_objects)

        return tracked_objects
