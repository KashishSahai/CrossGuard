"""Perception Adapter Layer for CrossGuard.

Bridges the Vision Detection/Tracking layer with the Spatial Analysis layer.
Maps tracked vision objects (TrackedDetection) into spatially classified
detections (SpatialDetection) using ground-contact / bottom-center coordinates.
Pure Python; no UI dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.spatial_analysis import Approach, SpatialAnalyzer, SpatialZone, ZoneType
from core.camera_calibration import CameraCalibration
from vision.models import TrackedDetection


@dataclass(frozen=True)
class CoordinateTransformer:
    """Reusable linear coordinate scaling layer from camera resolution to canonical spatial resolution (Step 13).

    Maps:
      x' = x * target_width / source_width
      y' = y * target_height / source_height
    """
    source_width: float = 1000.0
    source_height: float = 800.0
    target_width: float = 1000.0
    target_height: float = 800.0

    @property
    def is_identity(self) -> bool:
        """Whether source and target dimensions are identical (identity transform)."""
        return (
            abs(self.source_width - self.target_width) < 1e-6
            and abs(self.source_height - self.target_height) < 1e-6
        )

    @property
    def scale_x(self) -> float:
        if self.source_width <= 0:
            return 1.0
        return self.target_width / self.source_width

    @property
    def scale_y(self) -> float:
        if self.source_height <= 0:
            return 1.0
        return self.target_height / self.source_height

    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a 2D point (x, y) from source space into target canonical space."""
        if self.is_identity:
            return point
        return (point[0] * self.scale_x, point[1] * self.scale_y)

    def transform_bbox(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Transform a bounding box (x1, y1, x2, y2) from source space into target space."""
        if self.is_identity:
            return bbox
        x1, y1, x2, y2 = bbox
        sx, sy = self.scale_x, self.scale_y
        return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


@dataclass(frozen=True)
class SpatialDetection:
    """A tracked perception detection enriched with spatial intersection context."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    bottom_center: Tuple[float, float]
    spatial_point: Tuple[float, float]
    zone: SpatialZone
    approach: Optional[Approach]
    zone_type: Optional[ZoneType]
    frame_index: int
    camera_bbox: Optional[Tuple[float, float, float, float]] = None
    camera_bottom_center: Optional[Tuple[float, float]] = None

    @property
    def is_in_zone(self) -> bool:
        """Whether the detection falls into any meaningful traffic zone."""
        return self.zone != SpatialZone.NONE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize spatial detection to a dictionary."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(c, 2) for c in self.bbox],
            "center": (round(self.center[0], 2), round(self.center[1], 2)),
            "bottom_center": (round(self.bottom_center[0], 2), round(self.bottom_center[1], 2)),
            "spatial_point": (round(self.spatial_point[0], 2), round(self.spatial_point[1], 2)),
            "zone": self.zone.value,
            "approach": self.approach.value if self.approach else None,
            "zone_type": self.zone_type.value if self.zone_type else None,
            "frame_index": self.frame_index,
        }


class PerceptionAdapter:
    """Adapts tracked detections into spatially classified detections.

    Applies coordinate normalization (if configured) so camera coordinates
    are mapped onto the canonical 1000x800 intersection geometry.
    Uses the detection's bottom-center coordinate (representing physical ground/road contact)
    and passes it through the SpatialAnalyzer without altering spatial geometry rules.
    """

    def __init__(
        self,
        spatial_analyzer: Optional[SpatialAnalyzer] = None,
        use_bottom_center: bool = True,
        transformer: Optional[CoordinateTransformer] = None,
        source_width: float = 1000.0,
        source_height: float = 800.0,
        target_width: float = 1000.0,
        target_height: float = 800.0,
        calibration: Optional[CameraCalibration] = None,
    ) -> None:
        self.spatial_analyzer = spatial_analyzer or SpatialAnalyzer()
        self.use_bottom_center = use_bottom_center
        self.calibration = calibration
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = CoordinateTransformer(
                source_width=source_width,
                source_height=source_height,
                target_width=target_width,
                target_height=target_height,
            )

    def set_calibration(self, calibration: Optional[CameraCalibration]) -> None:
        """Configure or clear camera homography calibration."""
        self.calibration = calibration

    def set_source_resolution(self, width: float, height: float) -> None:
        """Update source resolution dynamically (e.g. from camera/video metadata)."""
        self.transformer = CoordinateTransformer(
            source_width=float(width),
            source_height=float(height),
            target_width=self.transformer.target_width,
            target_height=self.transformer.target_height,
        )

    def set_target_resolution(self, width: float, height: float) -> None:
        """Update target canonical resolution."""
        self.transformer = CoordinateTransformer(
            source_width=self.transformer.source_width,
            source_height=self.transformer.source_height,
            target_width=float(width),
            target_height=float(height),
        )

    def classify_detection(self, detection: TrackedDetection) -> SpatialDetection:
        """Classify a single TrackedDetection into a SpatialDetection."""
        # 1. Transform spatial points from source coordinates into canonical target space
        if self.calibration is not None and self.calibration.is_calibrated:
            norm_center = self.calibration.transform_point(detection.center)
            norm_bottom_center = self.calibration.transform_point(detection.bottom_center)
            norm_bbox = self.calibration.transform_bbox(detection.bbox)
        else:
            norm_center = self.transformer.transform_point(detection.center)
            norm_bottom_center = self.transformer.transform_point(detection.bottom_center)
            norm_bbox = self.transformer.transform_bbox(detection.bbox)

        # 2. Select spatial reference point (bottom-center for ground contact, or center)
        if self.use_bottom_center:
            target_point = norm_bottom_center
        else:
            target_point = norm_center

        # 3. Call the existing spatial analysis layer with canonical point
        zone = self.spatial_analyzer.classify(
            point=target_point,
            object_class=detection.class_name,
        )

        return SpatialDetection(
            track_id=detection.track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=norm_bbox,
            center=norm_center,
            bottom_center=norm_bottom_center,
            spatial_point=target_point,
            zone=zone,
            approach=zone.approach,
            zone_type=zone.zone_type,
            frame_index=detection.frame_index,
            camera_bbox=detection.bbox,
            camera_bottom_center=detection.bottom_center,
        )

    def adapt(self, detections: Sequence[TrackedDetection]) -> List[SpatialDetection]:
        """Adapt a collection of TrackedDetections into SpatialDetections."""
        if not detections:
            return []
        return [self.classify_detection(d) for d in detections]

    def __call__(self, detections: Sequence[TrackedDetection]) -> List[SpatialDetection]:
        """Convenience callable interface."""
        return self.adapt(detections)
