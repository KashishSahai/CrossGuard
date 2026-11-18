"""Visual perception overlay utilities for CrossGuard Live Vision.

Draws bounding boxes, tracking identifiers, class labels, detection confidence,
and spatial zone classifications onto video frames using existing detection data.
Pure visualization; does not modify detection data or geometry.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

from core.perception_adapter import SpatialDetection
from core.spatial_analysis import SpatialZone, ZoneType, is_pedestrian, is_vehicle
from vision.models import TrackedDetection


# High-contrast, professional BGR color palettes
COLOR_PEDESTRIAN_WAITING = (0, 165, 255)    # Amber / Orange
COLOR_PEDESTRIAN_CROSSING = (255, 215, 0)   # Cyan / Sky
COLOR_PEDESTRIAN_OTHER = (0, 140, 255)      # Deep Orange

COLOR_VEHICLE_APPROACH = (50, 205, 50)      # Emerald Green
COLOR_VEHICLE_QUEUE = (180, 105, 255)       # Purple / Magenta
COLOR_VEHICLE_OTHER = (200, 160, 100)       # Slate Blue

COLOR_OTHER_CLASS = (160, 160, 160)         # Gray


def get_detection_color(spatial_det: SpatialDetection) -> Tuple[int, int, int]:
    """Determine bounding box and banner color based on object class and spatial zone."""
    cls_lower = spatial_det.class_name.lower().strip()

    if is_pedestrian(cls_lower):
        if spatial_det.zone_type == ZoneType.WAITING:
            return COLOR_PEDESTRIAN_WAITING
        elif spatial_det.zone_type == ZoneType.CROSSING:
            return COLOR_PEDESTRIAN_CROSSING
        return COLOR_PEDESTRIAN_OTHER

    if is_vehicle(cls_lower):
        if spatial_det.zone_type == ZoneType.APPROACH:
            return COLOR_VEHICLE_APPROACH
        elif spatial_det.zone_type == ZoneType.QUEUE:
            return COLOR_VEHICLE_QUEUE
        return COLOR_VEHICLE_OTHER

    return COLOR_OTHER_CLASS


def format_detection_label(spatial_det: SpatialDetection) -> str:
    """Format human-readable label string for bounding box overlay."""
    cls_name = spatial_det.class_name.capitalize()
    tid = spatial_det.track_id
    conf = spatial_det.confidence
    zone_str = spatial_det.zone.value if spatial_det.zone else "NONE"

    # Shorten zone name for clean display
    if zone_str == "NONE":
        zone_short = "NONE"
    else:
        # e.g. EAST_WAITING -> E-WAIT, SOUTH_CROSSING -> S-CROSS
        parts = zone_str.split("_")
        if len(parts) == 2:
            zone_short = f"{parts[0][0]}-{parts[1][:5]}"
        else:
            zone_short = zone_str[:8]

    return f"{cls_name} #{tid} {conf:.2f} [{zone_short}]"


def draw_perception_overlay(
    frame: np.ndarray,
    spatial_detections: Sequence[SpatialDetection],
    draw_spatial_points: bool = True,
    tracked_detections: Optional[Sequence[TrackedDetection]] = None,
) -> np.ndarray:
    """Overlay bounding boxes, labels, and track IDs onto a copy of the video frame.

    Args:
        frame: BGR image from OpenCV VideoCapture.
        spatial_detections: List of SpatialDetection objects from PerceptionAdapter.
        draw_spatial_points: If True, draws a small circle at the bottom-center contact point.
        tracked_detections: Optional list of TrackedDetection objects for direct camera frame coordinates.

    Returns:
        Annotated BGR copy of the image.
    """
    if frame is None or len(spatial_detections) == 0:
        return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = frame.copy()
    img_h, img_w = annotated.shape[:2]

    tracked_map = {t.track_id: t for t in tracked_detections} if tracked_detections else {}

    for s_det in spatial_detections:
        # Determine bounding box on the camera frame:
        # 1. Use camera_bbox on SpatialDetection if available
        # 2. Look up tracked_detection by track_id if available
        # 3. Fall back to s_det.bbox (for uncalibrated/canonical frames)
        if getattr(s_det, "camera_bbox", None) is not None:
            x1, y1, x2, y2 = s_det.camera_bbox
            bc_x, bc_y = s_det.camera_bottom_center or ((x1 + x2) / 2.0, y2)
        elif tracked_map and s_det.track_id in tracked_map:
            t = tracked_map[s_det.track_id]
            x1, y1, x2, y2 = t.bbox
            bc_x, bc_y = t.bottom_center
        else:
            x1, y1, x2, y2 = s_det.bbox
            bc_x, bc_y = s_det.bottom_center

        ix1 = max(0, min(img_w - 1, int(round(x1))))
        iy1 = max(0, min(img_h - 1, int(round(y1))))
        ix2 = max(0, min(img_w - 1, int(round(x2))))
        iy2 = max(0, min(img_h - 1, int(round(y2))))

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        color = get_detection_color(s_det)

        # 1. Bounding box
        cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), color, 2)

        # 2. Ground contact / bottom-center dot
        if draw_spatial_points:
            ibc_x = max(0, min(img_w - 1, int(round(bc_x))))
            ibc_y = max(0, min(img_h - 1, int(round(bc_y))))
            cv2.circle(annotated, (ibc_x, ibc_y), 4, color, -1)
            cv2.circle(annotated, (ibc_x, ibc_y), 5, (255, 255, 255), 1)

        # 3. Label text with solid background tag
        label = format_detection_label(s_det)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        tag_y1 = max(0, iy1 - text_h - 6)
        tag_y2 = iy1
        tag_x2 = min(img_w - 1, ix1 + text_w + 6)

        # Draw tag background
        cv2.rectangle(annotated, (ix1, tag_y1), (tag_x2, tag_y2), color, -1)
        # Draw tag text in white
        text_y = tag_y2 - 3
        cv2.putText(
            annotated,
            label,
            (ix1 + 3, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated
