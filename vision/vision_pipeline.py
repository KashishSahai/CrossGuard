"""
CrossGuard AI Vision Pipeline
=============================
Ultralytics YOLOv8n object detection and ByteTrack persistent multi-object tracking.
Processes surveillance frames, extracts pedestrian and vehicular bounding boxes,
maintains unique track IDs, and produces high-tech annotated surveillance overlays.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.settings import (
    CLASS_NAMES,
    PEDESTRIAN_CLASS_IDS,
    TARGET_CLASS_IDS,
    VEHICLE_CLASS_IDS,
    VisionConfig,
)


@dataclass
class TrackedDetection:
    """Represents a single verified tracked entity in a surveillance frame."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]     # (cx, cy)
    assigned_zone: str = "TRANSIT"
    dwell_time_seconds: float = 0.0
    is_pedestrian: bool = False
    is_vehicle: bool = False


@dataclass
class VisionFrameResult:
    """Aggregated output from processing a single video frame."""
    frame_index: int
    timestamp_seconds: float
    tracked_detections: List[TrackedDetection]
    annotated_frame: np.ndarray
    pedestrian_count: int
    vehicle_count: int
    inference_time_ms: float
    fps: float
    device_used: str


# Global cached model singleton to avoid reloading per frame
_CACHED_MODEL: Optional[YOLO] = None
_CACHED_DEVICE: Optional[str] = None


def get_vision_model(model_name: str = "yolov8n.pt") -> Tuple[YOLO, str]:
    """
    Lazily loads and caches the YOLO model on CUDA (if available) or CPU.
    Thread-safe and persists across Streamlit reruns.
    """
    global _CACHED_MODEL, _CACHED_DEVICE
    if _CACHED_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_name)
        model.to(device)
        _CACHED_MODEL = model
        _CACHED_DEVICE = device
    return _CACHED_MODEL, _CACHED_DEVICE or "cpu"


class VisionPipeline:
    """
    Surveillance frame processing pipeline integrating YOLOv8n and ByteTrack.
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self.model, self.device = get_vision_model(self.config.model_name)
        # Track previous centroids for velocity estimation: {track_id: (cx, cy, last_time)}
        self._prev_positions: Dict[int, Tuple[float, float, float]] = {}

    def get_class_threshold(self, cls_id: int) -> float:
        """Returns the confidence threshold for a specific class."""
        if cls_id in PEDESTRIAN_CLASS_IDS:
            return getattr(self.config, "pedestrian_conf_threshold", 0.18)
        return getattr(self.config, "vehicle_conf_threshold", 0.22)

    def _get_class_threshold(self, cls_id: int) -> float:
        return self.get_class_threshold(cls_id)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        fps_video: float = 30.0,
        polygon_zones: Optional[Dict[str, np.ndarray]] = None,
    ) -> VisionFrameResult:
        """
        Runs object detection and ByteTrack tracking on a single frame.

        Args:
            frame: BGR numpy image frame from OpenCV.
            frame_index: Sequential frame index in video.
            fps_video: Video native frame rate for timestamp calculation.
            polygon_zones: Optional dictionary of named polygon arrays for visual overlay.

        Returns:
            VisionFrameResult containing tracked detections and annotated overlay.
        """
        t_start = time.perf_counter()
        timestamp = frame_index / fps_video if fps_video > 0 else 0.0

        # Determine effective confidence threshold based on settings
        conf_thresh = min(
            self.config.confidence_threshold,
            getattr(self.config, "pedestrian_conf_threshold", 0.18),
        )
        imgsz_target = getattr(self.config, "inference_size", 960)

        # Run YOLO with ByteTrack
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.config.tracker_type,
            classes=self.config.target_classes,
            conf=conf_thresh,
            iou=self.config.iou_threshold,
            imgsz=imgsz_target,
            verbose=False,
            device=self.device,
        )

        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000.0
        fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 30.0

        tracked_detections: List[TrackedDetection] = []
        res = results[0]

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy_arr))
            cls_arr = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy_arr))
            id_arr = boxes.id.cpu().numpy() if boxes.id is not None else np.arange(len(xyxy_arr))

            for i in range(len(xyxy_arr)):
                x1, y1, x2, y2 = [int(v) for v in xyxy_arr[i]]
                track_id = int(id_arr[i]) if id_arr is not None else i + 1
                cls_id = int(cls_arr[i])
                conf = float(conf_arr[i])
                cls_name = CLASS_NAMES.get(cls_id, "unknown")

                is_ped = cls_id in PEDESTRIAN_CLASS_IDS
                is_veh = cls_id in VEHICLE_CLASS_IDS

                # Enforce class-specific confidence filtering
                if is_ped and conf < getattr(self.config, "pedestrian_conf_threshold", 0.18):
                    continue
                if is_veh and conf < getattr(self.config, "vehicle_conf_threshold", 0.22):
                    continue

                cx = float((x1 + x2) / 2.0)
                if is_ped:
                    cy = float(y2)  # Base/feet centroid for accurate ground contact polygon test
                else:
                    cy = float((y1 + y2) / 2.0)  # Bounding box center for vehicle road lanes

                det = TrackedDetection(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    centroid=(cx, cy),
                    is_pedestrian=is_ped,
                    is_vehicle=is_veh,
                )
                tracked_detections.append(det)

        # Count active detections
        ped_count = sum(1 for d in tracked_detections if d.is_pedestrian)
        veh_count = sum(1 for d in tracked_detections if d.is_vehicle)

        # Generate annotated frame
        annotated = self._render_annotations(
            frame.copy(), tracked_detections, polygon_zones, fps
        )

        return VisionFrameResult(
            frame_index=frame_index,
            timestamp_seconds=round(timestamp, 2),
            tracked_detections=tracked_detections,
            annotated_frame=annotated,
            pedestrian_count=ped_count,
            vehicle_count=veh_count,
            inference_time_ms=round(inference_time_ms, 2),
            fps=round(fps, 1),
            device_used=self.device,
        )

    def _render_annotations(
        self,
        frame: np.ndarray,
        detections: List[TrackedDetection],
        polygon_zones: Optional[Dict[str, np.ndarray]],
        fps: float,
    ) -> np.ndarray:
        """Draws bounding boxes, labels, and polygon zones onto the frame."""
        h, w = frame.shape[:2]

        # 1. Draw spatial polygon zones if provided
        if polygon_zones:
            overlay = frame.copy()
            zone_colors = {
                "waiting_north": (250, 139, 167), # BGR Lavender
                "waiting_south": (250, 139, 167),
                "crosswalk": (255, 240, 0),      # BGR Cyan
                "vehicle_approach": (180, 50, 240) # BGR Magenta
            }
            for zone_name, pts in polygon_zones.items():
                if len(pts) >= 3:
                    pts_reshaped = pts.reshape((-1, 1, 2)).astype(np.int32)
                    color = zone_colors.get(zone_name, (0, 240, 255))
                    cv2.fillPoly(overlay, [pts_reshaped], color)
                    cv2.polylines(frame, [pts_reshaped], True, color, 2)
            # Alpha blend overlay for transparent filled zone
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        # 2. Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            if det.is_pedestrian:
                color = (255, 240, 0) # Neon Cyan in BGR
                label = f"#{det.track_id} PED {det.confidence:.2f}"
            else:
                color = (250, 139, 167) # Electric lavender
                label = f"#{det.track_id} {det.class_name.upper()} {det.confidence:.2f}"

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Ground-contact centroid indicator
            cx, cy = int(det.centroid[0]), int(det.centroid[1])
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # Label banner
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 6, y1), (10, 14, 23), -1)
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        # 3. Surveillance Telemetry Watermark in top-left
        hud_text = f"CROSSGUARD AI // FPS: {fps:.1f} | OBJS: {len(detections)} | DEV: {self.device.upper()}"
        cv2.putText(
            frame,
            hud_text,
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 240, 255),
            1,
            cv2.LINE_AA,
        )

        return frame
