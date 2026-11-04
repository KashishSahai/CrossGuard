"""
CrossGuard Central Video Processing Session Manager
===================================================
Establishes a unified, single-source-of-truth session managing video ingestion,
incremental frame decoding, YOLOv8n + ByteTrack inference, spatial analysis,
adaptive signal controller state, Bloom City digital twin, and session diagnostics.
"""

import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config.settings import NormalizedZoneConfig, SignalFSMConfig, VisionConfig
from core.adaptive_controller import AdaptiveSignalController, ControllerDecision, SignalPhase
from core.spatial_analysis import SpatialSnapshot, SpatialZoneEngine
from core.telemetry_manager import TelemetryManager
from simulation.bloom_city_engine import BloomCityEngine
from vision.vision_pipeline import TrackedDetection, VisionFrameResult, VisionPipeline


class SessionStatus(str, Enum):
    """Lifecycle status states for the shared video processing session."""
    AWAITING_UPLOAD = "AWAITING_UPLOAD"
    VIDEO_UPLOADED = "VIDEO_UPLOADED"
    INITIALIZING = "INITIALIZING"
    PROCESSING = "PROCESSING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass
class SessionDiagnostics:
    """Comprehensive pipeline diagnostics and data-integrity metrics."""
    total_frames_processed: int = 0
    total_detections_count: int = 0
    detections_by_class: Dict[str, int] = field(default_factory=lambda: {
        "person": 0, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0
    })
    unique_tracks_by_class: Dict[str, Set[int]] = field(default_factory=lambda: {
        "person": set(), "car": set(), "motorcycle": set(), "bus": set(), "truck": set()
    })
    unique_track_ids: Set[int] = field(default_factory=set)
    active_track_ids: List[int] = field(default_factory=list)
    zone_entries: Dict[str, int] = field(default_factory=lambda: {
        "waiting_north": 0, "waiting_south": 0, "crosswalk": 0, "vehicle_approach": 0
    })
    zero_detection_frames: int = 0
    tracking_failure_frames: int = 0  # Detections present but no ByteTrack IDs assigned
    confidence_sum: float = 0.0
    confidence_count: int = 0
    inference_errors: List[str] = field(default_factory=list)

    @property
    def avg_confidence(self) -> float:
        return (self.confidence_sum / self.confidence_count) if self.confidence_count > 0 else 0.0

    @property
    def unique_vehicles_count(self) -> int:
        return len(
            self.unique_tracks_by_class.get("car", set()) |
            self.unique_tracks_by_class.get("bus", set()) |
            self.unique_tracks_by_class.get("truck", set()) |
            self.unique_tracks_by_class.get("motorcycle", set())
        )

    @property
    def unique_pedestrians_count(self) -> int:
        return len(self.unique_tracks_by_class.get("person", set()))


class VideoSession:
    """
    Central Video Processing Session: The single source of truth for the entire application.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        vision_config: Optional[VisionConfig] = None,
        zone_config: Optional[NormalizedZoneConfig] = None,
        signal_config: Optional[SignalFSMConfig] = None,
    ):
        self.session_id: str = session_id or f"SES-{uuid.uuid4().hex[:8].upper()}"
        self.status: SessionStatus = SessionStatus.AWAITING_UPLOAD
        self.error_message: Optional[str] = None

        # Video source parameters
        self.video_path: Optional[str] = None
        self.video_metadata: Optional[Dict[str, Any]] = None

        # Configurations
        self.vision_config = vision_config or VisionConfig()
        self.zone_config = zone_config or NormalizedZoneConfig()
        self.signal_config = signal_config or SignalFSMConfig()
        self.frame_stride: int = 3
        self.inference_size: int = self.vision_config.inference_size

        # Subsystems
        self.pipeline: Optional[VisionPipeline] = None
        self.spatial_engine = SpatialZoneEngine(zone_config=self.zone_config)
        self.controller = AdaptiveSignalController(config=self.signal_config)
        self.bloom_engine = BloomCityEngine()
        self.telemetry_manager = TelemetryManager()

        # Processing pointer & stats
        self.current_frame_idx: int = 0
        self.processed_frames_count: int = 0
        self.start_wall_time: Optional[float] = None
        self.end_wall_time: Optional[float] = None
        self.total_proc_seconds: float = 0.0
        self.inference_latencies_ms: List[float] = []
        self._processed_config_fingerprint: Optional[str] = None

        # Diagnostics
        self.diagnostics = SessionDiagnostics()

        # Latest state snapshots (for UI rendering across dashboard pages)
        self.latest_result: Optional[VisionFrameResult] = None
        self.latest_snapshot: SpatialSnapshot = SpatialSnapshot(
            timestamp=0.0,
            waiting_pedestrians_count=0,
            waiting_track_ids=[],
            max_pedestrian_waiting_time=0.0,
            crosswalk_occupied=False,
            crossing_pedestrians_count=0,
            crossing_track_ids=[],
            approach_vehicles_count=0,
            vehicle_track_ids=[],
            queue_density_level="EMPTY",
        )
        self.latest_decision: ControllerDecision = ControllerDecision(
            timestamp=0.0,
            current_phase=SignalPhase.VEHICLE_GREEN,
            previous_phase=SignalPhase.VEHICLE_GREEN,
            phase_elapsed_seconds=0.0,
            phase_remaining_seconds=12.0,
            decision_case="CASE_A",
            reason="Awaiting surveillance video stream ingestion.",
            pedestrians_waiting=0,
            max_wait_time=0.0,
            crosswalk_occupied=False,
            vehicles_in_approach=0,
            clearance_extensions=0,
        )

        # Cached preview frame for zone calibration
        self._cached_first_frame: Optional[np.ndarray] = None

    @property
    def unique_pedestrians_count(self) -> int:
        """Returns cumulative unique pedestrian track IDs observed across the session."""
        return self.diagnostics.unique_pedestrians_count if self.diagnostics else 0

    @property
    def unique_vehicles_count(self) -> int:
        """Returns cumulative unique vehicle track IDs observed across the session."""
        return self.diagnostics.unique_vehicles_count if self.diagnostics else 0

    @property
    def total_detections_count(self) -> int:
        """Returns cumulative frame-level detections count."""
        return self.diagnostics.total_detections_count if self.diagnostics else 0

    @property
    def unique_tracks_count(self) -> int:
        """Returns cumulative unique target entities (all classes)."""
        return len(self.diagnostics.unique_track_ids) if self.diagnostics else 0

    @property
    def active_frame_detections_count(self) -> int:
        """Returns number of detections currently visible in the active frame."""
        return len(self.latest_result.tracked_detections) if self.latest_result else 0

    @property
    def active_pedestrians_count(self) -> int:
        """Returns number of pedestrians visible in the active frame."""
        if not self.latest_result or not self.latest_result.tracked_detections:
            return 0
        return sum(1 for d in self.latest_result.tracked_detections if d.is_pedestrian)

    @property
    def active_vehicles_count(self) -> int:
        """Returns number of vehicles visible in the active frame."""
        if not self.latest_result or not self.latest_result.tracked_detections:
            return 0
        return sum(1 for d in self.latest_result.tracked_detections if not d.is_pedestrian)

    @property
    def pedestrians_served_count(self) -> int:
        """Returns number of unique pedestrians who entered the crosswalk and were served."""
        return len(self.telemetry_manager._served_pedestrians) if self.telemetry_manager else 0

    @property
    def total_source_frames(self) -> int:
        """Returns total frame count from original video source stream."""
        if not self.video_metadata:
            return 0
        return int(self.video_metadata.get("frame_count", 0))

    @property
    def effective_inference_fps(self) -> float:
        """Returns true average FPS of actual inference runs."""
        if not self.inference_latencies_ms:
            return 0.0
        avg_lat = sum(self.inference_latencies_ms) / len(self.inference_latencies_ms)
        return (1000.0 / avg_lat) if avg_lat > 0 else 0.0

    @property
    def mean_inference_latency_ms(self) -> float:
        """Returns mean inference latency in milliseconds."""
        if not self.inference_latencies_ms:
            return 0.0
        return sum(self.inference_latencies_ms) / len(self.inference_latencies_ms)

    @property
    def config_fingerprint(self) -> str:
        """Returns deterministic hash signature of current processing configuration."""
        zone_coords = [
            self.zone_config.waiting_zone_north,
            self.zone_config.waiting_zone_south,
            self.zone_config.crosswalk_polygon,
            self.zone_config.vehicle_approach,
        ]
        s = (
            f"{self.video_path}:"
            f"{self.vision_config.inference_size}:"
            f"{self.frame_stride}:"
            f"{self.vision_config.confidence_threshold:.3f}:"
            f"{self.vision_config.pedestrian_conf_threshold:.3f}:"
            f"{self.vision_config.vehicle_conf_threshold:.3f}:"
            f"{str(zone_coords)}"
        )
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    @property
    def is_stale(self) -> bool:
        """Returns True if processing parameters were altered after frames were processed."""
        if self.processed_frames_count == 0:
            return False
        if self._processed_config_fingerprint is None:
            return False
        return self._processed_config_fingerprint != self.config_fingerprint

    def load_video(self, video_path: str, metadata: Dict[str, Any]) -> None:
        """Loads a verified video and initializes spatial calibration."""
        if not os.path.exists(video_path):
            self.status = SessionStatus.FAILED
            self.error_message = f"Video file not found at: {video_path}"
            return

        self.video_path = video_path
        self.video_metadata = metadata
        self.status = SessionStatus.VIDEO_UPLOADED
        self.error_message = None

        # Re-scale spatial engine to video resolution
        w = metadata.get("width", 1920)
        h = metadata.get("height", 1080)
        self.spatial_engine.recalculate_pixel_polygons(w, h)

        # Extract first frame for visual calibration preview
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                self._cached_first_frame = frame
            cap.release()

        # Reset counters for the new video
        self.reset_processing()

    def reset_processing(self) -> None:
        """Resets frame pointers, FSM state, and telemetry while preserving video file."""
        self.current_frame_idx = 0
        self.processed_frames_count = 0
        self.total_proc_seconds = 0.0
        self.inference_latencies_ms.clear()
        self.diagnostics = SessionDiagnostics()
        self.telemetry_manager = TelemetryManager()
        self.spatial_engine = SpatialZoneEngine(
            frame_width=self.video_metadata.get("width", 1920) if self.video_metadata else 1920,
            frame_height=self.video_metadata.get("height", 1080) if self.video_metadata else 1080,
            zone_config=self.zone_config,
        )
        self.controller = AdaptiveSignalController(config=self.signal_config)
        self.bloom_engine = BloomCityEngine()
        self.latest_result = None
        self._processed_config_fingerprint = None
        if self.pipeline:
            self.pipeline._prev_positions.clear()

        if self.video_path and os.path.exists(self.video_path):
            self.status = SessionStatus.VIDEO_UPLOADED
        else:
            self.status = SessionStatus.AWAITING_UPLOAD

    def reset_session(self) -> None:
        """Completely resets the session, releasing the video and clearing state."""
        self.video_path = None
        self.video_metadata = None
        self._cached_first_frame = None
        self.reset_processing()
        self.status = SessionStatus.AWAITING_UPLOAD

    def update_zones(self, zone_config: NormalizedZoneConfig) -> None:
        """Updates zone configuration and recalculates pixel boundaries."""
        self.zone_config = zone_config
        self.spatial_engine.config = zone_config
        w = self.video_metadata.get("width", 1920) if self.video_metadata else 1920
        h = self.video_metadata.get("height", 1080) if self.video_metadata else 1080
        self.spatial_engine.recalculate_pixel_polygons(w, h)
        self._cached_first_frame = None

    def step_frame(
        self, frame_idx: Optional[int] = None
    ) -> Tuple[Optional[VisionFrameResult], SpatialSnapshot, ControllerDecision]:
        """Processes a single sequential frame and returns the resulting vision, spatial, and controller snapshots."""
        if not self.video_path or not os.path.exists(self.video_path):
            raise ValueError("No video loaded to step.")
        target_idx = frame_idx if frame_idx is not None else self.current_frame_idx
        cap = cv2.VideoCapture(self.video_path)
        try:
            self.process_step(cap, target_idx)
            self.current_frame_idx = target_idx + 1
        finally:
            cap.release()
        return self.latest_result, self.latest_snapshot, self.latest_decision

    def get_calibration_preview_frame(self, draw_detections: bool = True) -> Optional[np.ndarray]:
        """Returns the first frame of the uploaded video with configured zones drawn on top, optionally with detection centroids."""
        if self._cached_first_frame is None and self.video_path and os.path.exists(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret and frame is not None:
                self._cached_first_frame = frame
            cap.release()

        if self._cached_first_frame is None:
            return None

        # Draw current polygons onto a copy of the frame
        frame = self._cached_first_frame.copy()
        overlay = frame.copy()

        colors = {
            "waiting_north": (250, 139, 167), # Lavender
            "waiting_south": (250, 139, 167), # Lavender
            "crosswalk": (255, 240, 0),       # Cyan
            "vehicle_approach": (180, 50, 240) # Magenta
        }
        labels = {
            "waiting_north": "NORTH WAITING ZONE",
            "waiting_south": "SOUTH WAITING ZONE",
            "crosswalk": "ZEBRA CROSSWALK",
            "vehicle_approach": "APPROACH QUEUE"
        }

        for name, pts in self.spatial_engine.pixel_polygons.items():
            if len(pts) >= 3:
                pts_reshaped = pts.reshape((-1, 1, 2)).astype(np.int32)
                col = colors.get(name, (0, 240, 255))
                cv2.fillPoly(overlay, [pts_reshaped], col)
                cv2.polylines(frame, [pts_reshaped], True, col, 2)
                # Label
                centroid_x = int(np.mean([p[0] for p in pts]))
                centroid_y = int(np.mean([p[1] for p in pts]))
                cv2.putText(
                    frame,
                    labels.get(name, name.upper()),
                    (centroid_x - 40, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Draw centroids if detections exist from the latest processed frame
        if draw_detections and self.latest_result and self.latest_result.tracked_detections:
            for det in self.latest_result.tracked_detections:
                cx, cy = int(det.centroid[0]), int(det.centroid[1])
                is_ped = det.is_pedestrian
                dot_col = (255, 240, 0) if is_ped else (180, 50, 240) # Cyan for ped, Magenta for veh
                cv2.circle(frame, (cx, cy), 6, dot_col, -1)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    f"#{det.track_id} {det.class_name[:3]}",
                    (cx + 8, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def _ensure_pipeline_initialized(self) -> VisionPipeline:
        """Initializes the VisionPipeline lazily on demand."""
        if self.pipeline is None:
            self.pipeline = VisionPipeline(config=self.vision_config)
        return self.pipeline

    def process_step(self, cap: cv2.VideoCapture, target_frame_idx: int) -> bool:
        """Processes a single frame at the target sequential index without redundant seeks."""
        curr_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if curr_pos != target_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            return False

        t0 = time.perf_counter()
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

        pipeline = self._ensure_pipeline_initialized()

        # 1. Vision Detection & ByteTrack
        res = pipeline.process_frame(
            frame=frame,
            frame_index=target_frame_idx,
            fps_video=fps_video,
            polygon_zones=self.spatial_engine.pixel_polygons,
        )

        # 2. Spatial Analysis
        snapshot = self.spatial_engine.update(
            tracked_detections=res.tracked_detections,
            timestamp=res.timestamp_seconds,
        )

        # 3. Adaptive Signal Controller
        decision = self.controller.step(snapshot, timestamp=res.timestamp_seconds)

        # 4. Telemetry Recording
        self.telemetry_manager.record_step(snapshot, decision, frame_idx=target_frame_idx)

        # 5. Update Diagnostics
        diag = self.diagnostics
        diag.total_frames_processed += 1
        diag.total_detections_count += len(res.tracked_detections)
        diag.active_track_ids = [d.track_id for d in res.tracked_detections]

        if len(res.tracked_detections) == 0:
            diag.zero_detection_frames += 1

        for det in res.tracked_detections:
            diag.unique_track_ids.add(det.track_id)
            diag.confidence_sum += det.confidence
            diag.confidence_count += 1
            cname = det.class_name.lower()
            if cname in diag.detections_by_class:
                diag.detections_by_class[cname] += 1
            else:
                diag.detections_by_class[cname] = 1

            if cname in diag.unique_tracks_by_class:
                diag.unique_tracks_by_class[cname].add(det.track_id)
            else:
                diag.unique_tracks_by_class[cname] = {det.track_id}

            if det.assigned_zone == "WAITING_ZONE":
                diag.zone_entries["waiting_north"] += 1
            elif det.assigned_zone == "CROSSWALK":
                diag.zone_entries["crosswalk"] += 1
            elif det.assigned_zone == "APPROACH_QUEUE":
                diag.zone_entries["vehicle_approach"] += 1

        # 6. Bloom City Digital Twin Sync
        self.bloom_engine.sync(
            snapshot,
            decision,
            delta_time=0.1,
            unique_vehicle_count=diag.unique_vehicles_count,
            unique_pedestrian_count=diag.unique_pedestrians_count,
        )

        t_elapsed = time.perf_counter() - t0
        self.total_proc_seconds += t_elapsed
        self.inference_latencies_ms.append(res.inference_time_ms)
        self.processed_frames_count += 1
        self.current_frame_idx = target_frame_idx

        # Save latest state
        self.latest_result = res
        self.latest_snapshot = snapshot
        self.latest_decision = decision

        return True

    def process_frames_chunk(
        self,
        num_frames: int = 10,
        stride: int = 3,
    ) -> int:
        """Processes an incremental batch of frames preserving chronological tracking."""
        if not self.video_path or not os.path.exists(self.video_path):
            self.status = SessionStatus.FAILED
            self.error_message = "No valid video file uploaded."
            return 0

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status = SessionStatus.FAILED
            self.error_message = "Failed to open video file stream."
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.status = SessionStatus.PROCESSING

        processed_in_chunk = 0
        try:
            for _ in range(num_frames):
                if self.current_frame_idx >= total_frames:
                    self.status = SessionStatus.COMPLETED
                    break

                target_idx = self.current_frame_idx
                success = self.process_step(cap, target_idx)
                if not success:
                    break
                processed_in_chunk += 1
                self.current_frame_idx += stride

                # Fast-forward intermediate frames with grab (7x faster than keyframe seeking)
                for _ in range(stride - 1):
                    if not cap.grab():
                        break

            self._processed_config_fingerprint = self.config_fingerprint
            if self.current_frame_idx >= total_frames:
                self.status = SessionStatus.COMPLETED
            else:
                self.status = SessionStatus.PAUSED
        except Exception as ex:
            self.status = SessionStatus.FAILED
            self.error_message = f"Processing error: {str(ex)}"
            self.diagnostics.inference_errors.append(str(ex))
        finally:
            cap.release()

        return processed_in_chunk

    def process_video_full(
        self,
        stride: int = 3,
        frame_limit: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> int:
        """
        Runs processing on the uploaded video with configurable stride and progress reporting.
        Uses sequential frame grabbing to eliminate seek latency on CPU.
        """
        if not self.video_path or not os.path.exists(self.video_path):
            self.status = SessionStatus.FAILED
            self.error_message = "No valid video file loaded."
            return 0

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status = SessionStatus.FAILED
            self.error_message = "Unable to open video stream."
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if frame_limit is not None and frame_limit > 0:
            total_frames = min(total_frames, frame_limit)

        self.status = SessionStatus.PROCESSING
        self.start_wall_time = time.perf_counter()

        processed_count = 0
        try:
            while self.current_frame_idx < total_frames:
                if stop_check and stop_check():
                    self.status = SessionStatus.STOPPED
                    break

                target_idx = self.current_frame_idx
                success = self.process_step(cap, target_idx)
                if not success:
                    break
                processed_count += 1

                if progress_callback:
                    pct = min(1.0, target_idx / total_frames)
                    progress_callback(target_idx, total_frames, pct)

                self.current_frame_idx += stride

                # Fast-forward intermediate frames with grab
                for _ in range(stride - 1):
                    if not cap.grab():
                        break

            self.end_wall_time = time.perf_counter()
            self._processed_config_fingerprint = self.config_fingerprint
            if self.status != SessionStatus.STOPPED:
                self.status = SessionStatus.COMPLETED
        except Exception as ex:
            self.status = SessionStatus.FAILED
            self.error_message = f"Inference pipeline exception: {str(ex)}"
            self.diagnostics.inference_errors.append(str(ex))
        finally:
            cap.release()

        return processed_count
