"""
CrossGuard Spatial Zone and Pedestrian Occupancy Engine
======================================================
Performs resolution-scaled polygon containment tests, tracked entity zone assignment,
waiting dwell-time calculation, and hysteresis-damped crosswalk occupancy tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config.settings import NormalizedZoneConfig
from vision.vision_pipeline import TrackedDetection


@dataclass
class TrackHistory:
    """Temporal tracking state for an individual pedestrian or vehicle."""
    track_id: int
    class_id: int
    is_pedestrian: bool
    is_vehicle: bool
    first_seen_time: float
    last_seen_time: float
    current_zone: str = "TRANSIT"
    waiting_start_time: Optional[float] = None
    current_wait_seconds: float = 0.0
    consecutive_waiting_frames: int = 0
    consecutive_crosswalk_frames: int = 0
    consecutive_absent_frames: int = 0


@dataclass
class SpatialSnapshot:
    """Comprehensive snapshot of spatial occupancy at a discrete video timecode."""
    timestamp: float
    waiting_pedestrians_count: int
    waiting_track_ids: List[int]
    max_pedestrian_waiting_time: float
    crosswalk_occupied: bool
    crossing_pedestrians_count: int
    crossing_track_ids: List[int]
    approach_vehicles_count: int
    vehicle_track_ids: List[int]
    queue_density_level: str  # "EMPTY", "LOW", "MEDIUM", "HIGH", "SATURATED"


class SpatialZoneEngine:
    """
    Manages resolution-scaled geometric zones, point-in-polygon assignment,
    dwell time accumulation, and crosswalk occupancy hysteresis.
    """

    def __init__(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        zone_config: Optional[NormalizedZoneConfig] = None,
    ):
        self.width = frame_width
        self.height = frame_height
        self.config = zone_config or NormalizedZoneConfig()

        # Cache of individual track temporal states
        self.track_histories: Dict[int, TrackHistory] = {}

        # Crosswalk occupancy hysteresis tracking
        self.crosswalk_consecutive_occupied_frames: int = 0
        self.crosswalk_consecutive_clear_frames: int = 0
        self.crosswalk_is_occupied_hysteresis: bool = False

        # Build initial pixel polygon arrays
        self.pixel_polygons: Dict[str, np.ndarray] = {}
        self.recalculate_pixel_polygons(self.width, self.height)

    @property
    def zone_config(self) -> NormalizedZoneConfig:
        return self.config

    @zone_config.setter
    def zone_config(self, val: NormalizedZoneConfig) -> None:
        self.config = val
        self.recalculate_pixel_polygons(self.width, self.height)

    def recalculate_pixel_polygons(self, width: int, height: int) -> None:
        """Scales normalized polygon definitions to the target video resolution."""
        self.width = max(width, 1)
        self.height = max(height, 1)

        def _scale(coords: List[Tuple[float, float]]) -> np.ndarray:
            pts = [[int(x * self.width), int(y * self.height)] for x, y in coords]
            return np.array(pts, dtype=np.int32)

        self.pixel_polygons = {
            "waiting_north": _scale(self.config.waiting_zone_north),
            "waiting_south": _scale(self.config.waiting_zone_south),
            "crosswalk": _scale(self.config.crosswalk_polygon),
            "vehicle_approach": _scale(self.config.vehicle_approach),
        }

    def point_in_polygon(self, pt: Tuple[float, float], polygon_name: str) -> bool:
        """Tests whether a 2D point (x, y) falls inside or on the boundary of a zone."""
        poly = self.pixel_polygons.get(polygon_name)
        if poly is None or len(poly) < 3:
            return False
        # cv2.pointPolygonTest returns positive if inside, 0 on edge, negative if outside
        res = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False)
        return res >= 0

    def update(
        self,
        tracked_detections: List[TrackedDetection],
        timestamp: float,
    ) -> SpatialSnapshot:
        """
        Updates tracking histories, calculates waiting dwell times, and generates a spatial snapshot.

        Args:
            tracked_detections: Detections output from VisionPipeline for the current frame.
            timestamp: Video elapsed time in seconds.

        Returns:
            SpatialSnapshot containing validated occupancy and dwell metrics.
        """
        current_frame_ids: Set[int] = set()

        raw_waiting_ids: List[int] = []
        raw_crossing_ids: List[int] = []
        raw_approach_vehicle_ids: List[int] = []

        # 1. Process active detections
        for det in tracked_detections:
            tid = det.track_id
            current_frame_ids.add(tid)
            cx, cy = det.centroid

            # Retrieve or initialize track history
            if tid not in self.track_histories:
                self.track_histories[tid] = TrackHistory(
                    track_id=tid,
                    class_id=det.class_id,
                    is_pedestrian=det.is_pedestrian,
                    is_vehicle=det.is_vehicle,
                    first_seen_time=timestamp,
                    last_seen_time=timestamp,
                )

            th = self.track_histories[tid]
            th.last_seen_time = timestamp
            th.consecutive_absent_frames = 0

            # Spatial Zone Classification
            if det.is_pedestrian:
                in_crosswalk = self.point_in_polygon((cx, cy), "crosswalk")
                in_north_wait = self.point_in_polygon((cx, cy), "waiting_north")
                in_south_wait = self.point_in_polygon((cx, cy), "waiting_south")
                in_waiting_zone = in_north_wait or in_south_wait

                # Update crosswalk presence with debounce
                if in_crosswalk:
                    th.consecutive_crosswalk_frames += 1
                    th.consecutive_waiting_frames = 0
                    th.current_zone = "CROSSWALK"
                    det.assigned_zone = "CROSSWALK"
                    # Reset waiting timer once pedestrian enters crosswalk
                    th.waiting_start_time = None
                    raw_crossing_ids.append(tid)
                elif in_waiting_zone:
                    th.consecutive_waiting_frames += 1
                    th.consecutive_crosswalk_frames = 0
                    th.current_zone = "WAITING_ZONE"
                    det.assigned_zone = "WAITING_ZONE"

                    # Accumulate dwell time
                    if th.waiting_start_time is None:
                        th.waiting_start_time = timestamp
                    th.current_wait_seconds = max(0.0, timestamp - th.waiting_start_time)
                    det.dwell_time_seconds = th.current_wait_seconds
                    raw_waiting_ids.append(tid)
                else:
                    th.current_zone = "TRANSIT"
                    det.assigned_zone = "TRANSIT"
                    th.consecutive_waiting_frames = 0
                    th.consecutive_crosswalk_frames = 0

            elif det.is_vehicle:
                # Test center of vehicle and bottom bumper ground contact
                in_approach = self.point_in_polygon((cx, cy), "vehicle_approach") or \
                              self.point_in_polygon((cx, float(det.bbox[3])), "vehicle_approach")
                if in_approach:
                    th.current_zone = "APPROACH_QUEUE"
                    det.assigned_zone = "APPROACH_QUEUE"
                    raw_approach_vehicle_ids.append(tid)
                else:
                    th.current_zone = "ROADWAY"
                    det.assigned_zone = "ROADWAY"

        # 2. Handle absent tracks & cleanup stale IDs (> 90 frames absent)
        stale_ids = []
        for tid, th in self.track_histories.items():
            if tid not in current_frame_ids:
                th.consecutive_absent_frames += 1
                if th.consecutive_absent_frames > 90:
                    stale_ids.append(tid)
        for tid in stale_ids:
            del self.track_histories[tid]

        # 3. Apply Crosswalk Hysteresis
        # Require consecutive observations to flip occupied flag to avoid flicker
        active_in_crosswalk_count = len(raw_crossing_ids)
        if active_in_crosswalk_count > 0:
            self.crosswalk_consecutive_occupied_frames += 1
            self.crosswalk_consecutive_clear_frames = 0
            if self.crosswalk_consecutive_occupied_frames >= self.config.consecutive_observations_required:
                self.crosswalk_is_occupied_hysteresis = True
        else:
            self.crosswalk_consecutive_clear_frames += 1
            self.crosswalk_consecutive_occupied_frames = 0
            if self.crosswalk_consecutive_clear_frames >= self.config.consecutive_observations_required:
                self.crosswalk_is_occupied_hysteresis = False

        # 4. Compute Max Pedestrian Wait Time among verified waiting tracks
        waiting_peds = [
            th for th in self.track_histories.values()
            if th.is_pedestrian and th.track_id in raw_waiting_ids
        ]
        max_wait = max([th.current_wait_seconds for th in waiting_peds], default=0.0)

        # 5. Classify Queue Density Level
        num_vehicles = len(raw_approach_vehicle_ids)
        if num_vehicles == 0:
            queue_level = "EMPTY"
        elif num_vehicles <= 2:
            queue_level = "LOW"
        elif num_vehicles <= 4:
            queue_level = "MEDIUM"
        elif num_vehicles <= 6:
            queue_level = "HIGH"
        else:
            queue_level = "SATURATED"

        return SpatialSnapshot(
            timestamp=round(timestamp, 2),
            waiting_pedestrians_count=len(raw_waiting_ids),
            waiting_track_ids=sorted(raw_waiting_ids),
            max_pedestrian_waiting_time=round(max_wait, 1),
            crosswalk_occupied=self.crosswalk_is_occupied_hysteresis,
            crossing_pedestrians_count=active_in_crosswalk_count,
            crossing_track_ids=sorted(raw_crossing_ids),
            approach_vehicles_count=num_vehicles,
            vehicle_track_ids=sorted(raw_approach_vehicle_ids),
            queue_density_level=queue_level,
        )
