"""
Unit tests for Spatial Zone Engine, Point-in-Polygon, Dwell Time, and Hysteresis
"""

import numpy as np
import pytest
from config.settings import NormalizedZoneConfig
from core.spatial_analysis import SpatialSnapshot, SpatialZoneEngine
from vision.vision_pipeline import TrackedDetection


def _create_detection(
    track_id: int,
    class_id: int,
    cx: float,
    cy: float,
    is_ped: bool = True,
    is_veh: bool = False,
) -> TrackedDetection:
    """Helper to generate a tracked detection for spatial testing."""
    return TrackedDetection(
        track_id=track_id,
        class_id=class_id,
        class_name="person" if is_ped else "car",
        confidence=0.85,
        bbox=(int(cx - 10), int(cy - 20), int(cx + 10), int(cy)),
        centroid=(cx, cy),
        is_pedestrian=is_ped,
        is_vehicle=is_veh,
    )


def test_spatial_engine_polygon_scaling():
    """Verifies that normalized coordinates scale properly to arbitrary video dimensions."""
    cfg = NormalizedZoneConfig()
    engine = SpatialZoneEngine(frame_width=1000, frame_height=500, zone_config=cfg)

    # Check scaled pixels
    north_pts = engine.pixel_polygons["waiting_north"]
    assert len(north_pts) == 4
    # First point is (0.05, 0.45) -> (50, 225)
    assert north_pts[0][0] == 50
    assert north_pts[0][1] == 225


def test_point_in_polygon_detection():
    """Verifies accurate polygon containment checks."""
    engine = SpatialZoneEngine(frame_width=1000, frame_height=1000)

    # Point clearly inside north waiting zone (0.15 to 0.35, 0.45 to 0.58) -> (250, 500)
    assert engine.point_in_polygon((250, 500), "waiting_north") is True

    # Point outside all zones
    assert engine.point_in_polygon((50, 50), "waiting_north") is False
    assert engine.point_in_polygon((50, 50), "crosswalk") is False


def test_pedestrian_dwell_time_accumulation():
    """Verifies that dwell time accumulates while pedestrian remains in waiting zone."""
    engine = SpatialZoneEngine(frame_width=1000, frame_height=1000)

    # Frame 1 at t=1.0: Pedestrian #1 arrives in North waiting zone (250, 500)
    det = _create_detection(track_id=1, class_id=0, cx=250, cy=500, is_ped=True)
    snap1 = engine.update([det], timestamp=1.0)
    assert snap1.waiting_pedestrians_count == 1
    assert snap1.max_pedestrian_waiting_time == 0.0

    # Frame 2 at t=6.5: Pedestrian #1 still in waiting zone
    snap2 = engine.update([det], timestamp=6.5)
    assert snap2.waiting_pedestrians_count == 1
    assert snap2.max_pedestrian_waiting_time == pytest.approx(5.5, abs=0.1)


def test_crosswalk_occupancy_hysteresis():
    """Verifies debounce hysteresis: requires consecutive observations to flip occupied status."""
    cfg = NormalizedZoneConfig(consecutive_observations_required=3)
    engine = SpatialZoneEngine(frame_width=1000, frame_height=1000, zone_config=cfg)

    # Crosswalk is (0.35 to 0.65, 0.42 to 0.62) -> (500, 500)
    crossing_det = _create_detection(track_id=2, class_id=0, cx=500, cy=500, is_ped=True)

    # Frame 1: Occupied 1st observation -> Hysteresis not yet flipped
    s1 = engine.update([crossing_det], timestamp=1.0)
    assert s1.crosswalk_occupied is False

    # Frame 2: Occupied 2nd observation
    s2 = engine.update([crossing_det], timestamp=1.1)
    assert s2.crosswalk_occupied is False

    # Frame 3: Occupied 3rd observation -> Now flipped to True!
    s3 = engine.update([crossing_det], timestamp=1.2)
    assert s3.crosswalk_occupied is True


def test_vehicle_queue_density_levels():
    """Verifies approach vehicle queue counting and density level categorization."""
    engine = SpatialZoneEngine(frame_width=1000, frame_height=1000)

    # Approach is (0.20 to 0.80, 0.30 to 0.68) -> (500, 450)
    vehs = [
        _create_detection(track_id=10 + i, class_id=2, cx=500, cy=400 + i * 20, is_ped=False, is_veh=True)
        for i in range(5)
    ]
    snap = engine.update(vehs, timestamp=1.0)

    assert snap.approach_vehicles_count == 5
    assert snap.queue_density_level == "HIGH"
