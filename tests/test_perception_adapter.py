"""Unit tests for the Perception Adapter (Step 3)."""
from __future__ import annotations

import pytest

from core.perception_adapter import PerceptionAdapter, SpatialDetection
from core.spatial_analysis import Approach, SpatialZone, ZoneType
from vision.models import TrackedDetection


@pytest.fixture
def adapter() -> PerceptionAdapter:
    return PerceptionAdapter()


def make_detection(
    track_id: int = 1,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.9,
    x1: float = 0.0,
    y1: float = 0.0,
    x2: float = 10.0,
    y2: float = 10.0,
    frame_index: int = 1,
) -> TrackedDetection:
    """Helper to create a TrackedDetection with automatically computed centers."""
    return TrackedDetection(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        frame_index=frame_index,
    )


# -----------------------------------------------------------------------------
# Test 1 — Person waiting zone
# -----------------------------------------------------------------------------
def test_person_waiting_zone(adapter: PerceptionAdapter):
    # North waiting curb at (416, 230)
    det = make_detection(
        track_id=7,
        class_id=0,
        class_name="person",
        x1=410.0,
        y1=190.0,
        x2=422.0,
        y2=230.0,  # bottom_center = (416.0, 230.0)
    )

    sd = adapter.classify_detection(det)

    assert isinstance(sd, SpatialDetection)
    assert sd.track_id == 7
    assert sd.class_name == "person"
    assert sd.zone == SpatialZone.NORTH_WAITING
    assert sd.approach == Approach.NORTH
    assert sd.zone_type == ZoneType.WAITING
    assert sd.is_in_zone is True


# -----------------------------------------------------------------------------
# Test 2 — Person crossing zone
# -----------------------------------------------------------------------------
def test_person_crossing_zone(adapter: PerceptionAdapter):
    # South crossing zebra at (500, 470)
    det = make_detection(
        track_id=8,
        class_id=0,
        class_name="person",
        x1=494.0,
        y1=430.0,
        x2=506.0,
        y2=470.0,  # bottom_center = (500.0, 470.0)
    )

    sd = adapter.classify_detection(det)

    assert sd.track_id == 8
    assert sd.class_name == "person"
    assert sd.zone == SpatialZone.SOUTH_CROSSING
    assert sd.approach == Approach.SOUTH
    assert sd.zone_type == ZoneType.CROSSING
    assert sd.is_in_zone is True


# -----------------------------------------------------------------------------
# Test 3 — Vehicle queue
# -----------------------------------------------------------------------------
def test_vehicle_queue(adapter: PerceptionAdapter):
    # East approach queue lane at (700, 315)
    det = make_detection(
        track_id=12,
        class_id=2,
        class_name="car",
        x1=680.0,
        y1=300.0,
        x2=720.0,
        y2=315.0,  # bottom_center = (700.0, 315.0)
    )

    sd = adapter.classify_detection(det)

    assert sd.track_id == 12
    assert sd.class_name == "car"
    assert sd.zone == SpatialZone.EAST_QUEUE
    assert sd.approach == Approach.EAST
    assert sd.zone_type == ZoneType.QUEUE
    assert sd.is_in_zone is True


# -----------------------------------------------------------------------------
# Test 4 — Vehicle approach
# -----------------------------------------------------------------------------
def test_vehicle_approach(adapter: PerceptionAdapter):
    # West approach advance lane at (150, 385)
    det = make_detection(
        track_id=15,
        class_id=2,
        class_name="car",
        x1=130.0,
        y1=370.0,
        x2=170.0,
        y2=385.0,  # bottom_center = (150.0, 385.0)
    )

    sd = adapter.classify_detection(det)

    assert sd.track_id == 15
    assert sd.class_name == "car"
    assert sd.zone == SpatialZone.WEST_APPROACH
    assert sd.approach == Approach.WEST
    assert sd.zone_type == ZoneType.APPROACH
    assert sd.is_in_zone is True


# -----------------------------------------------------------------------------
# Test 5 — Unknown/outside area
# -----------------------------------------------------------------------------
def test_unknown_outside_area(adapter: PerceptionAdapter):
    # Far in the Northwest residential corner (100, 100)
    det = make_detection(
        track_id=20,
        class_id=2,
        class_name="car",
        x1=80.0,
        y1=70.0,
        x2=120.0,
        y2=100.0,  # bottom_center = (100.0, 100.0)
    )

    sd = adapter.classify_detection(det)

    assert sd.track_id == 20
    assert sd.zone == SpatialZone.NONE
    assert sd.approach is None
    assert sd.zone_type is None
    assert sd.is_in_zone is False


# -----------------------------------------------------------------------------
# Test 6 — Object-class separation
# -----------------------------------------------------------------------------
def test_object_class_separation(adapter: PerceptionAdapter):
    # A person standing in a vehicle queue zone (465, 180) must NOT become QUEUE
    det_person = make_detection(
        track_id=30,
        class_id=0,
        class_name="person",
        x1=459.0,
        y1=140.0,
        x2=471.0,
        y2=180.0,  # bottom_center = (465.0, 180.0)
    )
    sd_person = adapter.classify_detection(det_person)
    assert sd_person.zone == SpatialZone.NONE
    assert sd_person.approach is None

    # A car in a pedestrian waiting zone (416, 230) must NOT become WAITING
    det_car = make_detection(
        track_id=31,
        class_id=2,
        class_name="car",
        x1=396.0,
        y1=200.0,
        x2=436.0,
        y2=230.0,  # bottom_center = (416.0, 230.0)
    )
    sd_car = adapter.classify_detection(det_car)
    assert sd_car.zone == SpatialZone.NONE
    assert sd_car.approach is None


# -----------------------------------------------------------------------------
# Test 7 — Track identity preservation
# -----------------------------------------------------------------------------
def test_track_identity_preservation(adapter: PerceptionAdapter):
    for expected_id in (1, 42, 999, 10502):
        det = make_detection(
            track_id=expected_id,
            class_id=0,
            class_name="person",
            x1=410.0,
            y1=190.0,
            x2=422.0,
            y2=230.0,
        )
        sd = adapter.classify_detection(det)
        assert sd.track_id == expected_id


# -----------------------------------------------------------------------------
# Test 8 — Bottom-center usage
# -----------------------------------------------------------------------------
def test_bottom_center_usage():
    # Construct a bounding box where center is outside the North waiting zone,
    # but bottom_center (feet/ground contact) is inside the North waiting zone.
    # North waiting zone: x in [400, 430), y in [205, 255]
    # Here: x1=410, x2=422 -> cx = 416
    # y1=150, y2=230 -> center_y = 190 (OUTSIDE, < 205), bottom_y = 230 (INSIDE, in [205, 255])
    det = make_detection(
        track_id=55,
        class_id=0,
        class_name="person",
        x1=410.0,
        y1=150.0,
        x2=422.0,
        y2=230.0,
    )

    # 1. With bottom_center (default behavior)
    adapter_bc = PerceptionAdapter(use_bottom_center=True)
    sd_bc = adapter_bc.classify_detection(det)
    assert sd_bc.spatial_point == (416.0, 230.0)
    assert sd_bc.zone == SpatialZone.NORTH_WAITING

    # 2. With center
    adapter_center = PerceptionAdapter(use_bottom_center=False)
    sd_center = adapter_center.classify_detection(det)
    assert sd_center.spatial_point == (416.0, 190.0)
    assert sd_center.zone == SpatialZone.NONE


# -----------------------------------------------------------------------------
# Test 9 — Empty input
# -----------------------------------------------------------------------------
def test_empty_input(adapter: PerceptionAdapter):
    assert adapter.adapt([]) == []
    assert adapter([]) == []


# -----------------------------------------------------------------------------
# Additional Tests: Batch adaptation and serialization
# -----------------------------------------------------------------------------
def test_batch_adaptation(adapter: PerceptionAdapter):
    detections = [
        make_detection(track_id=1, class_name="person", x1=410, y1=190, x2=422, y2=230),  # North waiting
        make_detection(track_id=2, class_name="person", x1=494, y1=430, x2=506, y2=470),  # South crossing
        make_detection(track_id=3, class_name="car", x1=680, y1=300, x2=720, y2=315),     # East queue
        make_detection(track_id=4, class_name="truck", x1=130, y1=370, x2=170, y2=385),   # West approach
        make_detection(track_id=5, class_name="car", x1=80, y1=70, x2=120, y2=100),        # Outside (NONE)
    ]

    results = adapter.adapt(detections)
    assert len(results) == 5

    assert results[0].zone == SpatialZone.NORTH_WAITING
    assert results[1].zone == SpatialZone.SOUTH_CROSSING
    assert results[2].zone == SpatialZone.EAST_QUEUE
    assert results[3].zone == SpatialZone.WEST_APPROACH
    assert results[4].zone == SpatialZone.NONE

    # Test serialization of the first result
    d0 = results[0].to_dict()
    assert d0["track_id"] == 1
    assert d0["class_name"] == "person"
    assert d0["zone"] == "NORTH_WAITING"
    assert d0["approach"] == "NORTH"
    assert d0["zone_type"] == "WAITING"
