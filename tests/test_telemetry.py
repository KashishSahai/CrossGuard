"""Unit tests for the Telemetry Layer (Step 4)."""
from __future__ import annotations

import pytest

from core.adaptive_controller import AdaptiveController
from core.models import DemandSnapshot
from core.perception_adapter import SpatialDetection
from core.spatial_analysis import Approach, SpatialZone, ZoneType
from core.telemetry import ApproachDemandSnapshot, TelemetryAggregator, TelemetrySnapshot


@pytest.fixture
def aggregator() -> TelemetryAggregator:
    return TelemetryAggregator(source_name="vision")


def make_spatial_detection(
    track_id: int,
    class_name: str,
    zone: SpatialZone,
    approach: Approach | None = None,
    zone_type: ZoneType | None = None,
    confidence: float = 0.9,
    point: tuple[float, float] = (0.0, 0.0),
    frame_index: int = 1,
) -> SpatialDetection:
    """Helper to create synthetic SpatialDetection instances."""
    app = approach if approach is not None else zone.approach
    zt = zone_type if zone_type is not None else zone.zone_type
    return SpatialDetection(
        track_id=track_id,
        class_id=0 if class_name == "person" else 2,
        class_name=class_name,
        confidence=confidence,
        bbox=(point[0] - 10.0, point[1] - 10.0, point[0] + 10.0, point[1]),
        center=(point[0], point[1] - 5.0),
        bottom_center=point,
        spatial_point=point,
        zone=zone,
        approach=app,
        zone_type=zt,
        frame_index=frame_index,
    )


# -----------------------------------------------------------------------------
# Test 1 — Empty input
# -----------------------------------------------------------------------------
def test_empty_input(aggregator: TelemetryAggregator):
    snap = aggregator.aggregate([])

    assert isinstance(snap, TelemetrySnapshot)
    assert snap.waiting_pedestrians == 0
    assert snap.crossing_pedestrians == 0
    assert snap.approaching_vehicles == 0
    assert snap.queued_vehicles == 0
    assert snap.total_vehicle_demand == 0
    assert len(snap.tracked_ids) == 0

    demand = snap.demand
    assert isinstance(demand, DemandSnapshot)
    assert demand.is_valid()
    assert demand.pedestrians_waiting == 0
    assert demand.pedestrians_in_crosswalk == 0
    assert demand.vehicle_demand == 0
    assert demand.vehicle_queue == 0
    assert demand.source == "vision"


# -----------------------------------------------------------------------------
# Test 2 — One waiting pedestrian
# -----------------------------------------------------------------------------
def test_one_waiting_pedestrian(aggregator: TelemetryAggregator):
    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING)
    ]
    snap = aggregator.aggregate(dets)

    assert snap.waiting_pedestrians == 1
    assert snap.crossing_pedestrians == 0
    assert snap.waiting_on(Approach.NORTH) == 1
    assert snap.waiting_on(Approach.SOUTH) == 0

    assert snap.demand.pedestrians_waiting == 1
    assert snap.demand.pedestrians_in_crosswalk == 0
    assert snap.demand.vehicle_demand == 0


# -----------------------------------------------------------------------------
# Test 3 — Multiple waiting pedestrians
# -----------------------------------------------------------------------------
def test_multiple_waiting_pedestrians(aggregator: TelemetryAggregator):
    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=3, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=4, class_name="person", zone=SpatialZone.SOUTH_WAITING),
        make_spatial_detection(track_id=5, class_name="person", zone=SpatialZone.SOUTH_WAITING),
    ]
    snap = aggregator.aggregate(dets)

    assert snap.waiting_pedestrians == 5
    assert snap.waiting_on(Approach.NORTH) == 3
    assert snap.waiting_on(Approach.SOUTH) == 2
    assert snap.demand.pedestrians_waiting == 5


# -----------------------------------------------------------------------------
# Test 4 — Pedestrians distributed across approaches
# -----------------------------------------------------------------------------
def test_pedestrians_distributed_across_approaches(aggregator: TelemetryAggregator):
    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.EAST_WAITING),
        make_spatial_detection(track_id=3, class_name="person", zone=SpatialZone.EAST_WAITING),
        make_spatial_detection(track_id=4, class_name="person", zone=SpatialZone.SOUTH_WAITING),
        make_spatial_detection(track_id=5, class_name="person", zone=SpatialZone.SOUTH_WAITING),
        make_spatial_detection(track_id=6, class_name="person", zone=SpatialZone.SOUTH_WAITING),
        make_spatial_detection(track_id=7, class_name="person", zone=SpatialZone.WEST_WAITING),
        make_spatial_detection(track_id=8, class_name="person", zone=SpatialZone.WEST_WAITING),
        make_spatial_detection(track_id=9, class_name="person", zone=SpatialZone.WEST_WAITING),
        make_spatial_detection(track_id=10, class_name="person", zone=SpatialZone.WEST_WAITING),
    ]
    snap = aggregator.aggregate(dets)

    assert snap.waiting_pedestrians == 10
    assert snap.waiting_on(Approach.NORTH) == 1
    assert snap.waiting_on(Approach.EAST) == 2
    assert snap.waiting_on(Approach.SOUTH) == 3
    assert snap.waiting_on(Approach.WEST) == 4


# -----------------------------------------------------------------------------
# Test 5 — Crossing pedestrians
# -----------------------------------------------------------------------------
def test_crossing_pedestrians(aggregator: TelemetryAggregator):
    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_CROSSING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.SOUTH_CROSSING),
    ]
    snap = aggregator.aggregate(dets)

    # Crossing must be strictly distinguished from waiting
    assert snap.crossing_pedestrians == 2
    assert snap.waiting_pedestrians == 0
    assert snap.crossing_on(Approach.NORTH) == 1
    assert snap.crossing_on(Approach.SOUTH) == 1

    assert snap.demand.pedestrians_in_crosswalk == 2
    assert snap.demand.pedestrians_waiting == 0


# -----------------------------------------------------------------------------
# Test 6 — Queued vehicles
# -----------------------------------------------------------------------------
def test_queued_vehicles(aggregator: TelemetryAggregator):
    dets = [
        make_spatial_detection(track_id=10, class_name="car", zone=SpatialZone.NORTH_QUEUE),
        make_spatial_detection(track_id=11, class_name="car", zone=SpatialZone.NORTH_QUEUE),
        make_spatial_detection(track_id=12, class_name="bus", zone=SpatialZone.EAST_QUEUE),
    ]
    snap = aggregator.aggregate(dets)

    assert snap.queued_vehicles == 3
    assert snap.approaching_vehicles == 0
    assert snap.queued_on(Approach.NORTH) == 2
    assert snap.queued_on(Approach.EAST) == 1
    assert snap.total_vehicle_demand == 3

    assert snap.demand.vehicle_queue == 3
    assert snap.demand.vehicle_demand == 3


# -----------------------------------------------------------------------------
# Test 7 — Approaching vehicles
# -----------------------------------------------------------------------------
def test_approaching_vehicles(aggregator: TelemetryAggregator):
    dets = [
        make_spatial_detection(track_id=20, class_name="car", zone=SpatialZone.SOUTH_APPROACH),
        make_spatial_detection(track_id=21, class_name="truck", zone=SpatialZone.SOUTH_APPROACH),
        make_spatial_detection(track_id=22, class_name="car", zone=SpatialZone.WEST_APPROACH),
    ]
    snap = aggregator.aggregate(dets)

    assert snap.approaching_vehicles == 3
    assert snap.queued_vehicles == 0
    assert snap.approaching_on(Approach.SOUTH) == 2
    assert snap.approaching_on(Approach.WEST) == 1
    assert snap.total_vehicle_demand == 3

    assert snap.demand.vehicle_queue == 0
    assert snap.demand.vehicle_demand == 3


# -----------------------------------------------------------------------------
# Test 8 — Mixed pedestrian + vehicle demand
# -----------------------------------------------------------------------------
def test_mixed_pedestrian_and_vehicle_demand(aggregator: TelemetryAggregator):
    dets = [
        # Pedestrians: 2 waiting, 1 crossing
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.EAST_WAITING),
        make_spatial_detection(track_id=3, class_name="person", zone=SpatialZone.WEST_CROSSING),
        # Vehicles: 2 queued, 3 approaching
        make_spatial_detection(track_id=4, class_name="car", zone=SpatialZone.NORTH_QUEUE),
        make_spatial_detection(track_id=5, class_name="bus", zone=SpatialZone.SOUTH_QUEUE),
        make_spatial_detection(track_id=6, class_name="car", zone=SpatialZone.EAST_APPROACH),
        make_spatial_detection(track_id=7, class_name="truck", zone=SpatialZone.WEST_APPROACH),
        make_spatial_detection(track_id=8, class_name="car", zone=SpatialZone.WEST_APPROACH),
    ]
    snap = aggregator.aggregate(dets, frame_index=42)

    assert snap.waiting_pedestrians == 2
    assert snap.crossing_pedestrians == 1
    assert snap.queued_vehicles == 2
    assert snap.approaching_vehicles == 3
    assert snap.total_vehicle_demand == 5
    assert len(snap.tracked_ids) == 8
    assert snap.frame_index == 42

    demand = snap.demand
    assert demand.is_valid()
    assert demand.pedestrians_waiting == 2
    assert demand.pedestrians_in_crosswalk == 1
    assert demand.vehicle_demand == 5
    assert demand.vehicle_queue == 2


# -----------------------------------------------------------------------------
# Test 9 — Duplicate track IDs do not inflate counts
# -----------------------------------------------------------------------------
def test_duplicate_track_ids_do_not_inflate_counts(aggregator: TelemetryAggregator):
    dets = [
        make_spatial_detection(track_id=99, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=99, class_name="person", zone=SpatialZone.NORTH_WAITING),  # Duplicate
        make_spatial_detection(track_id=99, class_name="person", zone=SpatialZone.NORTH_WAITING),  # Duplicate
        make_spatial_detection(track_id=101, class_name="car", zone=SpatialZone.EAST_QUEUE),
        make_spatial_detection(track_id=101, class_name="car", zone=SpatialZone.EAST_QUEUE),       # Duplicate
    ]
    snap = aggregator.aggregate(dets)

    assert snap.waiting_pedestrians == 1
    assert snap.queued_vehicles == 1
    assert snap.total_vehicle_demand == 1
    assert len(snap.tracked_ids) == 2


# -----------------------------------------------------------------------------
# Test 10 — Objects in NONE / outside relevant zones
# -----------------------------------------------------------------------------
def test_objects_in_none_outside_relevant_zones(aggregator: TelemetryAggregator):
    dets = [
        make_spatial_detection(track_id=1, class_name="car", zone=SpatialZone.NONE,
                               approach=None, zone_type=None),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.NONE,
                               approach=None, zone_type=None),
    ]
    snap = aggregator.aggregate(dets)

    assert snap.waiting_pedestrians == 0
    assert snap.crossing_pedestrians == 0
    assert snap.approaching_vehicles == 0
    assert snap.queued_vehicles == 0
    assert snap.total_vehicle_demand == 0
    assert len(snap.tracked_ids) == 2  # Track ID was observed but not counted into demand

    assert snap.demand.is_valid()
    assert snap.demand.pedestrians_waiting == 0
    assert snap.demand.vehicle_demand == 0


# -----------------------------------------------------------------------------
# Test 11 — Correct approach attribution
# -----------------------------------------------------------------------------
def test_correct_approach_attribution(aggregator: TelemetryAggregator):
    dets = [
        # North
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.NORTH_CROSSING),
        make_spatial_detection(track_id=3, class_name="car", zone=SpatialZone.NORTH_QUEUE),
        make_spatial_detection(track_id=4, class_name="car", zone=SpatialZone.NORTH_APPROACH),
        # South
        make_spatial_detection(track_id=5, class_name="person", zone=SpatialZone.SOUTH_WAITING),
        make_spatial_detection(track_id=6, class_name="person", zone=SpatialZone.SOUTH_CROSSING),
        make_spatial_detection(track_id=7, class_name="car", zone=SpatialZone.SOUTH_QUEUE),
        make_spatial_detection(track_id=8, class_name="car", zone=SpatialZone.SOUTH_APPROACH),
        # East
        make_spatial_detection(track_id=9, class_name="person", zone=SpatialZone.EAST_WAITING),
        make_spatial_detection(track_id=10, class_name="person", zone=SpatialZone.EAST_CROSSING),
        make_spatial_detection(track_id=11, class_name="car", zone=SpatialZone.EAST_QUEUE),
        make_spatial_detection(track_id=12, class_name="car", zone=SpatialZone.EAST_APPROACH),
        # West
        make_spatial_detection(track_id=13, class_name="person", zone=SpatialZone.WEST_WAITING),
        make_spatial_detection(track_id=14, class_name="person", zone=SpatialZone.WEST_CROSSING),
        make_spatial_detection(track_id=15, class_name="car", zone=SpatialZone.WEST_QUEUE),
        make_spatial_detection(track_id=16, class_name="car", zone=SpatialZone.WEST_APPROACH),
    ]
    snap = aggregator.aggregate(dets)

    for app in Approach:
        assert snap.waiting_on(app) == 1
        assert snap.crossing_on(app) == 1
        assert snap.queued_on(app) == 1
        assert snap.approaching_on(app) == 1
        app_snap = snap.by_approach[app]
        assert app_snap.total_pedestrians == 2
        assert app_snap.vehicle_demand == 2


# -----------------------------------------------------------------------------
# Test 12 — Correct DemandSnapshot generation and AdaptiveController compatibility
# -----------------------------------------------------------------------------
def test_correct_demand_snapshot_generation(aggregator: TelemetryAggregator):
    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=3, class_name="car", zone=SpatialZone.SOUTH_QUEUE),
        make_spatial_detection(track_id=4, class_name="car", zone=SpatialZone.SOUTH_APPROACH),
    ]

    demand = aggregator.to_demand_snapshot(dets, longest_pedestrian_wait_s=12.5)

    assert isinstance(demand, DemandSnapshot)
    assert demand.is_valid()
    assert demand.pedestrians_waiting == 2
    assert demand.longest_pedestrian_wait_s == 12.5
    assert demand.pedestrians_in_crosswalk == 0
    assert demand.vehicle_demand == 2
    assert demand.vehicle_queue == 1
    assert demand.source == "vision"

    # Verify that the existing AdaptiveController directly accepts this snapshot without error
    ctrl = AdaptiveController()
    state = ctrl.step(dt=0.1, demand=demand)
    assert state is not None
    assert state.fail_safe_active is False
