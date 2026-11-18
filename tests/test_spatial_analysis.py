"""Unit tests for the Spatial Analysis Layer (Step 1)."""
from __future__ import annotations

import pytest

from core.spatial_analysis import (
    Approach,
    ApproachDemand,
    SpatialAnalyzer,
    SpatialDemandSummary,
    SpatialZone,
    TrackedObject,
    ZoneType,
    is_pedestrian,
    is_vehicle,
)
from simulation import geometry as G


@pytest.fixture
def analyzer() -> SpatialAnalyzer:
    return SpatialAnalyzer()


# -----------------------------------------------------------------------------
# Test 1 — North waiting pedestrian
# -----------------------------------------------------------------------------
def test_north_waiting_pedestrian(analyzer: SpatialAnalyzer):
    # End A (West curb): center is at (416, 230)
    assert analyzer.classify((416, 230), "person") == SpatialZone.NORTH_WAITING
    assert analyzer.classify((416, 230), "person") == "NORTH_WAITING"

    # End B (East curb): center is at (584, 230)
    assert analyzer.classify((584, 230), "person") == SpatialZone.NORTH_WAITING


# -----------------------------------------------------------------------------
# Test 2 — South crossing pedestrian
# -----------------------------------------------------------------------------
def test_south_crossing_pedestrian(analyzer: SpatialAnalyzer):
    # Mid-crossing on South zebra: (500, 470)
    assert analyzer.classify((500, 470), "person") == SpatialZone.SOUTH_CROSSING
    assert analyzer.classify((500, 470), "person") == "SOUTH_CROSSING"


# -----------------------------------------------------------------------------
# Test 3 — East approaching vehicle
# -----------------------------------------------------------------------------
def test_east_approaching_vehicle(analyzer: SpatialAnalyzer):
    # East incoming lane (westbound, y=315), upstream of queue zone (x > 744)
    assert analyzer.classify((850, 315), "car") == SpatialZone.EAST_APPROACH
    assert analyzer.classify((850, 315), "car") == "EAST_APPROACH"


# -----------------------------------------------------------------------------
# Test 4 — Vehicle queue
# -----------------------------------------------------------------------------
def test_vehicle_queue_all_approaches(analyzer: SpatialAnalyzer):
    # North queue: southbound lane x=465, y in [106, 206]
    assert analyzer.classify((465, 180), "car") == SpatialZone.NORTH_QUEUE

    # South queue: northbound lane x=535, y in [494, 594]
    assert analyzer.classify((535, 520), "bus") == SpatialZone.SOUTH_QUEUE

    # East queue: westbound lane y=315, x in [644, 744]
    assert analyzer.classify((700, 315), "truck") == SpatialZone.EAST_QUEUE

    # West queue: eastbound lane y=385, x in [256, 356]
    assert analyzer.classify((300, 385), "van") == SpatialZone.WEST_QUEUE


# -----------------------------------------------------------------------------
# Test 5 — Outside intersection
# -----------------------------------------------------------------------------
def test_outside_intersection_returns_none(analyzer: SpatialAnalyzer):
    # Northwest park / residential area
    assert analyzer.classify((100, 100), "person") == SpatialZone.NONE
    assert analyzer.classify((100, 100), "car") == SpatialZone.NONE

    # Southeast pond area
    assert analyzer.classify((800, 568), "person") == SpatialZone.NONE
    assert analyzer.classify((800, 568), "car") == SpatialZone.NONE

    # Out of world bounds
    assert analyzer.classify((-200, -200), "car") == SpatialZone.NONE
    assert analyzer.classify((1500, 1500), "person") == SpatialZone.NONE


# -----------------------------------------------------------------------------
# Test 6 — Object class separation
# -----------------------------------------------------------------------------
def test_object_class_separation(analyzer: SpatialAnalyzer):
    # A person in a vehicle approach or queue zone must NOT be classified as vehicle zone
    assert analyzer.classify((465, 180), "person") == SpatialZone.NONE
    assert analyzer.classify((850, 315), "person") == SpatialZone.NONE

    # A car in a pedestrian waiting or crossing zone must NOT be classified as pedestrian zone
    assert analyzer.classify((416, 230), "car") == SpatialZone.NONE
    assert analyzer.classify((500, 470), "car") == SpatialZone.NONE
    assert analyzer.classify((500, 230), "truck") == SpatialZone.NONE

    # Unknown or non-traffic classes
    assert analyzer.classify((416, 230), "dog") == SpatialZone.NONE
    assert analyzer.classify((465, 180), "traffic_cone") == SpatialZone.NONE
    assert analyzer.classify((465, 180), "") == SpatialZone.NONE


# -----------------------------------------------------------------------------
# Test 7 — Boundary behavior
# -----------------------------------------------------------------------------
def test_boundary_behavior_stop_lines(analyzer: SpatialAnalyzer):
    # On the stop line, vehicles are classified as QUEUE
    # North stop line: y = 206
    assert analyzer.classify((465, 206.0), "car") == SpatialZone.NORTH_QUEUE

    # South stop line: y = 494
    assert analyzer.classify((535, 494.0), "car") == SpatialZone.SOUTH_QUEUE

    # East stop line: x = 644
    assert analyzer.classify((644.0, 315), "car") == SpatialZone.EAST_QUEUE

    # West stop line: x = 356
    assert analyzer.classify((356.0, 385), "car") == SpatialZone.WEST_QUEUE

    # Past stop line (into the intersection / zebra) is not in queue or approach
    assert analyzer.classify((465, 207.0), "car") == SpatialZone.NONE
    assert analyzer.classify((535, 493.0), "car") == SpatialZone.NONE
    assert analyzer.classify((643.0, 315), "car") == SpatialZone.NONE
    assert analyzer.classify((357.0, 385), "car") == SpatialZone.NONE


def test_boundary_behavior_queue_approach_transition(analyzer: SpatialAnalyzer):
    # At the queue boundary (depth 100 upstream of stop line):
    # North stop line is 206, boundary is 106. Priority policy assigns 106 to QUEUE.
    assert analyzer.classify((465, 106.0), "car") == SpatialZone.NORTH_QUEUE
    # Just outside queue boundary into approach:
    assert analyzer.classify((465, 105.9), "car") == SpatialZone.NORTH_APPROACH

    # South: stop line 494, queue ends at 594.
    assert analyzer.classify((535, 594.0), "car") == SpatialZone.SOUTH_QUEUE
    assert analyzer.classify((535, 594.1), "car") == SpatialZone.SOUTH_APPROACH

    # East: stop line 644, queue ends at 744.
    assert analyzer.classify((744.0, 315), "car") == SpatialZone.EAST_QUEUE
    assert analyzer.classify((744.1, 315), "car") == SpatialZone.EAST_APPROACH

    # West: stop line 356, queue ends at 256.
    assert analyzer.classify((256.0, 385), "car") == SpatialZone.WEST_QUEUE
    assert analyzer.classify((255.9, 385), "car") == SpatialZone.WEST_APPROACH


def test_boundary_behavior_curb_waiting_vs_crossing(analyzer: SpatialAnalyzer):
    # For North crosswalk: curb boundary is x = 430.
    # At x = 430, pedestrian is stepping onto the zebra: CROSSING takes precedence.
    assert analyzer.classify((430.0, 230), "person") == SpatialZone.NORTH_CROSSING

    # Just on the sidewalk side: WAITING
    assert analyzer.classify((429.9, 230), "person") == SpatialZone.NORTH_WAITING

    # Outer boundary of North waiting zone: x = 400
    assert analyzer.classify((400.0, 230), "person") == SpatialZone.NORTH_WAITING
    assert analyzer.classify((389.9, 230), "person") == SpatialZone.NONE


# -----------------------------------------------------------------------------
# Additional Tests: All zones, BBox, Summarize, Opposing Lane
# -----------------------------------------------------------------------------
def test_all_approaches_waiting_and_crossing(analyzer: SpatialAnalyzer):
    # North
    assert analyzer.classify((416, 230), "person") == SpatialZone.NORTH_WAITING
    assert analyzer.classify((500, 230), "person") == SpatialZone.NORTH_CROSSING

    # South
    assert analyzer.classify((416, 470), "person") == SpatialZone.SOUTH_WAITING
    assert analyzer.classify((500, 470), "person") == SpatialZone.SOUTH_CROSSING

    # East
    assert analyzer.classify((620, 266), "person") == SpatialZone.EAST_WAITING
    assert analyzer.classify((620, 350), "person") == SpatialZone.EAST_CROSSING

    # West
    assert analyzer.classify((380, 266), "person") == SpatialZone.WEST_WAITING
    assert analyzer.classify((380, 350), "person") == SpatialZone.WEST_CROSSING


def test_opposing_departing_lane_is_not_approach(analyzer: SpatialAnalyzer):
    # North road: x in [430, 570]. Incoming southbound is [430, 500].
    # Outgoing northbound lane is [500, 570].
    # A vehicle at x=535, y=150 is moving away from the intersection northbound.
    assert analyzer.classify((535, 150), "car") == SpatialZone.NONE


def test_bbox_classification(analyzer: SpatialAnalyzer):
    # Bounding box bottom-center: (x0, y0, x1, y1) -> bottom-center is ((x0+x1)/2, y1)
    # Pedestrian at bottom-center (416, 230)
    bbox_ped = (410, 200, 422, 230)
    assert analyzer.classify(bbox=bbox_ped, object_class="person") == SpatialZone.NORTH_WAITING

    # Vehicle in North queue with bottom bumper at (465, 180)
    bbox_car = (450, 140, 480, 180)
    assert analyzer.classify(bbox=bbox_car, object_class="car") == SpatialZone.NORTH_QUEUE


def test_object_input_types(analyzer: SpatialAnalyzer):
    # TrackedObject instance
    obj1 = TrackedObject(id=1, x=465, y=50, object_class="car")
    assert analyzer.classify_object(obj1) == SpatialZone.NORTH_APPROACH

    # Dict format
    obj2 = {"x": 500, "y": 470, "object_class": "person"}
    assert analyzer.classify_object(obj2) == SpatialZone.SOUTH_CROSSING

    # Tuple format
    obj3 = ((644, 315), "truck")
    assert analyzer.classify_object(obj3) == SpatialZone.EAST_QUEUE

    obj4 = (380, 350, "person")
    assert analyzer.classify_object(obj4) == SpatialZone.WEST_CROSSING


def test_demand_summarize(analyzer: SpatialAnalyzer):
    objects = [
        # North: 2 waiting, 1 approaching vehicle
        TrackedObject(id=1, x=416, y=230, object_class="person"),
        TrackedObject(id=2, x=584, y=230, object_class="person"),
        TrackedObject(id=3, x=465, y=50, object_class="car"),
        # South: 1 crossing, 1 queued vehicle
        TrackedObject(id=4, x=500, y=470, object_class="pedestrian"),
        TrackedObject(id=5, x=535, y=520, object_class="bus"),
        # East: 2 queued vehicles
        TrackedObject(id=6, x=700, y=315, object_class="truck"),
        TrackedObject(id=7, x=650, y=315, object_class="car"),
        # Background / Non-demand objects
        TrackedObject(id=8, x=100, y=100, object_class="person"),  # outside
        TrackedObject(id=9, x=535, y=150, object_class="car"),     # departing lane
    ]

    summary = analyzer.summarize(objects)

    assert isinstance(summary, SpatialDemandSummary)
    assert summary.waiting_pedestrians == 2
    assert summary.crossing_pedestrians == 1
    assert summary.approaching_vehicles == 1
    assert summary.queued_vehicles == 3

    # Check by approach
    north = summary.by_approach[Approach.NORTH]
    assert north.waiting_pedestrians == 2
    assert north.crossing_pedestrians == 0
    assert north.approaching_vehicles == 1
    assert north.queued_vehicles == 0

    south = summary.by_approach[Approach.SOUTH]
    assert south.waiting_pedestrians == 0
    assert south.crossing_pedestrians == 1
    assert south.approaching_vehicles == 0
    assert south.queued_vehicles == 1

    east = summary.by_approach[Approach.EAST]
    assert east.queued_vehicles == 2


def test_all_16_zones_exist(analyzer: SpatialAnalyzer):
    zones = analyzer.all_zones()
    assert len(zones) == 16
    for z in SpatialZone:
        if z != SpatialZone.NONE:
            assert z in zones
            defn = analyzer.get_zone_definition(z)
            assert defn is not None
            assert defn.zone == z
            assert defn.approach in Approach
            assert defn.zone_type in ZoneType
