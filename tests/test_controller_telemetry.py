"""Integration tests connecting Telemetry DemandSnapshot to the AdaptiveController (Step 5)."""
from __future__ import annotations

import math
import pytest

from core.adaptive_controller import AdaptiveController
from core.models import (
    PHASE_ORDER,
    ControllerConfig,
    ControllerState,
    Decision,
    DemandSnapshot,
    PedestrianSignal,
    Phase,
    VehicleSignal,
)
from core.perception_adapter import SpatialDetection
from core.spatial_analysis import Approach, SpatialZone, ZoneType
from core.telemetry import TelemetryAggregator, TelemetrySnapshot


@pytest.fixture
def aggregator() -> TelemetryAggregator:
    return TelemetryAggregator(source_name="vision")


@pytest.fixture
def controller() -> AdaptiveController:
    return AdaptiveController()


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
    """Helper to construct synthetic SpatialDetection objects."""
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
# Test 1 — No demand (safe default behavior)
# -----------------------------------------------------------------------------
def test_controller_no_demand_maintains_safe_green(
    aggregator: TelemetryAggregator, controller: AdaptiveController
):
    # Empty detections -> zero demand snapshot
    demand = aggregator.to_demand_snapshot([])

    # Run for 25 seconds
    for _ in range(250):
        state = controller.step(dt=0.1, demand=demand)

    assert state.current_phase == Phase.VEHICLE_GREEN
    assert state.vehicle_signal == VehicleSignal.GREEN
    assert state.pedestrian_signal == PedestrianSignal.DONT_WALK
    assert state.decision == Decision.VEHICLE_PRIORITY
    assert state.fail_safe_active is False
    assert "No pedestrians waiting" in state.decision_reason


# -----------------------------------------------------------------------------
# Test 2 — Pedestrian demand (pursues pedestrian service per FSM rules)
# -----------------------------------------------------------------------------
def test_controller_pedestrian_demand_initiates_priority(
    aggregator: TelemetryAggregator, controller: AdaptiveController
):
    # Pass minimum green first under no demand
    zero_demand = aggregator.to_demand_snapshot([])
    for _ in range(101):  # 10.1 seconds (min_vehicle_green is 10.0s)
        controller.step(dt=0.1, demand=zero_demand)

    assert controller.state.current_phase == Phase.VEHICLE_GREEN

    # 3 waiting pedestrians detected on North curb
    peds = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=3, class_name="person", zone=SpatialZone.NORTH_WAITING),
    ]
    demand = aggregator.to_demand_snapshot(peds)

    state = controller.step(dt=0.1, demand=demand)

    # Controller must initiate pedestrian priority sequence via yellow
    assert state.current_phase == Phase.VEHICLE_YELLOW
    assert state.decision == Decision.PEDESTRIAN_PRIORITY
    assert "3 pedestrians waiting" in state.decision_reason


# -----------------------------------------------------------------------------
# Test 3 — Vehicle demand (heavy vehicle demand preserves vehicle priority)
# -----------------------------------------------------------------------------
def test_controller_heavy_vehicle_demand_defers_pedestrians(
    aggregator: TelemetryAggregator, controller: AdaptiveController
):
    cfg = controller.config  # vehicle_release_threshold is 7, max_pedestrian_wait is 30.0s

    # Create heavy vehicle demand (8 vehicles) and moderate waiting pedestrians (2 peds)
    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.NORTH_WAITING),
    ]
    for vid in range(10, 18):  # 8 vehicles
        dets.append(
            make_spatial_detection(track_id=vid, class_name="car", zone=SpatialZone.SOUTH_APPROACH)
        )

    demand = aggregator.to_demand_snapshot(dets, longest_pedestrian_wait_s=5.0)

    # Advance past minimum green
    for _ in range(150):  # 15.0 seconds
        state = controller.step(dt=0.1, demand=demand)

    # Vehicles keep the green because vehicle demand is heavy (8 >= 7)
    assert state.current_phase == Phase.VEHICLE_GREEN
    assert state.decision == Decision.VEHICLE_PRIORITY
    assert "vehicle demand is heavy" in state.decision_reason

    # When wait reaches max_pedestrian_wait (30.0s), priority forced to pedestrians
    demand_max_wait = aggregator.to_demand_snapshot(dets, longest_pedestrian_wait_s=cfg.max_pedestrian_wait)
    state = controller.step(dt=0.1, demand=demand_max_wait)
    assert state.current_phase == Phase.VEHICLE_YELLOW
    assert state.decision == Decision.PEDESTRIAN_PRIORITY


# -----------------------------------------------------------------------------
# Test 4 — Active pedestrian crossing (clearance extended and protected)
# -----------------------------------------------------------------------------
def test_controller_active_crossing_protects_clearance(
    aggregator: TelemetryAggregator, controller: AdaptiveController
):
    cfg = controller.config

    # Jump to PEDESTRIAN_CLEARANCE for inspection
    controller.force_phase(Phase.PEDESTRIAN_CLEARANCE)

    # Advance through base clearance time (8.0s) with someone still crossing
    crossing_det = [
        make_spatial_detection(track_id=99, class_name="person", zone=SpatialZone.NORTH_CROSSING)
    ]
    demand_crossing = aggregator.to_demand_snapshot(crossing_det)

    # Step past base clearance (8.0s)
    for _ in range(85):  # 8.5s
        state = controller.step(dt=0.1, demand=demand_crossing)

    # Still in clearance because pedestrian is still in crosswalk!
    assert state.current_phase == Phase.PEDESTRIAN_CLEARANCE
    assert state.pedestrian_signal == PedestrianSignal.CLEARING
    assert state.vehicle_signal == VehicleSignal.RED
    assert "extending clearance" in state.decision_reason

    # Once pedestrians leave, next step releases vehicles
    empty_demand = aggregator.to_demand_snapshot([])
    state_cleared = controller.step(dt=0.1, demand=empty_demand)
    assert state_cleared.current_phase == Phase.VEHICLE_GREEN
    assert state_cleared.vehicle_signal == VehicleSignal.GREEN


# -----------------------------------------------------------------------------
# Test 5 — Minimum vehicle green (pedestrian demand cannot cut green short)
# -----------------------------------------------------------------------------
def test_controller_minimum_green_enforced_under_demand(
    aggregator: TelemetryAggregator, controller: AdaptiveController
):
    cfg = controller.config
    assert cfg.min_vehicle_green == 10.0

    # Urgent pedestrian demand present right from t = 0
    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=3, class_name="person", zone=SpatialZone.NORTH_WAITING),
    ]
    demand = aggregator.to_demand_snapshot(dets, longest_pedestrian_wait_s=25.0)

    # Run for 9.5 seconds (under minimum green 10.0s)
    for _ in range(95):
        state = controller.step(dt=0.1, demand=demand)
        assert state.current_phase == Phase.VEHICLE_GREEN
        assert state.vehicle_signal == VehicleSignal.GREEN

    assert "minimum vehicle green protects vehicle release" in state.decision_reason


# -----------------------------------------------------------------------------
# Test 6 — Yellow transition (mandatory yellow occurs before red)
# -----------------------------------------------------------------------------
def test_controller_yellow_transition_mandatory(
    aggregator: TelemetryAggregator, controller: AdaptiveController
):
    cfg = controller.config
    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.NORTH_WAITING),
    ]
    demand = aggregator.to_demand_snapshot(dets)

    # Run past min green until yellow triggers
    while controller.state.current_phase == Phase.VEHICLE_GREEN:
        controller.step(dt=0.1, demand=demand)

    assert controller.state.current_phase == Phase.VEHICLE_YELLOW
    assert controller.state.vehicle_signal == VehicleSignal.YELLOW
    assert controller.state.pedestrian_signal == PedestrianSignal.DONT_WALK

    # Yellow must last for yellow_time (3.5s)
    # Step 2.0s into yellow: still yellow
    state_mid = controller.step(dt=2.0, demand=demand)
    assert state_mid.current_phase == Phase.VEHICLE_YELLOW
    assert math.isclose(state_mid.time_remaining, 1.5, abs_tol=0.1)

    # Step another 1.6s (total 3.6s >= 3.5s): transitions to ALL_RED_CLEARANCE
    state_after = controller.step(dt=1.6, demand=demand)
    assert state_after.current_phase == Phase.ALL_RED_CLEARANCE


# -----------------------------------------------------------------------------
# Test 7 — All-red clearance (all-red clearance empties intersection before WALK)
# -----------------------------------------------------------------------------
def test_controller_all_red_clearance_mandatory(
    aggregator: TelemetryAggregator, controller: AdaptiveController
):
    cfg = controller.config
    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.NORTH_WAITING),
    ]
    demand = aggregator.to_demand_snapshot(dets)

    # Step past green and yellow into all-red
    total_ticks = int((cfg.min_vehicle_green + cfg.yellow_time + 0.2) * 10)
    for _ in range(total_ticks):
        controller.step(dt=0.1, demand=demand)

    assert controller.state.current_phase == Phase.ALL_RED_CLEARANCE
    assert controller.state.vehicle_signal == VehicleSignal.RED
    assert controller.state.pedestrian_signal == PedestrianSignal.DONT_WALK
    assert "all-red clearance empties the intersection" in controller.state.decision_reason


# -----------------------------------------------------------------------------
# Test 8 — Fail-safe (malformed/None demand triggers safe fallback)
# -----------------------------------------------------------------------------
def test_controller_failsafe_on_invalid_or_missing_telemetry(controller: AdaptiveController):
    cfg = controller.config

    # Supply None for failsafe_after (3.0s)
    for _ in range(int((cfg.failsafe_after + 0.5) * 10)):
        state = controller.step(dt=0.1, demand=None)

    assert state.fail_safe_active is True
    assert state.decision == Decision.FIXED_TIME_FALLBACK
    assert "fixed-time fallback" in state.decision_reason

    # Recovery: valid snapshot clears fail-safe immediately
    valid_demand = DemandSnapshot(source="vision")
    state_recovered = controller.step(dt=0.1, demand=valid_demand)
    assert state_recovered.fail_safe_active is False


# -----------------------------------------------------------------------------
# Test 9 — Repeated snapshots (successive snapshots do not corrupt state)
# -----------------------------------------------------------------------------
def test_controller_repeated_snapshots_stability(
    aggregator: TelemetryAggregator, controller: AdaptiveController
):
    allowed_transitions = {
        Phase.VEHICLE_GREEN: {Phase.VEHICLE_GREEN, Phase.VEHICLE_YELLOW},
        Phase.VEHICLE_YELLOW: {Phase.VEHICLE_YELLOW, Phase.ALL_RED_CLEARANCE},
        Phase.ALL_RED_CLEARANCE: {Phase.ALL_RED_CLEARANCE, Phase.PEDESTRIAN_WALK},
        Phase.PEDESTRIAN_WALK: {Phase.PEDESTRIAN_WALK, Phase.PEDESTRIAN_CLEARANCE},
        Phase.PEDESTRIAN_CLEARANCE: {Phase.PEDESTRIAN_CLEARANCE, Phase.VEHICLE_GREEN},
    }

    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="car", zone=SpatialZone.SOUTH_QUEUE),
    ]

    prev_phase = controller.state.current_phase

    # Run for 200 consecutive ticks
    for frame_i in range(200):
        snap = aggregator.aggregate(dets, frame_index=frame_i)
        state = controller.step(dt=0.1, demand=snap.demand)
        assert state.current_phase in allowed_transitions[prev_phase]
        prev_phase = state.current_phase


# -----------------------------------------------------------------------------
# Test 10 — Controller accepts both DemandSnapshot and TelemetrySnapshot
# -----------------------------------------------------------------------------
def test_controller_accepts_both_demand_snapshot_and_telemetry_snapshot(
    aggregator: TelemetryAggregator, controller: AdaptiveController
):
    dets = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
    ]

    # 1. Feed pure DemandSnapshot
    pure_demand = aggregator.to_demand_snapshot(dets)
    s1 = controller.step(dt=0.1, demand=pure_demand)
    assert s1 is not None

    # 2. Feed rich TelemetrySnapshot directly
    rich_snapshot = aggregator.aggregate(dets)
    assert isinstance(rich_snapshot, TelemetrySnapshot)
    s2 = controller.step(dt=0.1, demand=rich_snapshot)
    assert s2 is not None


# -----------------------------------------------------------------------------
# Test 11 — End-to-End Logic Integration Test
# (SpatialDetection[] -> TelemetryAggregator -> DemandSnapshot -> AdaptiveController -> Decision)
# -----------------------------------------------------------------------------
def test_end_to_end_spatial_to_telemetry_to_controller(
    aggregator: TelemetryAggregator, controller: AdaptiveController
):
    """Proves the perception-derived spatial detections drive controller decisions cleanly."""
    cfg = controller.config

    # Scene: 3 waiting pedestrians on North curb, 1 car approaching on South lane
    scene_detections = [
        make_spatial_detection(track_id=1, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=2, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=3, class_name="person", zone=SpatialZone.NORTH_WAITING),
        make_spatial_detection(track_id=4, class_name="car", zone=SpatialZone.SOUTH_APPROACH),
    ]

    # 1. Spatial detections -> Telemetry snapshot
    telemetry_snap = aggregator.aggregate(scene_detections)
    assert telemetry_snap.waiting_pedestrians == 3
    assert telemetry_snap.waiting_on(Approach.NORTH) == 3
    assert telemetry_snap.approaching_vehicles == 1
    assert telemetry_snap.approaching_on(Approach.SOUTH) == 1

    # 2. Controller consumes telemetry demand
    demand = telemetry_snap.demand
    assert demand.is_valid()
    assert demand.source == "vision"

    # Step through minimum vehicle green (10.0s)
    for _ in range(100):
        controller.step(dt=0.1, demand=demand)

    assert controller.state.current_phase == Phase.VEHICLE_GREEN
    assert controller.state.vehicle_signal == VehicleSignal.GREEN

    # Step at t = 10.1s: controller initiates pedestrian priority
    state_priority = controller.step(dt=0.1, demand=demand)
    assert state_priority.current_phase == Phase.VEHICLE_YELLOW
    assert state_priority.decision == Decision.PEDESTRIAN_PRIORITY
    assert state_priority.vehicle_signal == VehicleSignal.YELLOW

    # Advance through yellow (3.5s)
    for _ in range(int(cfg.yellow_time * 10)):
        controller.step(dt=0.1, demand=demand)

    # Advance through all-red (3.0s)
    for _ in range(int(cfg.all_red_time * 10)):
        controller.step(dt=0.1, demand=demand)

    # Intersection is now in PEDESTRIAN_WALK!
    assert controller.state.current_phase == Phase.PEDESTRIAN_WALK
    assert controller.state.vehicle_signal == VehicleSignal.RED
    assert controller.state.pedestrian_signal == PedestrianSignal.WALK
    assert controller.state.pedestrian_phases_served >= 1
