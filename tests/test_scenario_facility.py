"""
Unit and Integration Tests for CrossGuard Scenario Demonstration Facility
========================================================================
Validates that all 7 controlled scenarios (Cases A-F and Fail-Safe Stop)
function according to safety standards and FSM transition logic in isolation.
"""

import pytest
from core.adaptive_controller import AdaptiveSignalController, SignalPhase
from core.spatial_analysis import SpatialSnapshot
from dashboard.scenario_facility import SCENARIOS, load_scenario_preset, step_sandbox


def test_scenario_definitions_complete():
    """Ensures all 7 formal operational policies are defined in SCENARIOS."""
    expected_keys = {"CASE_A", "CASE_B", "CASE_C", "CASE_D", "CASE_E", "CASE_F", "FAIL_SAFE"}
    assert set(SCENARIOS.keys()) == expected_keys
    for k, sc in SCENARIOS.items():
        assert "title" in sc
        assert "description" in sc
        assert "peds_waiting" in sc
        assert "max_dwell" in sc
        assert "initial_phase" in sc


def test_scenario_sandbox_case_a_arterial_green():
    """Case A maintains green with zero pedestrians waiting."""
    ctrl = AdaptiveSignalController()
    snap = SpatialSnapshot(
        timestamp=20.0,
        waiting_pedestrians_count=0,
        waiting_track_ids=[],
        max_pedestrian_waiting_time=0.0,
        crosswalk_occupied=False,
        crossing_pedestrians_count=0,
        crossing_track_ids=[],
        approach_vehicles_count=8,
        vehicle_track_ids=[1, 2, 3],
        queue_density_level="HIGH",
    )
    dec = ctrl.step(snap, timestamp=20.0)
    assert dec.current_phase == SignalPhase.VEHICLE_GREEN
    assert dec.decision_case == "CASE_A"


def test_scenario_sandbox_case_c_platoon_priority():
    """Case C triggers vehicle yellow when platoon threshold (>=3) is met."""
    ctrl = AdaptiveSignalController()
    # Fast forward past min green (12.0s)
    ctrl.phase_elapsed = 14.0
    ctrl.phase_start_time = 0.0

    snap = SpatialSnapshot(
        timestamp=14.0,
        waiting_pedestrians_count=4,
        waiting_track_ids=[101, 102, 103, 104],
        max_pedestrian_waiting_time=2.0,
        crosswalk_occupied=False,
        crossing_pedestrians_count=0,
        crossing_track_ids=[],
        approach_vehicles_count=5,
        vehicle_track_ids=[1, 2],
        queue_density_level="MEDIUM",
    )
    dec = ctrl.step(snap, timestamp=14.0)
    assert dec.current_phase == SignalPhase.VEHICLE_YELLOW
    assert dec.decision_case == "CASE_C"


def test_scenario_sandbox_case_d_starvation_prevention():
    """Case D overrides vehicular flow when a single pedestrian exceeds max waiting time (25s)."""
    ctrl = AdaptiveSignalController()
    ctrl.phase_elapsed = 15.0
    ctrl.phase_start_time = 0.0

    snap = SpatialSnapshot(
        timestamp=26.0,
        waiting_pedestrians_count=1,
        waiting_track_ids=[99],
        max_pedestrian_waiting_time=26.5,
        crosswalk_occupied=False,
        crossing_pedestrians_count=0,
        crossing_track_ids=[],
        approach_vehicles_count=10,
        vehicle_track_ids=list(range(10)),
        queue_density_level="HIGH",
    )
    dec = ctrl.step(snap, timestamp=26.0)
    assert dec.current_phase == SignalPhase.VEHICLE_YELLOW
    assert dec.decision_case == "CASE_D"


def test_scenario_sandbox_case_e_clearance_extension():
    """Case E grants dynamic clearance extensions while crosswalk is occupied."""
    ctrl = AdaptiveSignalController()
    ctrl.current_phase = SignalPhase.PEDESTRIAN_CLEARANCE
    ctrl.current_nominal_duration = 12.0
    ctrl.phase_elapsed = 12.5

    snap = SpatialSnapshot(
        timestamp=12.5,
        waiting_pedestrians_count=0,
        waiting_track_ids=[],
        max_pedestrian_waiting_time=0.0,
        crosswalk_occupied=True,
        crossing_pedestrians_count=1,
        crossing_track_ids=[201],
        approach_vehicles_count=3,
        vehicle_track_ids=[1, 2],
        queue_density_level="LOW",
    )
    dec = ctrl.step(snap, timestamp=12.5)
    assert dec.decision_case == "CASE_E"
    assert dec.clearance_extensions == 1
    assert dec.current_phase == SignalPhase.PEDESTRIAN_CLEARANCE


def test_scenario_sandbox_case_f_clear_return_to_green():
    """Case F requires consecutive clear observations before releasing vehicle green."""
    ctrl = AdaptiveSignalController()
    ctrl.current_phase = SignalPhase.PEDESTRIAN_CLEARANCE
    ctrl.current_nominal_duration = 12.0
    ctrl.phase_elapsed = 12.5
    ctrl.consecutive_clear_observations = 2  # Needs 1 more to reach 3

    snap = SpatialSnapshot(
        timestamp=12.5,
        waiting_pedestrians_count=0,
        waiting_track_ids=[],
        max_pedestrian_waiting_time=0.0,
        crosswalk_occupied=False,
        crossing_pedestrians_count=0,
        crossing_track_ids=[],
        approach_vehicles_count=3,
        vehicle_track_ids=[1, 2],
        queue_density_level="LOW",
    )
    dec = ctrl.step(snap, timestamp=12.5)
    assert dec.decision_case == "CASE_F"
    assert dec.current_phase == SignalPhase.VEHICLE_GREEN


def test_scenario_sandbox_fail_safe():
    """Fail-safe interlock forces all-red stop."""
    ctrl = AdaptiveSignalController()
    ctrl.trigger_fault("Camera hardware fault", timestamp=10.0)
    assert ctrl.fault_active is True
    assert ctrl.current_phase == SignalPhase.FAIL_SAFE_STOP
    assert ctrl.last_decision.decision_case == "FAIL_SAFE"
