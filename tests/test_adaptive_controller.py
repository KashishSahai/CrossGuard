"""
Unit tests for the Adaptive Signal Controller FSM (Cases A through F & Fail-Safe)
"""

import pytest
from config.settings import SignalFSMConfig
from core.adaptive_controller import (
    AdaptiveSignalController,
    ControllerDecision,
    SignalPhase,
)
from core.spatial_analysis import SpatialSnapshot


def _mock_snapshot(
    waiting_peds: int = 0,
    max_wait: float = 0.0,
    crosswalk_occupied: bool = False,
    crossing_peds: int = 0,
    approach_vehicles: int = 0,
) -> SpatialSnapshot:
    """Helper to generate spatial snapshots for testing."""
    return SpatialSnapshot(
        timestamp=0.0,
        waiting_pedestrians_count=waiting_peds,
        waiting_track_ids=list(range(1, waiting_peds + 1)),
        max_pedestrian_waiting_time=max_wait,
        crosswalk_occupied=crosswalk_occupied,
        crossing_pedestrians_count=crossing_peds,
        crossing_track_ids=list(range(100, 100 + crossing_peds)),
        approach_vehicles_count=approach_vehicles,
        vehicle_track_ids=[],
        queue_density_level="LOW",
    )


def test_case_a_no_pedestrians_maintains_green():
    """Case A: When no pedestrians are waiting, vehicular green is maintained."""
    fsm = AdaptiveSignalController()
    snap = _mock_snapshot(waiting_peds=0)

    # Step at t=0.0
    dec = fsm.step(snap, timestamp=0.0)
    assert dec.current_phase == SignalPhase.VEHICLE_GREEN
    assert dec.decision_case == "CASE_A"

    # Step at t=20.0 (past min green of 12s)
    dec = fsm.step(snap, timestamp=20.0)
    assert dec.current_phase == SignalPhase.VEHICLE_GREEN
    assert dec.decision_case == "CASE_A"


def test_case_b_small_group_accumulation_interval():
    """Case B: Small group (< threshold) triggers accumulation interval before switching."""
    cfg = SignalFSMConfig(
        min_vehicle_green=10.0,
        pedestrian_group_threshold=3,
        accumulation_interval=6.0,
    )
    fsm = AdaptiveSignalController(cfg)
    snap = _mock_snapshot(waiting_peds=1, max_wait=2.0)

    # First arrival at t=12.0 (min green satisfied)
    dec = fsm.step(snap, timestamp=12.0)
    assert dec.current_phase == SignalPhase.VEHICLE_GREEN
    assert dec.decision_case == "CASE_B"
    assert "accumulation" in dec.reason.lower()

    # Step at t=15.0 (3s elapsed in accumulation < 6.0s)
    dec = fsm.step(snap, timestamp=15.0)
    assert dec.current_phase == SignalPhase.VEHICLE_GREEN
    assert dec.decision_case == "CASE_B"

    # Step at t=19.0 (7s elapsed in accumulation > 6.0s interval)
    dec = fsm.step(snap, timestamp=19.0)
    assert dec.current_phase == SignalPhase.VEHICLE_YELLOW
    assert dec.decision_case == "CASE_B"


def test_case_c_group_threshold_triggers_priority():
    """Case C: Group threshold reached triggers prompt vehicle yellow transition."""
    cfg = SignalFSMConfig(
        min_vehicle_green=10.0,
        pedestrian_group_threshold=3,
    )
    fsm = AdaptiveSignalController(cfg)
    snap = _mock_snapshot(waiting_peds=4, max_wait=3.0)

    # At t=12.0 (min green met, 4 >= 3 threshold)
    dec = fsm.step(snap, timestamp=12.0)
    assert dec.current_phase == SignalPhase.VEHICLE_YELLOW
    assert dec.decision_case == "CASE_C"


def test_case_d_max_waiting_time_prevents_starvation():
    """Case D: Single pedestrian waiting too long gets priority regardless of group size."""
    cfg = SignalFSMConfig(
        min_vehicle_green=10.0,
        pedestrian_group_threshold=5,
        max_waiting_time=20.0,
    )
    fsm = AdaptiveSignalController(cfg)
    # Only 1 pedestrian, but max wait is 21.0s >= 20.0s
    snap = _mock_snapshot(waiting_peds=1, max_wait=21.0)

    dec = fsm.step(snap, timestamp=12.0)
    assert dec.current_phase == SignalPhase.VEHICLE_YELLOW
    assert dec.decision_case == "CASE_D"
    assert "starvation" in dec.reason.lower() or "maximum pedestrian waiting time" in dec.reason.lower()


def test_case_e_crosswalk_occupied_extends_clearance():
    """Case E: Clearance is extended if pedestrians remain on the crossing."""
    cfg = SignalFSMConfig(
        pedestrian_clearance_duration=8.0,
        max_clearance_extensions=2,
        extension_seconds=3.0,
    )
    fsm = AdaptiveSignalController(cfg)

    # Fast forward into PEDESTRIAN_CLEARANCE at t=20.0
    fsm.current_phase = SignalPhase.PEDESTRIAN_CLEARANCE
    fsm.phase_start_time = 20.0
    fsm.current_nominal_duration = 8.0

    # Crosswalk occupied at t=28.0 (nominal 8.0s elapsed)
    snap = _mock_snapshot(crosswalk_occupied=True, crossing_peds=1)
    dec = fsm.step(snap, timestamp=28.0)

    assert dec.current_phase == SignalPhase.PEDESTRIAN_CLEARANCE
    assert dec.decision_case == "CASE_E"
    assert dec.clearance_extensions == 1
    assert "extend" in dec.reason.lower()


def test_case_f_crossing_clear_consecutive_observations():
    """Case F: Releasing conflicting vehicle traffic requires consecutive clear frames."""
    cfg = SignalFSMConfig(
        pedestrian_clearance_duration=8.0,
        consecutive_clear_observations=3,
    )
    fsm = AdaptiveSignalController(cfg)

    # In PEDESTRIAN_CLEARANCE at t=20.0
    fsm.current_phase = SignalPhase.PEDESTRIAN_CLEARANCE
    fsm.phase_start_time = 20.0
    fsm.current_nominal_duration = 8.0

    clear_snap = _mock_snapshot(crosswalk_occupied=False, crossing_peds=0)

    # Frame 1 clear at t=28.5 (needs 3 consecutive)
    dec1 = fsm.step(clear_snap, timestamp=28.5)
    assert dec1.current_phase == SignalPhase.PEDESTRIAN_CLEARANCE
    assert dec1.decision_case == "CASE_F"

    # Frame 2 clear at t=28.6
    dec2 = fsm.step(clear_snap, timestamp=28.6)
    assert dec2.current_phase == SignalPhase.PEDESTRIAN_CLEARANCE

    # Frame 3 clear at t=28.7 -> Now released to VEHICLE_GREEN!
    dec3 = fsm.step(clear_snap, timestamp=28.7)
    assert dec3.current_phase == SignalPhase.VEHICLE_GREEN
    assert dec3.decision_case == "CASE_F"
    assert "verified clear" in dec3.reason.lower()


def test_fail_safe_conservative_stop():
    """Fail-safe: Triggering fault stops all conflicting movements into FAIL_SAFE_STOP."""
    fsm = AdaptiveSignalController()
    fsm.trigger_fault("Camera stream dropout / occlusion", timestamp=5.0)

    snap = _mock_snapshot()
    dec = fsm.step(snap, timestamp=6.0)

    assert dec.current_phase == SignalPhase.FAIL_SAFE_STOP
    assert dec.decision_case == "FAIL_SAFE"
    assert "critical safety fault" in dec.reason.lower()
