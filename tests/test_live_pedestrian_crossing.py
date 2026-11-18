"""Regression test suite for Live Vision -> Adaptive FSM -> Bloom City synchronization.

Covers the 10 critical integration invariants:
1. Pedestrian track persistence
2. Pedestrian WAITING -> CROSSING transition
3. Crossing agent continues across multiple simulation frames
4. Pedestrian is not removed immediately after one missed detection (grace period)
5. Vehicle release is blocked while protected crosswalk is occupied
6. PEDESTRIAN_WALK maps to vehicle RED + pedestrian WALK
7. PEDESTRIAN_CLEARANCE maps correctly
8. Source video ending does not immediately terminate active simulation episode
9. Bloom City signal state always equals authoritative FSM state
10. No conflicting GREEN + WALK state
"""
from typing import List, Optional

import pytest

from core.adaptive_controller import AdaptiveController
from core.models import (
    PHASE_ORDER,
    PHASE_OUTPUTS,
    ControllerConfig,
    ControllerState,
    Decision,
    DemandSnapshot,
    PedestrianSignal,
    Phase,
    VehicleSignal,
)
from core.pipeline import PipelineFrameResult
from core.perception_adapter import SpatialDetection
from core.spatial_analysis import Approach, SpatialZone, ZoneType
from core.telemetry import TelemetryAggregator
from simulation.controller_bridge import (
    BloomCitySignalState,
    map_controller_state,
)
from simulation.live_adapter import BloomCityLiveAdapter
from simulation.simulation_agents import PedestrianAgent, PedestrianState


def _make_det(
    track_id: int,
    zone_type: ZoneType = ZoneType.WAITING,
    approach: Approach = Approach.EAST,
    zone: SpatialZone = SpatialZone.EAST_WAITING,
    frame_index: int = 0,
) -> SpatialDetection:
    return SpatialDetection(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=0.90,
        bbox=(500, 200, 520, 260),
        center=(510, 230),
        bottom_center=(510, 260),
        spatial_point=(620, 266),
        zone=zone,
        approach=approach,
        zone_type=zone_type,
        frame_index=frame_index,
    )


def _make_result(
    frame_index: int = 0,
    timestamp: float = 0.0,
    dt: float = 0.0417,
    phase: Phase = Phase.VEHICLE_GREEN,
    vehicle_signal: VehicleSignal = VehicleSignal.GREEN,
    pedestrian_signal: PedestrianSignal = PedestrianSignal.DONT_WALK,
    decision: Decision = Decision.VEHICLE_PRIORITY,
    decision_reason: str = "Vehicles released",
    waiting: int = 0,
    crossing: int = 0,
    longest_wait: float = 0.0,
    vehicle_demand: int = 2,
    spatial_detections: Optional[List[SpatialDetection]] = None,
) -> PipelineFrameResult:
    ctrl_st = ControllerState(
        current_phase=phase,
        vehicle_signal=vehicle_signal,
        pedestrian_signal=pedestrian_signal,
        decision=decision,
        decision_reason=decision_reason,
        phase_elapsed=timestamp,
        phase_target=10.0,
        time_remaining=max(0.0, 10.0 - timestamp),
        fail_safe_active=False,
        total_time=timestamp,
        cycles_completed=0,
        pedestrian_phases_served=0,
    )
    sig_st = map_controller_state(ctrl_st)
    demand = DemandSnapshot(
        pedestrians_waiting=waiting,
        longest_pedestrian_wait_s=longest_wait,
        pedestrians_in_crosswalk=crossing,
        vehicle_demand=vehicle_demand,
        vehicle_queue=0,
        source="vision",
    )
    sp = spatial_detections or []
    tel = TelemetryAggregator().aggregate(sp)
    return PipelineFrameResult(
        frame_index=frame_index,
        timestamp=timestamp,
        dt=dt,
        tracked_detections=[],
        spatial_detections=sp,
        telemetry_snapshot=tel,
        demand_snapshot=demand,
        controller_state=ctrl_st,
        signal_state=sig_st,
    )


# -----------------------------------------------------------------------------
# Test 1: Pedestrian track persistence across frames
# -----------------------------------------------------------------------------
def test_pedestrian_track_persistence():
    adapter = BloomCityLiveAdapter(seed=42)
    det1 = _make_det(track_id=101, frame_index=0)

    res0 = _make_result(frame_index=0, timestamp=0.0, waiting=1, spatial_detections=[det1])
    adapter.process_frame_result(res0)
    assert len(adapter.pedestrians) == 1
    ped_id = adapter.pedestrians[0].id
    assert adapter.pedestrians[0].track_id == 101

    # Frame 1: track continues
    det2 = _make_det(track_id=101, frame_index=1)
    res1 = _make_result(frame_index=1, timestamp=0.1, dt=0.1, waiting=1, spatial_detections=[det2])
    adapter.process_frame_result(res1)
    assert len(adapter.pedestrians) == 1
    # Identity is preserved; no spurious new agent created
    assert adapter.pedestrians[0].id == ped_id
    assert adapter.pedestrians[0].track_id == 101


# -----------------------------------------------------------------------------
# Test 2: Pedestrian WAITING -> CROSSING transition
# -----------------------------------------------------------------------------
def test_pedestrian_waiting_to_crossing_transition():
    adapter = BloomCityLiveAdapter(seed=42)
    det = _make_det(track_id=201)

    # Frame 0: WAITING under VEHICLE_GREEN
    res0 = _make_result(frame_index=0, timestamp=0.0, waiting=1, spatial_detections=[det])
    adapter.process_frame_result(res0)
    assert adapter.pedestrians[0].state == PedestrianState.WAITING

    # Frame 1: PEDESTRIAN_WALK active with Vehicle RED
    res1 = _make_result(
        frame_index=1,
        timestamp=0.1,
        dt=0.1,
        phase=Phase.PEDESTRIAN_WALK,
        vehicle_signal=VehicleSignal.RED,
        pedestrian_signal=PedestrianSignal.WALK,
        waiting=1,
        spatial_detections=[det],
    )
    adapter.process_frame_result(res1)
    assert adapter.pedestrians[0].state == PedestrianState.CROSSING
    assert adapter.pedestrians[0].progress > 0.0


# -----------------------------------------------------------------------------
# Test 3: Crossing agent continues across multiple simulation frames
# -----------------------------------------------------------------------------
def test_crossing_agent_continues_across_multiple_simulation_frames():
    adapter = BloomCityLiveAdapter(seed=42)
    det = _make_det(track_id=301)

    # Transition to CROSSING
    res0 = _make_result(
        frame_index=0,
        timestamp=0.0,
        phase=Phase.PEDESTRIAN_WALK,
        vehicle_signal=VehicleSignal.RED,
        pedestrian_signal=PedestrianSignal.WALK,
        waiting=1,
        spatial_detections=[det],
    )
    adapter.process_frame_result(res0)
    p = adapter.pedestrians[0]
    assert p.state == PedestrianState.CROSSING
    p_id = p.id
    prev_progress = p.progress

    # Step through 5 subsequent frames
    for i in range(1, 6):
        res_i = _make_result(
            frame_index=i,
            timestamp=0.1 * i,
            dt=0.1,
            phase=Phase.PEDESTRIAN_WALK,
            vehicle_signal=VehicleSignal.RED,
            pedestrian_signal=PedestrianSignal.WALK,
            waiting=0,
            crossing=1,
            spatial_detections=[_make_det(track_id=301, zone_type=ZoneType.CROSSING, frame_index=i)],
        )
        adapter.process_frame_result(res_i)
        assert len(adapter.pedestrians) == 1
        curr = adapter.pedestrians[0]
        assert curr.id == p_id
        assert curr.progress > prev_progress
        prev_progress = curr.progress


# -----------------------------------------------------------------------------
# Test 4: Pedestrian not removed immediately after one missed detection
# -----------------------------------------------------------------------------
def test_pedestrian_not_removed_immediately_after_missed_detection():
    adapter = BloomCityLiveAdapter(seed=42)
    det = _make_det(track_id=401)

    res0 = _make_result(frame_index=0, timestamp=0.0, waiting=1, spatial_detections=[det])
    adapter.process_frame_result(res0)
    assert len(adapter.pedestrians) == 1
    p_id = adapter.pedestrians[0].id

    # Frame 1: detection temporarily missing (e.g. YOLO missed for 1 frame)
    res1 = _make_result(frame_index=1, timestamp=0.1, dt=0.1, waiting=0, spatial_detections=[])
    adapter.process_frame_result(res1)

    # Representative pedestrian persists within grace period
    assert len(adapter.pedestrians) == 1
    assert adapter.pedestrians[0].id == p_id
    assert adapter.pedestrians[0].state == PedestrianState.WAITING


# -----------------------------------------------------------------------------
# Test 5: Vehicle release is blocked while protected crosswalk is occupied
# -----------------------------------------------------------------------------
def test_vehicle_release_blocked_while_crosswalk_occupied():
    config = ControllerConfig(
        pedestrian_clearance_time=4.0,
        max_clearance_extension=6.0,
    )
    ctrl = AdaptiveController(config=config, initial_phase=Phase.PEDESTRIAN_CLEARANCE)

    # Elapsed reaches base clearance time (4.0s), but pedestrian is still in crosswalk
    demand_occupied = DemandSnapshot(
        pedestrians_waiting=0,
        pedestrians_in_crosswalk=1,
        vehicle_demand=2,
        longest_pedestrian_wait_s=0.0,
    )

    # Step at 4.0s
    ctrl.step(4.0, demand_occupied)
    # Clearance must be extended; phase must NOT jump to VEHICLE_GREEN
    assert ctrl.state.current_phase == Phase.PEDESTRIAN_CLEARANCE
    assert ctrl.state.vehicle_signal == VehicleSignal.RED
    assert "extending clearance" in ctrl.state.decision_reason.lower()


# -----------------------------------------------------------------------------
# Test 6: PEDESTRIAN_WALK maps to vehicle RED + pedestrian WALK
# -----------------------------------------------------------------------------
def test_pedestrian_walk_maps_to_vehicle_red_and_ped_walk():
    ctrl_st = ControllerState(
        current_phase=Phase.PEDESTRIAN_WALK,
        vehicle_signal=VehicleSignal.RED,
        pedestrian_signal=PedestrianSignal.WALK,
        decision=Decision.PEDESTRIAN_PRIORITY,
        decision_reason="Walk active",
        phase_elapsed=1.0,
        phase_target=8.0,
        time_remaining=7.0,
        fail_safe_active=False,
        total_time=1.0,
        cycles_completed=0,
        pedestrian_phases_served=1,
    )
    sig = map_controller_state(ctrl_st)
    assert sig.phase == Phase.PEDESTRIAN_WALK
    assert sig.vehicle_signal == VehicleSignal.RED
    assert sig.pedestrian_signal == PedestrianSignal.WALK
    assert sig.is_safe is True


# -----------------------------------------------------------------------------
# Test 7: PEDESTRIAN_CLEARANCE maps correctly
# -----------------------------------------------------------------------------
def test_pedestrian_clearance_maps_correctly():
    ctrl_st = ControllerState(
        current_phase=Phase.PEDESTRIAN_CLEARANCE,
        vehicle_signal=VehicleSignal.RED,
        pedestrian_signal=PedestrianSignal.CLEARING,
        decision=Decision.PEDESTRIAN_PRIORITY,
        decision_reason="Clearance active",
        phase_elapsed=2.0,
        phase_target=8.0,
        time_remaining=6.0,
        fail_safe_active=False,
        total_time=10.0,
        cycles_completed=0,
        pedestrian_phases_served=1,
    )
    sig = map_controller_state(ctrl_st)
    assert sig.phase == Phase.PEDESTRIAN_CLEARANCE
    assert sig.vehicle_signal == VehicleSignal.RED
    assert sig.pedestrian_signal == PedestrianSignal.CLEARING
    assert sig.is_safe is True


# -----------------------------------------------------------------------------
# Test 8: Source video ending does not immediately terminate simulation episode
# -----------------------------------------------------------------------------
def test_source_video_ending_does_not_terminate_simulation_episode():
    adapter = BloomCityLiveAdapter(seed=42)
    det = _make_det(track_id=801)

    # Process 10 seconds of video where pedestrians are waiting
    for f in range(240):
        t = f / 24.0
        res = _make_result(
            frame_index=f,
            timestamp=t,
            dt=1.0 / 24.0,
            phase=Phase.VEHICLE_GREEN,
            vehicle_signal=VehicleSignal.GREEN,
            pedestrian_signal=PedestrianSignal.DONT_WALK,
            waiting=2,
            longest_wait=t,
            spatial_detections=[det],
        )
        adapter.process_frame_result(res)

    assert adapter.should_continue_episode() is True
    initial_recorded = len(adapter.recorded_frames)

    # Continue the episode
    cont_frames = adapter.continue_simulation_episode(max_duration=35.0, dt=0.05)
    assert len(cont_frames) > 0
    assert len(adapter.recorded_frames) > initial_recorded

    # Verify that the episode transitioned through pedestrian phases and returned to green
    phases_in_continuation = {f["ph"] for f in cont_frames}
    walk_idx = PHASE_ORDER.index(Phase.PEDESTRIAN_WALK)
    clearance_idx = PHASE_ORDER.index(Phase.PEDESTRIAN_CLEARANCE)
    green_idx = PHASE_ORDER.index(Phase.VEHICLE_GREEN)

    assert walk_idx in phases_in_continuation
    assert clearance_idx in phases_in_continuation
    assert adapter.current_phase == Phase.VEHICLE_GREEN


# -----------------------------------------------------------------------------
# Test 9: Bloom City signal state always equals authoritative FSM state
# -----------------------------------------------------------------------------
def test_bloom_city_signal_state_equals_authoritative_fsm_state():
    adapter = BloomCityLiveAdapter(seed=42)
    det = _make_det(track_id=901)

    for phase in (Phase.VEHICLE_GREEN, Phase.VEHICLE_YELLOW, Phase.ALL_RED_CLEARANCE,
                  Phase.PEDESTRIAN_WALK, Phase.PEDESTRIAN_CLEARANCE):
        v_sig, p_sig = PHASE_OUTPUTS[phase]
        res = _make_result(
            frame_index=PHASE_ORDER.index(phase),
            timestamp=float(PHASE_ORDER.index(phase)),
            dt=0.1,
            phase=phase,
            vehicle_signal=v_sig,
            pedestrian_signal=p_sig,
            waiting=1,
            spatial_detections=[det],
        )
        frame_dict = adapter.process_frame_result(res, force_record=True)
        assert frame_dict["ph"] == PHASE_ORDER.index(phase)
        assert frame_dict["vs"] == ("GREEN", "YELLOW", "RED").index(v_sig.value)
        assert frame_dict["ps"] == ("DONT_WALK", "WALK", "CLEARING").index(p_sig.value)


# -----------------------------------------------------------------------------
# Test 10: No conflicting GREEN + WALK state
# -----------------------------------------------------------------------------
def test_no_conflicting_green_and_walk_state():
    adapter = BloomCityLiveAdapter(seed=42)

    # Step through all phases in PHASE_ORDER
    for phase in PHASE_ORDER:
        v_sig, p_sig = PHASE_OUTPUTS[phase]
        assert not (v_sig == VehicleSignal.GREEN and p_sig == PedestrianSignal.WALK)
        assert not (v_sig == VehicleSignal.YELLOW and p_sig == PedestrianSignal.WALK)

        res = _make_result(
            frame_index=PHASE_ORDER.index(phase),
            timestamp=float(PHASE_ORDER.index(phase)),
            dt=0.1,
            phase=phase,
            vehicle_signal=v_sig,
            pedestrian_signal=p_sig,
        )
        adapter.process_frame_result(res, force_record=True)
        assert adapter.is_safe is True
