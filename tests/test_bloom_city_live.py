"""Unit and integration tests for Bloom City Live Vision State Adapter (Step 25 / Live AI Control).

Verifies that authoritative PipelineFrameResult objects drive Bloom City signals,
waiting/crossing pedestrians, vehicle queuing/advancement, and persistent batch progression.

IMPORTANT: Does NOT import streamlit or app at module top level to preserve
the 'assert streamlit not in sys.modules' test constraint.
"""
from __future__ import annotations

import json
import re
import pytest

from core.adaptive_controller import AdaptiveController
from core.models import (
    ControllerState,
    Decision,
    DemandSnapshot,
    PedestrianSignal,
    Phase,
    VehicleSignal,
)
from core.perception_adapter import SpatialDetection
from core.pipeline import PipelineFrameResult
from core.spatial_analysis import Approach, SpatialZone, ZoneType
from core.telemetry import TelemetryAggregator
from simulation import geometry as G
from simulation.controller_bridge import BloomCitySignalState, map_controller_state
from simulation.live_adapter import (
    APPROACH_TO_INDEX,
    BloomCityLiveAdapter,
)
from simulation.simulation_agents import (
    PedestrianState,
    VehicleKind,
    VehicleState,
)


def _make_frame_result(
    frame_index: int,
    timestamp: float,
    dt: float = 0.04,
    current_phase: Phase = Phase.VEHICLE_GREEN,
    vehicle_signal: VehicleSignal = VehicleSignal.GREEN,
    pedestrian_signal: PedestrianSignal = PedestrianSignal.DONT_WALK,
    decision: Decision = Decision.VEHICLE_PRIORITY,
    decision_reason: str = "Test reason",
    waiting_pedestrians: int = 0,
    vehicle_demand: int = 1,
    spatial_detections: list = None,
) -> PipelineFrameResult:
    """Helper to construct a deterministic, valid PipelineFrameResult."""
    if spatial_detections is None:
        spatial_detections = []

    c_state = ControllerState(
        current_phase=current_phase,
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
    sig = map_controller_state(c_state)
    dem = DemandSnapshot(
        pedestrians_waiting=waiting_pedestrians,
        longest_pedestrian_wait_s=0.0,
        pedestrians_in_crosswalk=0,
        vehicle_demand=vehicle_demand,
        vehicle_queue=0,
        source="test",
    )
    tel = TelemetryAggregator().aggregate(spatial_detections)

    return PipelineFrameResult(
        frame_index=frame_index,
        timestamp=timestamp,
        dt=dt,
        tracked_detections=[],
        spatial_detections=spatial_detections,
        telemetry_snapshot=tel,
        demand_snapshot=dem,
        controller_state=c_state,
        signal_state=sig,
    )


def _get_html_payload(html: str) -> dict:
    """Extract embedded JSON payload from rendered Bloom City HTML."""
    match = re.search(r'<script id="bcdata" type="application/json">(.*?)</script>', html, re.DOTALL)
    assert match is not None, "Could not find bcdata script tag in HTML"
    return json.loads(match.group(1).replace(r"<\/", "</"))


# -----------------------------------------------------------------------------
# Test 1: Initial state
# -----------------------------------------------------------------------------
def test_bloom_city_live_adapter_initial_state():
    adapter = BloomCityLiveAdapter(seed=7)
    assert adapter.time == 0.0
    assert len(adapter.vehicles) == 2  # North and South corridor cars
    assert len(adapter.pedestrians) == 0
    assert adapter.processed_frames_count == 0
    assert adapter.last_frame_index == -1
    assert len(adapter.recorded_frames) == 0


# -----------------------------------------------------------------------------
# Test 2: Spawning waiting pedestrian from EAST_WAITING spatial detection
# -----------------------------------------------------------------------------
def test_waiting_pedestrian_spawned_from_spatial_detection():
    adapter = BloomCityLiveAdapter(seed=7)

    det_east = SpatialDetection(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=(500, 200, 520, 260),
        center=(510, 230),
        bottom_center=(510, 260),
        spatial_point=(620, 266),
        zone=SpatialZone.EAST_WAITING,
        approach=Approach.EAST,
        zone_type=ZoneType.WAITING,
        frame_index=10,
    )

    res = _make_frame_result(
        frame_index=10,
        timestamp=0.4,
        current_phase=Phase.VEHICLE_GREEN,
        vehicle_signal=VehicleSignal.GREEN,
        pedestrian_signal=PedestrianSignal.DONT_WALK,
        waiting_pedestrians=1,
        spatial_detections=[det_east],
    )

    adapter.process_frame_result(res, video_name="intersection_50s.mp4")

    assert adapter.waiting_pedestrians_count == 1
    assert adapter.crossing_pedestrians_count == 0
    ped = adapter.pedestrians[0]
    assert ped.crosswalk == 1  # East crosswalk
    assert ped.state == PedestrianState.WAITING
    assert ped.progress == 0.0


# -----------------------------------------------------------------------------
# Test 3: PEDESTRIAN_WALK triggers transition to CROSSING
# -----------------------------------------------------------------------------
def test_pedestrian_walk_triggers_crossing():
    adapter = BloomCityLiveAdapter(seed=7)

    det_east = SpatialDetection(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=(500, 200, 520, 260),
        center=(510, 230),
        bottom_center=(510, 260),
        spatial_point=(620, 266),
        zone=SpatialZone.EAST_WAITING,
        approach=Approach.EAST,
        zone_type=ZoneType.WAITING,
        frame_index=0,
    )

    # Frame 0: Green, pedestrian waiting
    res0 = _make_frame_result(
        frame_index=0,
        timestamp=0.0,
        current_phase=Phase.VEHICLE_GREEN,
        vehicle_signal=VehicleSignal.GREEN,
        pedestrian_signal=PedestrianSignal.DONT_WALK,
        waiting_pedestrians=1,
        spatial_detections=[det_east],
    )
    adapter.process_frame_result(res0)
    assert adapter.waiting_pedestrians_count == 1
    assert adapter.crossing_pedestrians_count == 0

    # Frame 1: Walk granted (Vehicle RED, Pedestrian WALK)
    res1 = _make_frame_result(
        frame_index=1,
        timestamp=0.1,
        dt=0.1,
        current_phase=Phase.PEDESTRIAN_WALK,
        vehicle_signal=VehicleSignal.RED,
        pedestrian_signal=PedestrianSignal.WALK,
        decision=Decision.PEDESTRIAN_PRIORITY,
        decision_reason="Walk phase active",
        waiting_pedestrians=1,
        spatial_detections=[det_east],
    )
    adapter.process_frame_result(res1)

    assert adapter.waiting_pedestrians_count == 0
    assert adapter.crossing_pedestrians_count == 1
    ped = adapter.pedestrians[0]
    assert ped.state == PedestrianState.CROSSING
    assert ped.progress > 0.0


# -----------------------------------------------------------------------------
# Test 4: Pedestrian advances and completes crossing
# -----------------------------------------------------------------------------
def test_pedestrian_crossing_progression_and_completion():
    adapter = BloomCityLiveAdapter(seed=7)

    det_east = SpatialDetection(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=(500, 200, 520, 260),
        center=(510, 230),
        bottom_center=(510, 260),
        spatial_point=(620, 266),
        zone=SpatialZone.EAST_WAITING,
        approach=Approach.EAST,
        zone_type=ZoneType.WAITING,
        frame_index=0,
    )

    # Spawn waiting
    res0 = _make_frame_result(
        frame_index=0,
        timestamp=0.0,
        current_phase=Phase.VEHICLE_GREEN,
        vehicle_signal=VehicleSignal.GREEN,
        pedestrian_signal=PedestrianSignal.DONT_WALK,
        waiting_pedestrians=1,
        spatial_detections=[det_east],
    )
    adapter.process_frame_result(res0)

    # Step through WALK phase (duration ~7s, crossing length 168px at 26px/s = ~6.46s)
    for i in range(1, 80):
        t = i * 0.1
        phase = Phase.PEDESTRIAN_WALK if t < 5.0 else Phase.PEDESTRIAN_CLEARANCE
        p_sig = PedestrianSignal.WALK if t < 5.0 else PedestrianSignal.CLEARING
        res = _make_frame_result(
            frame_index=i,
            timestamp=t,
            dt=0.1,
            current_phase=phase,
            vehicle_signal=VehicleSignal.RED,
            pedestrian_signal=p_sig,
            decision=Decision.PEDESTRIAN_PRIORITY,
            decision_reason="Pedestrian crossing",
            waiting_pedestrians=0,
        )
        adapter.process_frame_result(res)

    # After 8.0s, pedestrian has crossed (progress >= 1.0) and departed
    assert adapter.crossing_pedestrians_count == 0


# -----------------------------------------------------------------------------
# Test 5: Vehicle deceleration on yellow and red
# -----------------------------------------------------------------------------
def test_vehicle_deceleration_on_yellow_and_red():
    adapter = BloomCityLiveAdapter(seed=7)
    v = adapter.vehicles[0]
    line = G.STOP_LINE_S[v.approach]

    # Run yellow for 2.0s
    for i in range(40):
        t = i * 0.05
        res = _make_frame_result(
            frame_index=i,
            timestamp=t,
            dt=0.05,
            current_phase=Phase.VEHICLE_YELLOW,
            vehicle_signal=VehicleSignal.YELLOW,
            pedestrian_signal=PedestrianSignal.DONT_WALK,
            decision=Decision.VEHICLE_PRIORITY,
            decision_reason="Yellow clearance",
            vehicle_demand=1,
        )
        adapter.process_frame_result(res)

    # Vehicle must have stopped cleanly before the stop line
    assert v.front <= line
    assert v.speed < 1.0
    assert v.state == VehicleState.STOPPED


# -----------------------------------------------------------------------------
# Test 6: Vehicle advancement on green
# -----------------------------------------------------------------------------
def test_vehicle_advancement_on_green():
    adapter = BloomCityLiveAdapter(seed=7)
    v = adapter.vehicles[0]
    line = G.STOP_LINE_S[v.approach]
    init_front = v.front

    for i in range(60):
        t = i * 0.05
        res = _make_frame_result(
            frame_index=i,
            timestamp=t,
            dt=0.05,
            current_phase=Phase.VEHICLE_GREEN,
            vehicle_signal=VehicleSignal.GREEN,
            pedestrian_signal=PedestrianSignal.DONT_WALK,
            decision=Decision.VEHICLE_PRIORITY,
            decision_reason="Green corridor",
            vehicle_demand=1,
        )
        adapter.process_frame_result(res)

    # Vehicle has accelerated and crossed past the stop line
    assert v.front > line
    assert v.front > init_front + 100.0


# -----------------------------------------------------------------------------
# Test 7: Vehicle yields to crossing pedestrian even on green
# -----------------------------------------------------------------------------
def test_vehicle_yields_to_crossing_pedestrian():
    adapter = BloomCityLiveAdapter(seed=7)
    v = adapter.vehicles[0]
    line = G.STOP_LINE_S[v.approach]

    # Add a pedestrian currently crossing on crosswalk 0
    p = adapter.place_pedestrian(crosswalk=0, direction=1, slot=0)
    p.state = PedestrianState.CROSSING
    p.progress = 0.4

    for i in range(40):
        t = i * 0.05
        res = _make_frame_result(
            frame_index=i,
            timestamp=t,
            dt=0.05,
            current_phase=Phase.VEHICLE_GREEN,
            vehicle_signal=VehicleSignal.GREEN,
            pedestrian_signal=PedestrianSignal.DONT_WALK,
            vehicle_demand=1,
        )
        adapter.process_frame_result(res)

    # Vehicle yields before crosswalk / stop line
    assert v.front <= line
    assert v.speed < 1.0


# -----------------------------------------------------------------------------
# Test 8: Batch processing preserves persistent state across calls
# -----------------------------------------------------------------------------
def test_batch_processing_preserves_persistent_state():
    adapter = BloomCityLiveAdapter(seed=7)

    # Batch 1: frames 0..24 (approx 1s)
    batch1 = [
        _make_frame_result(
            frame_index=i,
            timestamp=round(i * 0.04, 3),
            dt=0.04,
            current_phase=Phase.VEHICLE_GREEN,
        )
        for i in range(25)
    ]
    adapter.process_batch(batch1, video_name="clip.mp4")

    assert adapter.processed_frames_count == 25
    assert adapter.last_frame_index == 24
    assert len(adapter.recorded_frames) > 0
    t_after_b1 = adapter.time
    veh_pos_b1 = adapter.vehicles[0].s

    # Batch 2: frames 25..49 (approx 1s more)
    batch2 = [
        _make_frame_result(
            frame_index=i,
            timestamp=round(i * 0.04, 3),
            dt=0.04,
            current_phase=Phase.VEHICLE_GREEN,
        )
        for i in range(25, 50)
    ]
    adapter.process_batch(batch2, video_name="clip.mp4")

    assert adapter.processed_frames_count == 50
    assert adapter.last_frame_index == 49
    assert adapter.time > t_after_b1
    assert adapter.vehicles[0].s > veh_pos_b1

    # Verify monotonically increasing frame timestamps in recorded_frames
    for k in range(len(adapter.recorded_frames) - 1):
        assert adapter.recorded_frames[k]["t"] <= adapter.recorded_frames[k + 1]["t"]


# -----------------------------------------------------------------------------
# Test 9: Reset clears state back to beginning
# -----------------------------------------------------------------------------
def test_reset_clears_state_and_reinitializes():
    adapter = BloomCityLiveAdapter(seed=7)
    batch = [_make_frame_result(frame_index=i, timestamp=i * 0.04) for i in range(20)]
    adapter.process_batch(batch)

    assert adapter.processed_frames_count == 20
    assert len(adapter.recorded_frames) > 0

    adapter.reset()

    assert adapter.processed_frames_count == 0
    assert adapter.last_frame_index == -1
    assert len(adapter.recorded_frames) == 0
    assert len(adapter.pedestrians) == 0
    assert len(adapter.vehicles) == 2


# -----------------------------------------------------------------------------
# Test 10: Safety invariant enforcement
# -----------------------------------------------------------------------------
def test_safety_invariant_enforcement():
    adapter = BloomCityLiveAdapter(seed=7)
    # Controller state with illegal GREEN vehicle + WALK pedestrian
    illegal_sig = BloomCitySignalState(
        phase=Phase.VEHICLE_GREEN,
        vehicle_signal=VehicleSignal.GREEN,
        pedestrian_signal=PedestrianSignal.WALK,  # ILLEGAL CONFLICT!
        decision=Decision.VEHICLE_PRIORITY,
        decision_reason="Illegal test state",
        phase_index=0,
        vehicle_signal_index=0,
        pedestrian_signal_index=1,
        decision_index=0,
        time_remaining=10.0,
        fail_safe_active=False,
        vehicle_lens_color="#38d889",
        pedestrian_lens_color="#38d889",
        pedestrian_glyph="🚶",
    )
    assert not illegal_sig.is_safe

    res = PipelineFrameResult(
        frame_index=0,
        timestamp=0.0,
        dt=0.04,
        tracked_detections=[],
        spatial_detections=[],
        telemetry_snapshot=TelemetryAggregator().aggregate([]),
        demand_snapshot=DemandSnapshot(),
        controller_state=ControllerState(
            current_phase=Phase.VEHICLE_GREEN,
            vehicle_signal=VehicleSignal.GREEN,
            pedestrian_signal=PedestrianSignal.WALK,
            decision=Decision.VEHICLE_PRIORITY,
            decision_reason="Conflict",
            phase_elapsed=0.0,
            phase_target=10.0,
            time_remaining=10.0,
            fail_safe_active=False,
            total_time=0.0,
            cycles_completed=0,
            pedestrian_phases_served=0,
        ),
        signal_state=illegal_sig,
    )

    with pytest.raises(AssertionError, match="Safety invariant violation"):
        adapter.process_frame_result(res)


# -----------------------------------------------------------------------------
# Test 11: HTML Rendering for Live AI Control
# -----------------------------------------------------------------------------
def test_html_rendering_live_ai_control():
    adapter = BloomCityLiveAdapter(seed=7)
    det = SpatialDetection(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(100, 100, 120, 150),
        center=(110, 125),
        bottom_center=(110, 150),
        spatial_point=(620, 266),
        zone=SpatialZone.EAST_WAITING,
        approach=Approach.EAST,
        zone_type=ZoneType.WAITING,
        frame_index=10,
    )
    res = _make_frame_result(
        frame_index=10,
        timestamp=0.5,
        current_phase=Phase.VEHICLE_GREEN,
        waiting_pedestrians=1,
        spatial_detections=[det],
    )
    adapter.process_frame_result(res, video_name="intersection_50s.mp4", force_record=True)

    html = adapter.render_html(theme="day")
    assert "LIVE AI CONTROL: Driven by video telemetry" in html
    assert "intersection_50s.mp4" in html

    payload = _get_html_payload(html)
    assert len(payload["frames"]) >= 1
    f0 = payload["frames"][0]
    assert "ph" in f0
    assert "vs" in f0
    assert "ps" in f0
    assert "v" in f0
    assert "p" in f0


# -----------------------------------------------------------------------------
# Test 12: Single-frame stepping safeguard prevents JS division by zero
# -----------------------------------------------------------------------------
def test_single_frame_stepping_safeguard():
    adapter = BloomCityLiveAdapter(seed=7)
    res = _make_frame_result(frame_index=0, timestamp=0.0)
    adapter.process_frame_result(res, force_record=True)

    assert len(adapter.recorded_frames) == 1
    render_frames = adapter.get_render_frames()
    assert len(render_frames) == 2
    # Second frame has t > 0 so T > 0 and prevents JS NaN
    assert render_frames[1]["t"] > render_frames[0]["t"]


# -----------------------------------------------------------------------------
# Test 13: Full 50s Video FSM Trajectory Simulation Benchmark
# -----------------------------------------------------------------------------
def test_50s_video_simulation_transitions():
    """Simulate the authoritative FSM transitions from tools/intersection_50s.mp4:
    - 0.0s .. 43.25s: VEHICLE_GREEN with EAST_WAITING pedestrian.
    - 43.25s: VEHICLE_YELLOW (approaching vehicles brake to 0).
    - 46.75s: ALL_RED_CLEARANCE.
    - 49.76s: PEDESTRIAN_WALK (pedestrian signal WALK, pedestrian crosses).
    """
    adapter = BloomCityLiveAdapter(seed=7)

    det_east = SpatialDetection(
        track_id=42,
        class_id=0,
        class_name="person",
        confidence=0.91,
        bbox=(500, 200, 520, 260),
        center=(510, 230),
        bottom_center=(510, 260),
        spatial_point=(620, 266),
        zone=SpatialZone.EAST_WAITING,
        approach=Approach.EAST,
        zone_type=ZoneType.WAITING,
        frame_index=0,
    )

    # 1. During VEHICLE_GREEN (0s - 40s)
    for frame_idx, t in [(0, 0.0), (500, 20.8), (1000, 41.7)]:
        res = _make_frame_result(
            frame_index=frame_idx,
            timestamp=t,
            dt=0.04,
            current_phase=Phase.VEHICLE_GREEN,
            vehicle_signal=VehicleSignal.GREEN,
            pedestrian_signal=PedestrianSignal.DONT_WALK,
            waiting_pedestrians=1,
            spatial_detections=[det_east],
        )
        adapter.process_frame_result(res)

    # Pedestrian is waiting at East crosswalk
    assert adapter.waiting_pedestrians_count == 1
    assert adapter.crossing_pedestrians_count == 0
    assert adapter.pedestrians[0].crosswalk == 1
    assert adapter.vehicle_signal == VehicleSignal.GREEN

    # 2. At 43.25s: VEHICLE_YELLOW
    res_yellow = _make_frame_result(
        frame_index=1037,
        timestamp=43.25,
        dt=0.04,
        current_phase=Phase.VEHICLE_YELLOW,
        vehicle_signal=VehicleSignal.YELLOW,
        pedestrian_signal=PedestrianSignal.DONT_WALK,
        decision_reason="1 pedestrian waiting; maximum vehicle green (40s) reached.",
        waiting_pedestrians=1,
        spatial_detections=[det_east],
    )
    adapter.process_frame_result(res_yellow)
    assert adapter.vehicle_signal == VehicleSignal.YELLOW
    assert adapter.waiting_pedestrians_count == 1

    # 3. At 46.75s: ALL_RED_CLEARANCE
    res_all_red = _make_frame_result(
        frame_index=1121,
        timestamp=46.75,
        dt=0.04,
        current_phase=Phase.ALL_RED_CLEARANCE,
        vehicle_signal=VehicleSignal.RED,
        pedestrian_signal=PedestrianSignal.DONT_WALK,
        waiting_pedestrians=1,
        spatial_detections=[det_east],
    )
    adapter.process_frame_result(res_all_red)
    assert adapter.vehicle_signal == VehicleSignal.RED
    assert adapter.pedestrian_signal == PedestrianSignal.DONT_WALK
    assert adapter.waiting_pedestrians_count == 1

    # 4. At 49.76s: PEDESTRIAN_WALK
    res_walk = _make_frame_result(
        frame_index=1193,
        timestamp=49.76,
        dt=0.04,
        current_phase=Phase.PEDESTRIAN_WALK,
        vehicle_signal=VehicleSignal.RED,
        pedestrian_signal=PedestrianSignal.WALK,
        decision=Decision.PEDESTRIAN_PRIORITY,
        decision_reason="Pedestrian walk phase active.",
        waiting_pedestrians=1,
        spatial_detections=[det_east],
    )
    adapter.process_frame_result(res_walk)

    # Verified: vehicles remain stopped at RED, waiting pedestrian transitions to CROSSING
    assert adapter.vehicle_signal == VehicleSignal.RED
    assert adapter.pedestrian_signal == PedestrianSignal.WALK
    assert adapter.waiting_pedestrians_count == 0
    assert adapter.crossing_pedestrians_count == 1
    assert adapter.pedestrians[0].state == PedestrianState.CROSSING
    assert adapter.pedestrians[0].crosswalk == 1
    assert adapter.pedestrians[0].progress > 0.0

    # Safety invariant holds
    assert adapter.is_safe

