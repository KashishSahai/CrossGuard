"""Focused regression tests for CrossGuard Live Vision -> Telemetry -> FSM -> Bloom City synchronization.

Covers the 11 verification points specified in Section 15:
1. Pedestrian bottom-center classification (ground-contact point vs bbox center)
2. Pedestrian homography transformation on tools/intersection_50s.mp4
3. Waiting pedestrian -> TelemetrySnapshot.waiting_pedestrians count
4. Crossing pedestrian -> TelemetrySnapshot.crossing_pedestrians count
5. Telemetry demand snapshot -> AdaptiveController.step() phase transitions
6. PipelineFrameResult type identity check (_is_pipeline_frame_result) across module reloads
7. Frame index synchronization between annotated frame and telemetry
8. Bloom City signal state equality with PipelineFrameResult.signal_state
9. Bloom City live demand synchronization with PipelineFrameResult.demand_snapshot
10. Reset clears live state and returns Bloom City to Scenario Demo
11. Scenario Demo mode remains fully functional before live vision starts
"""
from __future__ import annotations

import json
import re
from unittest.mock import MagicMock
import numpy as np
import pytest

from core.adaptive_controller import AdaptiveController
from core.camera_calibration import CameraCalibration
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
from core.perception_adapter import PerceptionAdapter, SpatialDetection
from core.pipeline import PipelineFrameResult
from core.spatial_analysis import Approach, SpatialAnalyzer, SpatialZone, ZoneType
from core.telemetry import ApproachDemandSnapshot, TelemetryAggregator, TelemetrySnapshot
from simulation.controller_bridge import (
    BloomCitySignalState,
    ControllerBridge,
    map_controller_state,
)
from simulation.live_adapter import BloomCityLiveAdapter, _is_pipeline_frame_result
from tests.live_asserts import assert_html_latest_frame_is
from vision.models import TrackedDetection


def _get_html_payload(html: str) -> dict:
    """Extract embedded JSON payload from rendered Bloom City HTML."""
    match = re.search(
        r'<script id="bcdata" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None, "Could not find bcdata script tag in HTML"
    return json.loads(match.group(1).replace(r"<\/", "</"))


def _make_dummy_controller_state(
    phase: Phase = Phase.VEHICLE_GREEN,
    decision: Decision = Decision.VEHICLE_PRIORITY,
    reason: str = "Clear flow",
    elapsed: float = 1.0,
    time_rem: float = 9.0,
) -> ControllerState:
    v_sig, p_sig = PHASE_OUTPUTS[phase]
    return ControllerState(
        current_phase=phase,
        vehicle_signal=v_sig,
        pedestrian_signal=p_sig,
        decision=decision,
        decision_reason=reason,
        phase_elapsed=elapsed,
        phase_target=elapsed + time_rem,
        time_remaining=time_rem,
        fail_safe_active=False,
        total_time=elapsed,
        cycles_completed=0,
        pedestrian_phases_served=0,
    )


def _make_dummy_demand_snapshot(
    waiting_peds: int = 0,
    crossing_peds: int = 0,
    approaching_vehs: int = 0,
    queued_vehs: int = 0,
) -> DemandSnapshot:
    return DemandSnapshot(
        pedestrians_waiting=waiting_peds,
        longest_pedestrian_wait_s=12.0 if waiting_peds > 0 else 0.0,
        pedestrians_in_crosswalk=crossing_peds,
        vehicle_demand=approaching_vehs,
        vehicle_queue=queued_vehs,
        source="vision",
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
    if spatial_detections is None:
        spatial_detections = []
    c_state = _make_dummy_controller_state(
        phase=current_phase,
        decision=decision,
        reason=decision_reason,
        elapsed=timestamp,
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
    tel = TelemetryAggregator().aggregate(spatial_detections, frame_index=frame_index)
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


# -----------------------------------------------------------------------------
# 1. Pedestrian Bottom-Center Classification
# -----------------------------------------------------------------------------
def test_pedestrian_bottom_center_classification():
    """Verify pedestrian spatial classification uses bottom-center ground contact."""
    # Person bbox from camera pixels (x1=100, y1=200, x2=150, y2=400)
    # Bbox center is (125, 300), bottom center is (125, 400)
    det = TrackedDetection(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.88,
        x1=100.0,
        y1=200.0,
        x2=150.0,
        y2=400.0,
        center_x=125.0,
        center_y=300.0,
        frame_index=0,
    )

    adapter = PerceptionAdapter(use_bottom_center=True)
    results = adapter.adapt([det])
    assert len(results) == 1
    sp = results[0]

    # Verify camera_bottom_center is populated exactly
    assert sp.camera_bottom_center == (125.0, 400.0)
    assert sp.camera_bbox == (100.0, 200.0, 150.0, 400.0)

    # When use_bottom_center=True, the classified point in canonical space
    # corresponds to bottom_center transformed, NOT center
    expected_x, expected_y = adapter.transformer.transform_point((125.0, 400.0))
    assert pytest.approx(sp.spatial_point[0], abs=1e-3) == expected_x
    assert pytest.approx(sp.spatial_point[1], abs=1e-3) == expected_y


# -----------------------------------------------------------------------------
# 2. Pedestrian Homography Transformation on tools/intersection_50s.mp4
# -----------------------------------------------------------------------------
def test_pedestrian_homography_transformation_50s():
    """Verify camera homography transforms Person #4 ground-contact into EAST_WAITING."""
    calib = CameraCalibration.for_traffic_intersection_50s()
    assert calib.is_calibrated is True

    # Real coordinates of Person #4 waiting at East curb in intersection_50s.mp4
    cam_x = 1770.6
    cam_y = 744.7

    canon_x, canon_y = calib.transform_point((cam_x, cam_y))
    assert pytest.approx(canon_x, abs=1.0) == 632.1
    assert pytest.approx(canon_y, abs=1.0) == 438.0

    analyzer = SpatialAnalyzer()
    zone = analyzer.classify((canon_x, canon_y), object_class="person")
    defn = analyzer.get_zone_definition(zone)

    assert zone == SpatialZone.EAST_WAITING
    assert defn.approach == Approach.EAST
    assert defn.zone_type == ZoneType.WAITING


# -----------------------------------------------------------------------------
# 3. Waiting Pedestrian -> Telemetry
# -----------------------------------------------------------------------------
def test_waiting_pedestrian_telemetry():
    """Verify waiting pedestrian detection propagates to TelemetrySnapshot."""
    agg = TelemetryAggregator()
    sp = SpatialDetection(
        track_id=42,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=(1700, 700, 1800, 780),
        center=(1750, 740),
        bottom_center=(1750, 780),
        spatial_point=(632.1, 438.0),
        zone=SpatialZone.EAST_WAITING,
        approach=Approach.EAST,
        zone_type=ZoneType.WAITING,
        frame_index=15,
        camera_bbox=(1700, 700, 1800, 780),
        camera_bottom_center=(1750, 780),
    )

    snap = agg.aggregate([sp], frame_index=15, longest_pedestrian_wait_s=14.5)
    assert snap.waiting_pedestrians == 1
    assert snap.crossing_pedestrians == 0
    assert snap.by_approach[Approach.EAST].waiting_pedestrians == 1
    assert snap.frame_index == 15

    # Check DemandSnapshot
    assert snap.demand.pedestrians_waiting == 1
    assert snap.demand.pedestrians_in_crosswalk == 0
    assert snap.demand.longest_pedestrian_wait_s == 14.5


# -----------------------------------------------------------------------------
# 4. Crossing Pedestrian -> Telemetry
# -----------------------------------------------------------------------------
def test_crossing_pedestrian_telemetry():
    """Verify crossing pedestrian detection propagates to TelemetrySnapshot."""
    agg = TelemetryAggregator()
    sp = SpatialDetection(
        track_id=88,
        class_id=0,
        class_name="person",
        confidence=0.85,
        bbox=(500, 600, 550, 700),
        center=(525, 650),
        bottom_center=(525, 700),
        spatial_point=(500.0, 520.0),
        zone=SpatialZone.SOUTH_CROSSING,
        approach=Approach.SOUTH,
        zone_type=ZoneType.CROSSING,
        frame_index=20,
        camera_bbox=(500, 600, 550, 700),
        camera_bottom_center=(525, 700),
    )

    snap = agg.aggregate([sp], frame_index=20)
    assert snap.waiting_pedestrians == 0
    assert snap.crossing_pedestrians == 1
    assert snap.by_approach[Approach.SOUTH].crossing_pedestrians == 1

    # Check DemandSnapshot
    assert snap.demand.pedestrians_waiting == 0
    assert snap.demand.pedestrians_in_crosswalk == 1


# -----------------------------------------------------------------------------
# 5. Telemetry -> Adaptive FSM
# -----------------------------------------------------------------------------
def test_telemetry_to_adaptive_fsm():
    """Verify waiting pedestrian demand triggers AdaptiveController transition."""
    config = ControllerConfig(
        min_vehicle_green=10.0,
        max_vehicle_green=40.0,
        yellow_time=3.5,
        all_red_time=2.0,
        pedestrian_walk_time=8.0,
        single_pedestrian_wait=10.0,
    )
    ctrl = AdaptiveController(config=config)
    assert ctrl.state.current_phase == Phase.VEHICLE_GREEN

    # Step for 9.0s with waiting pedestrian demand
    demand = _make_dummy_demand_snapshot(waiting_peds=1)
    s9 = ctrl.step(dt=9.0, demand=demand)
    # Min vehicle green protects vehicle flow
    assert s9.current_phase == Phase.VEHICLE_GREEN
    assert "minimum vehicle green protects" in s9.decision_reason

    # Step past min_vehicle_green (dt=1.5s -> total 10.5s >= 10.0s)
    s10 = ctrl.step(dt=1.5, demand=demand)
    # Controller transitions to VEHICLE_YELLOW with PEDESTRIAN_PRIORITY
    assert s10.current_phase == Phase.VEHICLE_YELLOW
    assert s10.decision == Decision.PEDESTRIAN_PRIORITY

    # Step through yellow (3.5s)
    s_red = ctrl.step(dt=3.6, demand=demand)
    assert s_red.current_phase == Phase.ALL_RED_CLEARANCE

    # Step through all-red (2.0s)
    s_walk = ctrl.step(dt=2.1, demand=demand)
    assert s_walk.current_phase == Phase.PEDESTRIAN_WALK
    assert s_walk.decision == Decision.PEDESTRIAN_PRIORITY


# -----------------------------------------------------------------------------
# 6. PipelineFrameResult Type Identity Check
# -----------------------------------------------------------------------------
def test_pipeline_frame_result_type_identity():
    """Verify _is_pipeline_frame_result duck-typing survives Streamlit hot reloads."""
    demand = _make_dummy_demand_snapshot()
    ctrl = _make_dummy_controller_state()
    bridge = ControllerBridge()
    sig = bridge.map_state(ctrl)

    agg = TelemetryAggregator()
    snap = agg.aggregate([])

    # 1. Canonical class
    canonical_res = PipelineFrameResult(
        frame_index=1,
        timestamp=0.1,
        dt=0.1,
        tracked_detections=[],
        spatial_detections=[],
        telemetry_snapshot=snap,
        demand_snapshot=demand,
        controller_state=ctrl,
        signal_state=sig,
    )
    assert _is_pipeline_frame_result(canonical_res) is True

    # 2. Simulated reloaded class from another module/memory address
    class MockReloadedPipelineFrameResult:
        def __init__(self):
            self.frame_index = 1
            self.controller_state = ctrl
            self.demand_snapshot = demand

    reloaded_res = MockReloadedPipelineFrameResult()
    # Class name is MockReloadedPipelineFrameResult, so fails
    assert _is_pipeline_frame_result(reloaded_res) is False

    # Simulate class having the exact name 'PipelineFrameResult'
    MockReloadedPipelineFrameResult.__name__ = "PipelineFrameResult"
    assert _is_pipeline_frame_result(reloaded_res) is True

    # 3. Invalid objects fail
    assert _is_pipeline_frame_result(None) is False
    assert _is_pipeline_frame_result({}) is False
    assert _is_pipeline_frame_result("PipelineFrameResult") is False


# -----------------------------------------------------------------------------
# 7. Annotated Frame and Telemetry Frame Index Synchronization
# -----------------------------------------------------------------------------
def test_frame_index_synchronization():
    """Verify annotated frame and telemetry have identical frame_index."""
    agg = TelemetryAggregator()
    snap = agg.aggregate([], frame_index=42)
    assert snap.frame_index == 42

    ctrl = _make_dummy_controller_state()
    sig = map_controller_state(ctrl)
    annotated = np.zeros((480, 640, 3), dtype=np.uint8)

    res = PipelineFrameResult(
        frame_index=42,
        timestamp=1.4,
        dt=0.033,
        tracked_detections=[],
        spatial_detections=[],
        telemetry_snapshot=snap,
        demand_snapshot=snap.demand,
        controller_state=ctrl,
        signal_state=sig,
        annotated_frame=annotated,
    )

    assert res.frame_index == 42
    assert res.telemetry_snapshot.frame_index == 42
    assert res.annotated_frame.shape == (480, 640, 3)
    assert res.signal_state.is_safe is True


# -----------------------------------------------------------------------------
# 8. Bloom City Signal State Equality
# -----------------------------------------------------------------------------
def test_bloom_city_signal_state_equality():
    """Verify Bloom City frames match PipelineFrameResult signal state exactly."""
    from app import sync_bloom_city_with_signal

    # Test PEDESTRIAN_WALK
    ctrl = _make_dummy_controller_state(
        phase=Phase.PEDESTRIAN_WALK,
        decision=Decision.PEDESTRIAN_PRIORITY,
        reason="Pedestrian Walk active",
    )
    sig = map_controller_state(ctrl)
    assert sig.vehicle_signal == VehicleSignal.RED
    assert sig.pedestrian_signal == PedestrianSignal.WALK
    assert sig.is_safe is True

    frames, reasons, html = sync_bloom_city_with_signal(signal_state=sig, scenario="balanced")
    assert_html_latest_frame_is(html, Phase.PEDESTRIAN_WALK)
    assert len(frames) > 0
    payload = _get_html_payload(html)
    f0 = payload["frames"][0]
    assert f0["ph"] == PHASE_ORDER.index(Phase.PEDESTRIAN_WALK)
    assert f0["vs"] == 2  # RED
    assert f0["ps"] == 1  # WALK


# -----------------------------------------------------------------------------
# 9. Bloom City Live Demand Synchronization
# -----------------------------------------------------------------------------
def test_bloom_city_live_adapter_demand_sync():
    """Verify Bloom City live adapter mirrors demand snapshot counts."""
    adapter = BloomCityLiveAdapter(seed=7)
    demand = _make_dummy_demand_snapshot(waiting_peds=2, crossing_peds=1, approaching_vehs=3, queued_vehs=1)
    ctrl = _make_dummy_controller_state(phase=Phase.VEHICLE_GREEN)
    sig = map_controller_state(ctrl)

    agg = TelemetryAggregator()
    snap = agg.aggregate([])

    res = PipelineFrameResult(
        frame_index=10,
        timestamp=1.0,
        dt=0.1,
        tracked_detections=[],
        spatial_detections=[],
        telemetry_snapshot=snap,
        demand_snapshot=demand,
        controller_state=ctrl,
        signal_state=sig,
    )

    frame_dict = adapter.process_frame_result(res)
    assert adapter.processed_frames_count == 1
    assert adapter.last_frame_index == 10
    assert adapter.demand == demand
    assert adapter.ctrl == ctrl
    assert adapter.time == 1.0
    assert frame_dict["ph"] == PHASE_ORDER.index(Phase.VEHICLE_GREEN)


# -----------------------------------------------------------------------------
# 10. Reset Clears Live State
# -----------------------------------------------------------------------------
def test_reset_clears_live_state():
    """Verify reset clears live adapter state and returns to Scenario Demo."""
    from app import sync_bloom_city_with_signal

    adapter = BloomCityLiveAdapter(seed=7)
    demand = _make_dummy_demand_snapshot(waiting_peds=1)
    ctrl = _make_dummy_controller_state()
    agg = TelemetryAggregator()
    snap = agg.aggregate([])

    res = PipelineFrameResult(
        frame_index=5,
        timestamp=0.5,
        dt=0.1,
        tracked_detections=[],
        spatial_detections=[],
        telemetry_snapshot=snap,
        demand_snapshot=demand,
        controller_state=ctrl,
        signal_state=map_controller_state(ctrl),
    )
    adapter.process_frame_result(res)
    assert adapter.processed_frames_count == 1

    # Reset adapter
    adapter.reset()
    assert adapter.processed_frames_count == 0
    assert adapter.last_frame_index == -1
    assert adapter.time == 0.0

    # Calling sync with None returns SCENARIO DEMO
    frames, reasons, html = sync_bloom_city_with_signal(signal_state=None)
    assert "SCENARIO DEMO" in html
    assert "LIVE CONTROLLER" not in html
    assert "Representative agents derived from scenario telemetry" in html


# -----------------------------------------------------------------------------
# 11. Scenario Demo Mode Functional
# -----------------------------------------------------------------------------
def test_scenario_demo_mode_functional():
    """Verify Scenario Demo mode remains fully operational before Live Vision."""
    from app import sync_bloom_city_with_signal
    from simulation.bloom_city_engine import SCENARIOS, build_frames

    for sc_name in ("balanced", "ped_surge", "heavy_traffic", "quiet"):
        assert sc_name in SCENARIOS
        frames, reasons, _ = build_frames(sc_name, seconds=1.0, seed=7)
        assert len(frames) > 0
        assert len(reasons) > 0

        frames, reasons, html = sync_bloom_city_with_signal(None, scenario=sc_name)
        assert "SCENARIO DEMO" in html
        assert SCENARIOS[sc_name].title in html


# -----------------------------------------------------------------------------
# 12. Strict Frame-by-Frame Synchronization Pipeline to Bloom City
# -----------------------------------------------------------------------------
def test_strict_frame_by_frame_synchronization_pipeline_to_bloom_city():
    """Verify sequential frame indices produce exact lockstep with Bloom City payload."""
    from app import sync_bloom_city_with_signal

    agg = TelemetryAggregator()

    sequence = [
        (10, Phase.VEHICLE_GREEN, Decision.VEHICLE_PRIORITY, "Traffic flowing normally", 1, 0, 4, 1),
        (20, Phase.VEHICLE_YELLOW, Decision.PEDESTRIAN_PRIORITY, "Transition to pedestrian walk", 2, 0, 2, 2),
        (30, Phase.PEDESTRIAN_WALK, Decision.PEDESTRIAN_PRIORITY, "Pedestrian walk active", 0, 3, 1, 3),
    ]

    for frame_idx, phase, decision, reason, wait_peds, cross_peds, app_vehs, q_vehs in sequence:
        ctrl = _make_dummy_controller_state(
            phase=phase,
            decision=decision,
            reason=reason,
            elapsed=float(frame_idx) * 0.1,
            time_rem=5.0,
        )
        sig = map_controller_state(ctrl)
        demand = _make_dummy_demand_snapshot(
            waiting_peds=wait_peds,
            crossing_peds=cross_peds,
            approaching_vehs=app_vehs,
            queued_vehs=q_vehs,
        )
        snap = agg.aggregate([], frame_index=frame_idx)

        res = PipelineFrameResult(
            frame_index=frame_idx,
            timestamp=float(frame_idx) * 0.033,
            dt=0.033,
            tracked_detections=[],
            spatial_detections=[],
            telemetry_snapshot=snap,
            demand_snapshot=demand,
            controller_state=ctrl,
            signal_state=sig,
        )

        assert res.frame_index == frame_idx
        assert res.telemetry_snapshot.frame_index == frame_idx
        assert res.signal_state.phase == phase
        assert res.controller_state.current_phase == phase

        frames, reasons, html = sync_bloom_city_with_signal(
            signal_state=res.signal_state,
            demand=res.demand_snapshot,
            scenario="balanced",
        )

        assert_html_latest_frame_is(html, phase)
        assert len(frames) > 0
        assert reasons[0] == reason

        payload = _get_html_payload(html)
        f0 = payload["frames"][0]

        # Phase lockstep
        assert f0["ph"] == PHASE_ORDER.index(phase)

        # Vehicle signal lockstep
        expected_vs = 0 if sig.vehicle_signal == VehicleSignal.GREEN else (1 if sig.vehicle_signal == VehicleSignal.YELLOW else 2)
        assert f0["vs"] == expected_vs

        # Pedestrian signal lockstep
        expected_ps = 1 if sig.pedestrian_signal == PedestrianSignal.WALK else (2 if sig.pedestrian_signal == PedestrianSignal.CLEARING else 0)
        assert f0["ps"] == expected_ps

        # Demand lockstep
        assert f0["nv"] == app_vehs
        assert f0["np"] == wait_peds
        assert f0["nc"] == cross_peds


# =============================================================================
# ACCEPTANCE TESTS (Separation of Modes, Live Progression, Lifecycle)
# =============================================================================

# -----------------------------------------------------------------------------
# Acceptance Test 1: Live simulation advances monotonically
# -----------------------------------------------------------------------------
def test_acceptance_1_live_simulation_advances():
    """Process several sequential live frames and verify Bloom City simulation frame advances."""
    adapter = BloomCityLiveAdapter(seed=7)
    ctrl = _make_dummy_controller_state(phase=Phase.VEHICLE_GREEN)
    sig = map_controller_state(ctrl)
    agg = TelemetryAggregator()

    initial_veh_s = adapter.vehicles[0].s
    recorded_indices = []

    for idx in range(5):
        snap = agg.aggregate([], frame_index=idx)
        demand = _make_dummy_demand_snapshot(waiting_peds=1)
        res = PipelineFrameResult(
            frame_index=idx,
            timestamp=float(idx) * 0.1,
            dt=0.1,
            tracked_detections=[],
            spatial_detections=[],
            telemetry_snapshot=snap,
            demand_snapshot=demand,
            controller_state=ctrl,
            signal_state=sig,
        )
        f_dict = adapter.process_frame_result(res, force_record=True)
        recorded_indices.append(f_dict["frame_index"])

    # Frame index increases monotonically
    assert recorded_indices == [0, 1, 2, 3, 4]
    assert adapter.processed_frames_count == 5
    assert adapter.last_frame_index == 4
    assert len(adapter.recorded_frames) == 5

    # Vehicles advanced progressively and Bloom City is not stuck at one frame
    final_veh_s = adapter.vehicles[0].s
    assert final_veh_s > initial_veh_s
    f0 = adapter.recorded_frames[0]
    f4 = adapter.recorded_frames[4]
    assert f0["v"][0][1] != f4["v"][0][1] or f0["v"][0][2] != f4["v"][0][2]


# -----------------------------------------------------------------------------
# Acceptance Test 2: Phase remains stable while simulation moves
# -----------------------------------------------------------------------------
def test_acceptance_2_phase_stable_while_simulation_moves():
    """Verify VEHICLE_GREEN remains for 8s while Bloom City representative agents continue moving."""
    adapter = BloomCityLiveAdapter(seed=7)
    initial_veh_s = adapter.vehicles[0].s
    agg = TelemetryAggregator()

    # Step for 8.0s (80 frames at dt=0.1s) under VEHICLE_GREEN
    for i in range(80):
        t = i * 0.1
        ctrl = _make_dummy_controller_state(
            phase=Phase.VEHICLE_GREEN,
            elapsed=t,
            time_rem=max(0.0, 8.0 - t),
        )
        sig = map_controller_state(ctrl)
        res = PipelineFrameResult(
            frame_index=i,
            timestamp=t,
            dt=0.1,
            tracked_detections=[],
            spatial_detections=[],
            telemetry_snapshot=agg.aggregate([], frame_index=i),
            demand_snapshot=_make_dummy_demand_snapshot(),
            controller_state=ctrl,
            signal_state=sig,
        )
        adapter.process_frame_result(res, force_record=True)

        # Every corresponding frame has GREEN vehicle signal and DONT_WALK pedestrian signal
        assert adapter.ctrl.vehicle_signal == VehicleSignal.GREEN
        assert adapter.ctrl.pedestrian_signal == PedestrianSignal.DONT_WALK

    assert adapter.processed_frames_count == 80
    assert len(adapter.recorded_frames) == 80
    # Vehicles have moved forward significantly despite phase remaining stable
    assert adapter.vehicles[0].s > initial_veh_s + 10.0


# -----------------------------------------------------------------------------
# Acceptance Test 3: Live signal transition
# -----------------------------------------------------------------------------
def test_acceptance_3_live_signal_transition():
    """Feed VEHICLE_GREEN -> VEHICLE_YELLOW -> ALL_RED_CLEARANCE -> PEDESTRIAN_WALK."""
    adapter = BloomCityLiveAdapter(seed=7)
    agg = TelemetryAggregator()

    transitions = [
        (Phase.VEHICLE_GREEN, VehicleSignal.GREEN, PedestrianSignal.DONT_WALK, 0, 0),
        (Phase.VEHICLE_YELLOW, VehicleSignal.YELLOW, PedestrianSignal.DONT_WALK, 1, 0),
        (Phase.ALL_RED_CLEARANCE, VehicleSignal.RED, PedestrianSignal.DONT_WALK, 2, 0),
        (Phase.PEDESTRIAN_WALK, VehicleSignal.RED, PedestrianSignal.WALK, 2, 1),
        (Phase.PEDESTRIAN_CLEARANCE, VehicleSignal.RED, PedestrianSignal.CLEARING, 2, 2),
    ]

    for idx, (ph, exp_vs, exp_ps, exp_vs_idx, exp_ps_idx) in enumerate(transitions):
        ctrl = _make_dummy_controller_state(phase=ph)
        sig = map_controller_state(ctrl)
        assert sig.vehicle_signal == exp_vs
        assert sig.pedestrian_signal == exp_ps

        res = PipelineFrameResult(
            frame_index=idx,
            timestamp=float(idx) * 0.1,
            dt=0.1,
            tracked_detections=[],
            spatial_detections=[],
            telemetry_snapshot=agg.aggregate([], frame_index=idx),
            demand_snapshot=_make_dummy_demand_snapshot(),
            controller_state=ctrl,
            signal_state=sig,
        )
        f = adapter.process_frame_result(res, force_record=True)
        assert f["vs"] == exp_vs_idx
        assert f["ps"] == exp_ps_idx
        assert f["ph"] == PHASE_ORDER.index(ph)


# -----------------------------------------------------------------------------
# Acceptance Test 4: Live demand synchronization
# -----------------------------------------------------------------------------
def test_acceptance_4_live_demand_synchronization():
    """Verify live demand numbers match exactly: 2 waiting, 1 crossing, 0 approaching, 2 queued."""
    from app import sync_bloom_city_with_signal

    ctrl = _make_dummy_controller_state(phase=Phase.VEHICLE_GREEN)
    sig = map_controller_state(ctrl)
    demand = _make_dummy_demand_snapshot(
        waiting_peds=2,
        crossing_peds=1,
        approaching_vehs=0,
        queued_vehs=2,
    )

    adapter = BloomCityLiveAdapter(seed=7)
    res = PipelineFrameResult(
        frame_index=1,
        timestamp=0.1,
        dt=0.1,
        tracked_detections=[],
        spatial_detections=[],
        telemetry_snapshot=TelemetryAggregator().aggregate([], frame_index=1),
        demand_snapshot=demand,
        controller_state=ctrl,
        signal_state=sig,
    )
    adapter.process_frame_result(res, force_record=True)

    frames, reasons, html = sync_bloom_city_with_signal(
        signal_state=sig,
        demand=demand,
        live_adapter=adapter,
    )

    payload = _get_html_payload(html)
    f0 = payload["frames"][0]
    assert f0["np"] == 2  # 2 waiting
    assert f0["nc"] == 1  # 1 in crosswalk
    assert f0["nv"] == 0  # 0 approaching
    assert f0["nq"] == 2  # 2 queued


# -----------------------------------------------------------------------------
# Acceptance Test 5: Scenario isolation in Live Controller Mode
# -----------------------------------------------------------------------------
def test_acceptance_5_scenario_isolation():
    """Verify scenario controls (background layout/scenario selection) do NOT appear in Live Controller Mode."""
    # Read app.py source to ensure no scenario selection dropdown is rendered under Live Controller Mode
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()

    # Locate the Live Controller Mode branch in app.py (ends where DEMO MODE begins)
    live_branch_start = src.find('if mode_choice == "Live Controller Mode"')
    live_branch_end = src.find('# DEMO MODE', live_branch_start)
    assert live_branch_start > 0 and live_branch_end > live_branch_start

    live_code = src[live_branch_start:live_branch_end]
    # In Live Controller Mode, scenario selectboxes, fictional background layout, and time of day MUST be hidden
    assert "Background Physical Layout" not in live_code
    assert "Traffic Scenario" not in live_code
    assert "bc_live_scenario" not in live_code
    assert "Balanced Compute" not in live_code
    assert "Pedestrian Surge" not in live_code
    assert "Time of Day" not in live_code
    assert 'selectbox("Controller Mode"' not in live_code
    assert "Freeze:" not in live_code
    # Standby card must be present when no frame has been processed
    assert "LIVE CONTROLLER MODE &bull; STANDBY" in live_code


# -----------------------------------------------------------------------------
# Acceptance Test 6: Scenario Demo still works
# -----------------------------------------------------------------------------
def test_acceptance_6_scenario_demo_still_works():
    """Verify Scenario Demo mode remains fully operational with all scenario options."""
    from app import simulate_bloom_city, sync_bloom_city_with_signal
    from simulation.bloom_city_engine import SCENARIOS

    for sc_name in ("balanced", "ped_surge", "heavy_traffic", "quiet"):
        assert sc_name in SCENARIOS
        frames, reasons = simulate_bloom_city(sc_name, None, seed=7)
        assert len(frames) > 0
        assert len(reasons) > 0

        frames, reasons, html = sync_bloom_city_with_signal(
            signal_state=None,
            scenario=sc_name,
        )
        assert "SCENARIO DEMO" in html
        assert SCENARIOS[sc_name].title in html
        payload = _get_html_payload(html)
        assert len(payload["frames"]) > 0
        assert payload["frames"][-1]["t"] > 0


# -----------------------------------------------------------------------------
# Acceptance Test 7: Reset clears live state
# -----------------------------------------------------------------------------
def test_acceptance_7_reset_clears_live_state():
    """Verify reset detaches live signal, clears recorded frames, and restores Scenario Demo."""
    from app import sync_bloom_city_with_signal

    adapter = BloomCityLiveAdapter(seed=7)
    demand = _make_dummy_demand_snapshot(waiting_peds=3)
    ctrl = _make_dummy_controller_state()
    res = PipelineFrameResult(
        frame_index=15,
        timestamp=1.5,
        dt=0.1,
        tracked_detections=[],
        spatial_detections=[],
        telemetry_snapshot=TelemetryAggregator().aggregate([]),
        demand_snapshot=demand,
        controller_state=ctrl,
        signal_state=map_controller_state(ctrl),
    )
    adapter.process_frame_result(res, force_record=True)
    assert adapter.processed_frames_count == 1
    assert len(adapter.recorded_frames) == 1

    # Perform Reset
    adapter.reset()
    assert adapter.processed_frames_count == 0
    assert adapter.last_frame_index == -1
    assert len(adapter.recorded_frames) == 0

    # Sync with None returns pure Scenario Demo
    frames, reasons, html = sync_bloom_city_with_signal(signal_state=None)
    assert "SCENARIO DEMO" in html
    assert "LIVE CONTROLLER" not in html
    assert "Representative agents derived from scenario telemetry" in html


# -----------------------------------------------------------------------------
# Acceptance Test 8: Video switching clears old state
# -----------------------------------------------------------------------------
def test_acceptance_8_video_switching_clears_old_state():
    """Verify video switching cleanly resets the adapter so new video starts from frame 0."""
    adapter = BloomCityLiveAdapter(seed=7)

    # Process clip 1 up to frame 20
    for idx in range(21):
        res1 = _make_frame_result(
            frame_index=idx,
            timestamp=float(idx) * 0.05,
            dt=0.05,
            waiting_pedestrians=2,
        )
        adapter.process_frame_result(res1, video_name="clip1.mp4", force_record=True)

    assert adapter.last_frame_index == 20
    assert adapter.processed_frames_count == 21
    assert adapter.video_name == "clip1.mp4"

    # Switch video: reset adapter for new clip
    adapter.reset()
    assert adapter.last_frame_index == -1
    assert adapter.processed_frames_count == 0

    # Clip 2 starts at its own frame 0
    res2 = _make_frame_result(
        frame_index=0,
        timestamp=0.0,
        dt=0.05,
        waiting_pedestrians=0,
    )
    adapter.process_frame_result(res2, video_name="clip2.mp4", force_record=True)
    assert adapter.last_frame_index == 0
    assert adapter.processed_frames_count == 1
    assert adapter.video_name == "clip2.mp4"
    assert adapter.demand.pedestrians_waiting == 0


# -----------------------------------------------------------------------------
# Acceptance Test 9: Bloom City renderer live mode payload & countdown logic
# -----------------------------------------------------------------------------
def test_acceptance_9_bloom_city_renderer_live_mode_has_no_wall_clock_timer():
    """Live mode must show the controller state of recorded frames - never a wall-clock countdown.

    (This test previously asserted the presence of a `live_tail` branch that counted `rm` down in
    real elapsed time, which made Bloom City show "0.0s left" regardless of the FSM.)
    """
    from app import sync_bloom_city_with_signal

    adapter = BloomCityLiveAdapter(seed=7)
    demand = _make_dummy_demand_snapshot(waiting_peds=1)
    ctrl = _make_dummy_controller_state(phase=Phase.VEHICLE_GREEN, time_rem=8.0)
    sig = map_controller_state(ctrl)

    res = PipelineFrameResult(
        frame_index=1,
        timestamp=0.1,
        dt=0.1,
        tracked_detections=[],
        spatial_detections=[],
        telemetry_snapshot=TelemetryAggregator().aggregate([], frame_index=1),
        demand_snapshot=demand,
        controller_state=ctrl,
        signal_state=sig,
    )
    adapter.process_frame_result(res, force_record=True)

    frames, reasons, html = sync_bloom_city_with_signal(
        signal_state=sig,
        demand=demand,
        live_adapter=adapter,
    )

    payload = _get_html_payload(html)
    assert payload.get("mode") == "LIVE CONTROLLER"
    for forbidden in ("live_tail", "liveRm", "tau>T"):
        assert forbidden not in html, f"wall-clock live extrapolation is back: {forbidden}"
    assert "isLive" in html and "Replay history" in html
    assert payload["frames"][-1]["rm"] == 8.0      # the controller's own value, untouched


# -----------------------------------------------------------------------------
# Acceptance Test 10: Simulation Source radio is always present and clean
# -----------------------------------------------------------------------------
def test_acceptance_10_simulation_source_radio_and_standby():
    """Verify Simulation Source radio is present regardless of live results and provides standby."""
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()

    # Simulation Source radio is not conditional on has_live_result
    assert 'mode_choice = st.radio(' in src
    assert '"Simulation Source"' in src
    assert '["Live Controller Mode", "Scenario Demo"]' in src
    assert "LIVE CONTROLLER MODE &bull; STANDBY" in src


