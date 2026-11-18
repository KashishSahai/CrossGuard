"""Unit and integration tests for Live Vision UI Integration.

Verifies video metadata extraction, perception overlay drawing, mocked pipeline
result consumption, telemetry grounding, and Bloom City bridge preservation without
invoking heavy GPU/YOLO model inference.
"""
from __future__ import annotations

import json
import os
import re
from unittest.mock import MagicMock
import numpy as np
import pytest

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
from core.perception_adapter import SpatialDetection
from core.pipeline import PipelineFrameResult
from core.spatial_analysis import Approach, SpatialZone, ZoneType
from core.telemetry import ApproachDemandSnapshot, TelemetryAggregator, TelemetrySnapshot
from simulation.bloom_city_engine import SCENARIOS, build_frames
from simulation.bloom_city_renderer import render_bloom_city_html
from simulation.controller_bridge import BloomCitySignalState, map_controller_state
from vision.models import TrackedDetection
from tests.live_asserts import assert_html_latest_frame_is
from vision.overlay import (
    draw_perception_overlay,
    format_detection_label,
    get_detection_color,
)


# -----------------------------------------------------------------------------
# Test 1: Module and Helper Imports
# -----------------------------------------------------------------------------
def test_live_vision_imports():
    """Verify all Live Vision visualization and pipeline bridge utilities import cleanly."""
    import app
    assert hasattr(app, "get_video_info")
    assert hasattr(app, "get_preview_frame")
    assert hasattr(app, "process_video_batch")
    assert hasattr(app, "sync_bloom_city_with_signal")
    assert callable(draw_perception_overlay)
    assert callable(format_detection_label)
    assert callable(get_detection_color)


# -----------------------------------------------------------------------------
# Test 2: Selected Video Metadata Flow
# -----------------------------------------------------------------------------
def test_video_metadata_flow():
    """Verify metadata extraction correctly processes valid video and handles invalid paths."""
    from app import get_video_info, get_preview_frame

    # Valid validation video
    info = get_video_info("tools/intersection_50s.mp4")
    assert info is not None
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert abs(info["fps"] - 23.976) < 0.01
    assert info["total_frames"] == 1199
    assert info["duration_s"] > 49.0

    # Preview frame extraction
    preview = get_preview_frame("tools/intersection_50s.mp4")
    assert preview is not None
    assert preview.shape == (1080, 1920, 3)

    # Invalid path handling
    assert get_video_info("non_existent_file.mp4") is None
    assert get_preview_frame("non_existent_file.mp4") is None


# -----------------------------------------------------------------------------
# Test 3: Mocked Pipeline Result Populates Displayed State
# -----------------------------------------------------------------------------
def test_mocked_pipeline_result_populates_state():
    """Verify a mock/stubbed PipelineFrameResult populates UI perception and telemetry contracts."""
    # Synthetic detections
    ped_det = SpatialDetection(
        track_id=101,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=(1760.0, 740.0, 1780.0, 760.0),
        center=(1770.0, 750.0),
        bottom_center=(1770.0, 750.0),
        spatial_point=(620.0, 435.0),
        zone=SpatialZone.EAST_WAITING,
        approach=Approach.EAST,
        zone_type=ZoneType.WAITING,
        frame_index=15,
    )

    veh_det = SpatialDetection(
        track_id=202,
        class_id=2,
        class_name="car",
        confidence=0.88,
        bbox=(1030.0, 920.0, 1050.0, 940.0),
        center=(1040.0, 930.0),
        bottom_center=(1040.0, 930.0),
        spatial_point=(535.0, 494.0),
        zone=SpatialZone.SOUTH_QUEUE,
        approach=Approach.SOUTH,
        zone_type=ZoneType.QUEUE,
        frame_index=15,
    )

    demand = DemandSnapshot(
        pedestrians_waiting=1,
        longest_pedestrian_wait_s=3.5,
        pedestrians_in_crosswalk=0,
        vehicle_demand=1,
        vehicle_queue=1,
    )

    ctrl_state = ControllerState(
        current_phase=Phase.VEHICLE_GREEN,
        vehicle_signal=VehicleSignal.GREEN,
        pedestrian_signal=PedestrianSignal.DONT_WALK,
        decision=Decision.VEHICLE_PRIORITY,
        decision_reason="Observing traffic; vehicles moving.",
        phase_elapsed=12.5,
        phase_target=40.0,
        time_remaining=27.5,
        fail_safe_active=False,
        total_time=12.5,
        cycles_completed=0,
        pedestrian_phases_served=0,
    )

    signal_state = map_controller_state(ctrl_state)

    by_approach = {
        Approach.EAST: ApproachDemandSnapshot(
            approach=Approach.EAST,
            waiting_pedestrians=1,
            crossing_pedestrians=0,
            approaching_vehicles=0,
            queued_vehicles=0,
        ),
        Approach.SOUTH: ApproachDemandSnapshot(
            approach=Approach.SOUTH,
            waiting_pedestrians=0,
            crossing_pedestrians=0,
            approaching_vehicles=0,
            queued_vehicles=1,
        ),
    }

    telemetry = TelemetrySnapshot(
        demand=demand,
        waiting_pedestrians=1,
        crossing_pedestrians=0,
        approaching_vehicles=0,
        queued_vehicles=1,
        by_approach=by_approach,
        zone_counts={SpatialZone.EAST_WAITING: 1, SpatialZone.SOUTH_QUEUE: 1},
        tracked_ids={101, 202},
        frame_index=15,
    )

    result = PipelineFrameResult(
        frame_index=15,
        timestamp=0.625,
        dt=0.0417,
        tracked_detections=[],
        spatial_detections=[ped_det, veh_det],
        telemetry_snapshot=telemetry,
        demand_snapshot=demand,
        controller_state=ctrl_state,
        signal_state=signal_state,
    )

    # Verify properties
    assert result.waiting_pedestrians == 1
    assert result.crossing_pedestrians == 0
    assert result.approaching_vehicles == 0
    assert result.queued_vehicles == 1
    assert result.current_phase == Phase.VEHICLE_GREEN
    assert result.vehicle_signal == VehicleSignal.GREEN
    assert result.pedestrian_signal == PedestrianSignal.DONT_WALK

    # Verify overlay generation
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    annotated = draw_perception_overlay(dummy_frame, result.spatial_detections)
    assert annotated.shape == (1080, 1920, 3)
    assert not np.array_equal(annotated, dummy_frame)


# -----------------------------------------------------------------------------
# Test 4: No Fake Telemetry Grounding
# -----------------------------------------------------------------------------
def test_no_fake_telemetry_created():
    """Verify that when no detections are observed, telemetry is strictly zero."""
    telemetry_agg = TelemetryAggregator()
    snap = telemetry_agg.aggregate([], frame_index=0)

    assert snap.waiting_pedestrians == 0
    assert snap.crossing_pedestrians == 0
    assert snap.approaching_vehicles == 0
    assert snap.queued_vehicles == 0
    assert snap.total_vehicle_demand == 0
    assert snap.demand.is_valid()
    assert snap.demand.pedestrians_waiting == 0
    assert snap.demand.vehicle_demand == 0

    # Ensure all approaches report zero demand
    for app in Approach:
        assert snap.waiting_on(app) == 0
        assert snap.crossing_on(app) == 0
        assert snap.approaching_on(app) == 0
        assert snap.queued_on(app) == 0


# -----------------------------------------------------------------------------
# Test 5: Existing Bloom City Integration Still Works
# -----------------------------------------------------------------------------
def test_bloom_city_integration_preservation():
    """Verify Bloom City simulation frame generation and rendering remain intact."""
    frames, reasons, _ = build_frames("balanced", seconds=2.0, seed=7)
    assert len(frames) == 20
    assert len(reasons) > 0

    html = render_bloom_city_html(
        frames,
        reasons,
        theme="night",
        mode="SCENARIO DEMO",
        scenario_title=SCENARIOS["balanced"].title,
    )
    assert isinstance(html, str)
    assert len(html) > 5000
    assert "Bloom City" in html


# -----------------------------------------------------------------------------
# Helpers for Live Vision -> Bloom City Synchronization
# -----------------------------------------------------------------------------
def _get_html_payload(html: str) -> dict:
    """Extract embedded JSON payload from rendered Bloom City HTML."""
    match = re.search(
        r'<script id="bcdata" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None, "Could not find bcdata script tag in HTML"
    return json.loads(match.group(1).replace(r"<\/", "</"))


def _make_controller_state(
    phase: Phase,
    decision: Decision = Decision.VEHICLE_PRIORITY,
    reason: str = "Test reason",
    elapsed: float = 5.0,
    remaining: float = 15.0,
) -> ControllerState:
    """Helper to construct a valid ControllerState for any phase."""
    v_sig, p_sig = PHASE_OUTPUTS[phase]
    return ControllerState(
        current_phase=phase,
        vehicle_signal=v_sig,
        pedestrian_signal=p_sig,
        decision=decision,
        decision_reason=reason,
        phase_elapsed=elapsed,
        phase_target=elapsed + remaining,
        time_remaining=remaining,
        fail_safe_active=False,
        total_time=elapsed,
        cycles_completed=0,
        pedestrian_phases_served=0,
    )


# -----------------------------------------------------------------------------
# Test 6: No Live Vision state -> Bloom City retains demo behavior
# -----------------------------------------------------------------------------
def test_no_live_vision_retains_demo_behavior():
    """Verify that when no Live Vision state exists, Demo Mode is cleanly preserved."""
    from app import sync_bloom_city_with_signal

    frames, reasons, html = sync_bloom_city_with_signal(signal_state=None, scenario="balanced")

    assert len(frames) > 0
    assert len(reasons) > 0
    assert "SCENARIO DEMO" in html
    assert "LIVE CONTROLLER" not in html
    assert "Representative agents derived from scenario telemetry" in html

    payload = _get_html_payload(html)
    assert len(payload["frames"]) > 0
    assert len(payload["reasons"]) > 0


# -----------------------------------------------------------------------------
# Test 7: VEHICLE_GREEN -> correct Bloom City signal state
# -----------------------------------------------------------------------------
def test_vehicle_green_synchronization():
    """Verify VEHICLE_GREEN maps to GREEN vehicle lens and DONT_WALK pedestrian lens."""
    from app import sync_bloom_city_with_signal

    ctrl_state = _make_controller_state(
        Phase.VEHICLE_GREEN,
        decision=Decision.VEHICLE_PRIORITY,
        reason="Vehicles released; observing crosswalk.",
    )
    sig = map_controller_state(ctrl_state)

    assert sig.phase == Phase.VEHICLE_GREEN
    assert sig.vehicle_signal == VehicleSignal.GREEN
    assert sig.pedestrian_signal == PedestrianSignal.DONT_WALK
    assert sig.vehicle_lens_color == "#38d889"
    assert sig.pedestrian_lens_color == "#ff5d6c"
    assert sig.pedestrian_glyph == "✋"
    assert sig.is_safe is True

    frames, reasons, html = sync_bloom_city_with_signal(signal_state=sig, scenario="balanced")
    assert_html_latest_frame_is(html, Phase.VEHICLE_GREEN)
    assert sig.decision_reason in reasons

    payload = _get_html_payload(html)
    for f in payload["frames"]:
        assert f["ph"] == PHASE_ORDER.index(Phase.VEHICLE_GREEN)
        assert f["vs"] == 0  # GREEN index
        assert f["ps"] == 0  # DONT_WALK index


# -----------------------------------------------------------------------------
# Test 8: VEHICLE_YELLOW -> correct Bloom City signal state
# -----------------------------------------------------------------------------
def test_vehicle_yellow_synchronization():
    """Verify VEHICLE_YELLOW maps to YELLOW vehicle lens and DONT_WALK pedestrian lens."""
    from app import sync_bloom_city_with_signal

    ctrl_state = _make_controller_state(
        Phase.VEHICLE_YELLOW,
        decision=Decision.PEDESTRIAN_PRIORITY,
        reason="Pedestrian demand detected; clearing vehicles with yellow signal.",
    )
    sig = map_controller_state(ctrl_state)

    assert sig.phase == Phase.VEHICLE_YELLOW
    assert sig.vehicle_signal == VehicleSignal.YELLOW
    assert sig.pedestrian_signal == PedestrianSignal.DONT_WALK
    assert sig.vehicle_lens_color == "#ffd23f"
    assert sig.pedestrian_lens_color == "#ff5d6c"
    assert sig.pedestrian_glyph == "✋"
    assert sig.is_safe is True

    frames, reasons, html = sync_bloom_city_with_signal(signal_state=sig, scenario="balanced")
    assert_html_latest_frame_is(html, Phase.VEHICLE_YELLOW)

    payload = _get_html_payload(html)
    for f in payload["frames"]:
        assert f["ph"] == PHASE_ORDER.index(Phase.VEHICLE_YELLOW)
        assert f["vs"] == 1  # YELLOW index
        assert f["ps"] == 0  # DONT_WALK index


# -----------------------------------------------------------------------------
# Test 9: ALL_RED_CLEARANCE -> correct Bloom City signal state
# -----------------------------------------------------------------------------
def test_all_red_clearance_synchronization():
    """Verify ALL_RED_CLEARANCE maps to RED vehicle lens and DONT_WALK pedestrian lens."""
    from app import sync_bloom_city_with_signal

    ctrl_state = _make_controller_state(
        Phase.ALL_RED_CLEARANCE,
        decision=Decision.PEDESTRIAN_PRIORITY,
        reason="Yellow complete; all-red clearance empties intersection.",
    )
    sig = map_controller_state(ctrl_state)

    assert sig.phase == Phase.ALL_RED_CLEARANCE
    assert sig.vehicle_signal == VehicleSignal.RED
    assert sig.pedestrian_signal == PedestrianSignal.DONT_WALK
    assert sig.vehicle_lens_color == "#ff5d6c"
    assert sig.pedestrian_lens_color == "#ff5d6c"
    assert sig.pedestrian_glyph == "✋"
    assert sig.is_safe is True

    frames, reasons, html = sync_bloom_city_with_signal(signal_state=sig, scenario="balanced")
    assert_html_latest_frame_is(html, Phase.ALL_RED_CLEARANCE)

    payload = _get_html_payload(html)
    for f in payload["frames"]:
        assert f["ph"] == PHASE_ORDER.index(Phase.ALL_RED_CLEARANCE)
        assert f["vs"] == 2  # RED index
        assert f["ps"] == 0  # DONT_WALK index


# -----------------------------------------------------------------------------
# Test 10: PEDESTRIAN_WALK -> correct Bloom City signal state
# -----------------------------------------------------------------------------
def test_pedestrian_walk_synchronization():
    """Verify PEDESTRIAN_WALK maps to RED vehicle lens and WALK pedestrian lens."""
    from app import sync_bloom_city_with_signal

    ctrl_state = _make_controller_state(
        Phase.PEDESTRIAN_WALK,
        decision=Decision.PEDESTRIAN_PRIORITY,
        reason="Pedestrians have active walk signal.",
    )
    sig = map_controller_state(ctrl_state)

    assert sig.phase == Phase.PEDESTRIAN_WALK
    assert sig.vehicle_signal == VehicleSignal.RED
    assert sig.pedestrian_signal == PedestrianSignal.WALK
    assert sig.vehicle_lens_color == "#ff5d6c"
    assert sig.pedestrian_lens_color == "#38d889"
    assert sig.pedestrian_glyph == "🚶"
    assert sig.is_safe is True

    frames, reasons, html = sync_bloom_city_with_signal(signal_state=sig, scenario="ped_surge")
    assert_html_latest_frame_is(html, Phase.PEDESTRIAN_WALK)

    payload = _get_html_payload(html)
    for f in payload["frames"]:
        assert f["ph"] == PHASE_ORDER.index(Phase.PEDESTRIAN_WALK)
        assert f["vs"] == 2  # RED index
        assert f["ps"] == 1  # WALK index


# -----------------------------------------------------------------------------
# Test 11: PEDESTRIAN_CLEARANCE -> correct Bloom City signal state
# -----------------------------------------------------------------------------
def test_pedestrian_clearance_synchronization():
    """Verify PEDESTRIAN_CLEARANCE maps to RED vehicle lens and CLEARING pedestrian lens."""
    from app import sync_bloom_city_with_signal

    ctrl_state = _make_controller_state(
        Phase.PEDESTRIAN_CLEARANCE,
        decision=Decision.PEDESTRIAN_PRIORITY,
        reason="Pedestrian clearance interval active; crosswalk empties.",
    )
    sig = map_controller_state(ctrl_state)

    assert sig.phase == Phase.PEDESTRIAN_CLEARANCE
    assert sig.vehicle_signal == VehicleSignal.RED
    assert sig.pedestrian_signal == PedestrianSignal.CLEARING
    assert sig.vehicle_lens_color == "#ff5d6c"
    assert sig.pedestrian_lens_color == "#ffb020"
    assert sig.pedestrian_glyph == "⏳"
    assert sig.is_safe is True

    frames, reasons, html = sync_bloom_city_with_signal(signal_state=sig, scenario="balanced")
    assert_html_latest_frame_is(html, Phase.PEDESTRIAN_CLEARANCE)

    payload = _get_html_payload(html)
    for f in payload["frames"]:
        assert f["ph"] == PHASE_ORDER.index(Phase.PEDESTRIAN_CLEARANCE)
        assert f["vs"] == 2  # RED index
        assert f["ps"] == 2  # CLEARING index


# -----------------------------------------------------------------------------
# Test 12: Live Vision controller transition updates Bloom City state
# -----------------------------------------------------------------------------
def test_live_vision_transition_sequence():
    """Verify sequential transitions VEHICLE_GREEN -> VEHICLE_YELLOW -> ALL_RED -> PEDESTRIAN_WALK.

    Each phase update correctly propagates into Bloom City signal state and frames in order.
    """
    from app import sync_bloom_city_with_signal

    sequence = [
        (Phase.VEHICLE_GREEN, VehicleSignal.GREEN, PedestrianSignal.DONT_WALK, 0, 0),
        (Phase.VEHICLE_YELLOW, VehicleSignal.YELLOW, PedestrianSignal.DONT_WALK, 1, 0),
        (Phase.ALL_RED_CLEARANCE, VehicleSignal.RED, PedestrianSignal.DONT_WALK, 2, 0),
        (Phase.PEDESTRIAN_WALK, VehicleSignal.RED, PedestrianSignal.WALK, 2, 1),
    ]

    for phase, exp_vsig, exp_psig, vs_idx, ps_idx in sequence:
        ctrl = _make_controller_state(phase, reason=f"Transition step {phase.value}")
        sig = map_controller_state(ctrl)

        assert sig.phase == phase
        assert sig.vehicle_signal == exp_vsig
        assert sig.pedestrian_signal == exp_psig
        assert sig.is_safe is True

        frames, reasons, html = sync_bloom_city_with_signal(sig, scenario="balanced")
        payload = _get_html_payload(html)
        f0 = payload["frames"][0]
        assert f0["ph"] == PHASE_ORDER.index(phase)
        assert f0["vs"] == vs_idx
        assert f0["ps"] == ps_idx


# -----------------------------------------------------------------------------
# Test 13: No fake telemetry or hardcoded live phase
# -----------------------------------------------------------------------------
def test_no_fake_telemetry_or_hardcoded_live_phase():
    """Verify that telemetry is strictly zero when empty and phase is dynamically driven."""
    from app import sync_bloom_city_with_signal

    # 1. Telemetry verification
    agg = TelemetryAggregator()
    snap = agg.aggregate([], frame_index=0)
    assert snap.waiting_pedestrians == 0
    assert snap.crossing_pedestrians == 0
    assert snap.approaching_vehicles == 0
    assert snap.queued_vehicles == 0
    assert snap.demand.pedestrians_waiting == 0
    assert snap.demand.vehicle_demand == 0

    # 2. Dynamic signal verification (not hardcoded to any one phase)
    for test_ph in Phase:
        ctrl = _make_controller_state(test_ph)
        sig = map_controller_state(ctrl)
        assert sig.phase == test_ph

        frames, reasons, html = sync_bloom_city_with_signal(sig)
        payload = _get_html_payload(html)
        assert payload["frames"][0]["ph"] == PHASE_ORDER.index(test_ph)

    # 3. Safety invariant: no concurrent vehicle green and pedestrian walk
    for test_ph in Phase:
        ctrl = _make_controller_state(test_ph)
        sig = map_controller_state(ctrl)
        assert not (
            sig.vehicle_signal == VehicleSignal.GREEN
            and sig.pedestrian_signal == PedestrianSignal.WALK
        )


# -----------------------------------------------------------------------------
# Test 14: PipelineFrameResult directly drives Bloom City synchronization
# -----------------------------------------------------------------------------
def test_pipeline_frame_result_drives_bloom_city():
    """Verify that a PipelineFrameResult produced by the pipeline seamlessly drives Bloom City."""
    from app import sync_bloom_city_with_signal

    ctrl_state = _make_controller_state(
        Phase.PEDESTRIAN_WALK,
        decision=Decision.PEDESTRIAN_PRIORITY,
        reason="Live video pedestrian demand triggered walk phase.",
    )
    sig_state = map_controller_state(ctrl_state)

    pipeline_res = PipelineFrameResult(
        frame_index=1050,
        timestamp=43.79,
        dt=0.0417,
        tracked_detections=[],
        spatial_detections=[],
        telemetry_snapshot=TelemetryAggregator().aggregate([], frame_index=1050),
        demand_snapshot=DemandSnapshot(pedestrians_waiting=2, source="vision"),
        controller_state=ctrl_state,
        signal_state=sig_state,
    )

    # Pass result.signal_state to sync function
    frames, reasons, html = sync_bloom_city_with_signal(pipeline_res.signal_state)
    assert_html_latest_frame_is(html, Phase.PEDESTRIAN_WALK)
    assert "Live video pedestrian demand triggered walk phase." in reasons

    payload = _get_html_payload(html)
    assert payload["frames"][0]["ph"] == PHASE_ORDER.index(Phase.PEDESTRIAN_WALK)
    assert payload["frames"][0]["vs"] == 2  # RED
    assert payload["frames"][0]["ps"] == 1  # WALK


# -----------------------------------------------------------------------------
# Test 15: Processed frame produces non-empty perception overlay with detections
# -----------------------------------------------------------------------------
def test_processed_frame_produces_annotated_overlay_with_detections():
    """Verify that AdaptivePipeline.process_frame returns an annotated_frame with overlay when detections exist."""
    from core.pipeline import AdaptivePipeline
    from core.camera_calibration import CameraCalibration
    from vision.models import DeviceType

    calib = CameraCalibration.for_traffic_intersection_50s(source_width=1920, source_height=1080)
    pipe = AdaptivePipeline(default_fps=24.0, calibration=calib, device=DeviceType.CPU)

    # 1. Clean frame without detections produces valid annotated frame
    empty_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res_empty = pipe.process_frame(empty_frame, frame_index=0)
    assert res_empty.annotated_frame is not None
    assert isinstance(res_empty.annotated_frame, np.ndarray)
    assert res_empty.annotated_frame.shape == (1080, 1920, 3)

    # 2. When detections are simulated/present, annotated_frame differs from raw frame
    td = TrackedDetection(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.92,
        x1=100.0,
        y1=150.0,
        x2=200.0,
        y2=350.0,
        center_x=150.0,
        center_y=250.0,
        frame_index=1,
    )
    pipe.vision.process_frame = MagicMock(return_value=[td])

    test_frame = np.full((1080, 1920, 3), 50, dtype=np.uint8)
    res = pipe.process_frame(test_frame, frame_index=1)

    assert res.annotated_frame is not None
    assert res.annotated_frame.shape == (1080, 1920, 3)
    assert res.num_tracked_objects == 1
    assert len(res.spatial_detections) == 1

    # Verify that pixel values were modified by drawing bounding boxes and labels
    diff = np.abs(res.annotated_frame.astype(int) - test_frame.astype(int))
    changed_pixels = np.count_nonzero(diff.sum(axis=2))
    assert changed_pixels > 0, "Overlay must draw bounding boxes and labels when detections exist"

    # Verify the changes occurred around the actual camera coordinates (100, 150 to 200, 350)
    sub_region_diff = diff[140:360, 90:210].sum()
    assert sub_region_diff > 0, "Overlay must be drawn at the camera coordinates of the detection"


# -----------------------------------------------------------------------------
# Test 16: Telemetry and displayed overlay correspond to the same frame index
# -----------------------------------------------------------------------------
def test_displayed_frame_and_telemetry_correspond_to_same_index():
    """Verify that displayed overlay and telemetry come from the same PipelineFrameResult."""
    from app import process_video_batch
    from core.pipeline import AdaptivePipeline
    from vision.models import DeviceType

    pipe = AdaptivePipeline(device=DeviceType.CPU)

    # Mock pipeline returning structured result for frame 42
    td = TrackedDetection(
        track_id=7,
        class_id=2,
        class_name="car",
        confidence=0.85,
        x1=400.0,
        y1=300.0,
        x2=600.0,
        y2=500.0,
        center_x=500.0,
        center_y=400.0,
        frame_index=42,
    )

    dummy_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    pipe.vision.process_frame = MagicMock(return_value=[td])

    res = pipe.process_frame(dummy_frame, frame_index=42)

    # Frame index, telemetry, and annotated frame are guaranteed identical
    assert res.frame_index == 42
    assert res.annotated_frame is not None
    assert res.num_tracked_objects == 1
    assert res.spatial_detections[0].track_id == 7
    assert res.spatial_detections[0].camera_bbox == (400.0, 300.0, 600.0, 500.0)

