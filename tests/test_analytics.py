"""Unit tests for the Analytics & Research Evaluation Module.

Verifies AnalyticsSession metrics aggregation, demand histories, phase transition
recording, spatial peak tracking, performance calculation, and safety invariant validation
without invoking YOLO or importing Streamlit.
"""
from __future__ import annotations

import pytest

from core.analytics import AnalyticsSession, ApproachPeakDemand, PhaseTransitionEvent
from core.models import (
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
from simulation.controller_bridge import BloomCitySignalState, map_controller_state


def _make_dummy_frame_result(
    frame_index: int,
    timestamp: float,
    phase: Phase,
    dt: float = 0.0417,
    waiting_peds: int = 0,
    crossing_peds: int = 0,
    approaching_vehs: int = 0,
    queued_vehs: int = 0,
    decision: Decision = Decision.VEHICLE_PRIORITY,
    reason: str = "Test decision reason",
    ped_wait_s: float = 0.0,
    override_signals: tuple = None,
) -> PipelineFrameResult:
    """Helper to synthesize a valid PipelineFrameResult without vision overhead."""
    if override_signals:
        v_sig, p_sig = override_signals
        sig_state = BloomCitySignalState(
            phase=phase,
            vehicle_signal=v_sig,
            pedestrian_signal=p_sig,
            decision=decision,
            decision_reason=reason,
            phase_index=0,
            vehicle_signal_index=0,
            pedestrian_signal_index=0,
            decision_index=0,
            time_remaining=35.0,
            fail_safe_active=False,
            vehicle_lens_color="#38d889",
            pedestrian_lens_color="#38d889",
            pedestrian_glyph="🚶",
        )
    else:
        from core.models import PHASE_OUTPUTS
        v_sig, p_sig = PHASE_OUTPUTS[phase]
        sig_state = map_controller_state(
            ControllerState(
                current_phase=phase,
                vehicle_signal=v_sig,
                pedestrian_signal=p_sig,
                decision=decision,
                decision_reason=reason,
                phase_elapsed=5.0,
                phase_target=40.0,
                time_remaining=35.0,
                fail_safe_active=False,
                total_time=timestamp,
                cycles_completed=0,
                pedestrian_phases_served=1 if "pedestrian" in reason.lower() else 0,
            )
        )

    ctrl_state = ControllerState(
        current_phase=phase,
        vehicle_signal=v_sig,
        pedestrian_signal=p_sig,
        decision=decision,
        decision_reason=reason,
        phase_elapsed=5.0,
        phase_target=40.0,
        time_remaining=35.0,
        fail_safe_active=False,
        total_time=timestamp,
        cycles_completed=0,
        pedestrian_phases_served=1 if "pedestrian" in reason.lower() else 0,
    )

    demand = DemandSnapshot(
        pedestrians_waiting=waiting_peds,
        longest_pedestrian_wait_s=ped_wait_s,
        pedestrians_in_crosswalk=crossing_peds,
        vehicle_demand=approaching_vehs + queued_vehs,
        vehicle_queue=queued_vehs,
        source="test",
    )

    by_approach = {
        Approach.EAST: ApproachDemandSnapshot(
            approach=Approach.EAST,
            waiting_pedestrians=waiting_peds,
            crossing_pedestrians=crossing_peds,
            approaching_vehicles=0,
            queued_vehicles=0,
        ),
        Approach.SOUTH: ApproachDemandSnapshot(
            approach=Approach.SOUTH,
            waiting_pedestrians=0,
            crossing_pedestrians=0,
            approaching_vehicles=approaching_vehs,
            queued_vehicles=queued_vehs,
        ),
    }

    telemetry = TelemetrySnapshot(
        demand=demand,
        waiting_pedestrians=waiting_peds,
        crossing_pedestrians=crossing_peds,
        approaching_vehicles=approaching_vehs,
        queued_vehicles=queued_vehs,
        by_approach=by_approach,
        zone_counts={},
        tracked_ids=set(),
        frame_index=frame_index,
    )

    return PipelineFrameResult(
        frame_index=frame_index,
        timestamp=timestamp,
        dt=dt,
        tracked_detections=[],
        spatial_detections=[],
        telemetry_snapshot=telemetry,
        demand_snapshot=demand,
        controller_state=ctrl_state,
        signal_state=sig_state,
    )


# -----------------------------------------------------------------------------
# Test 1: Empty Analytics Session
# -----------------------------------------------------------------------------
def test_empty_analytics_session():
    """Verify clean empty state when no frames have been analyzed."""
    session = AnalyticsSession(video_name="test.mp4", video_duration_s=50.0)
    assert session.is_empty is True
    assert session.frames_processed == 0
    assert session.analyzed_duration_s == 0.0
    assert session.measured_fps == 0.0
    assert session.avg_frame_latency_ms == 0.0
    assert session.avg_yolo_latency_ms is None
    assert session.peak_waiting_pedestrians == 0
    assert session.peak_crossing_pedestrians == 0
    assert session.peak_approaching_vehicles == 0
    assert session.peak_queued_vehicles == 0
    assert session.initial_phase is None
    assert session.final_phase is None
    assert session.num_transitions == 0
    assert session.safety_violations == 0
    assert session.green_walk_conflicts == 0


# -----------------------------------------------------------------------------
# Test 2: Single Frame Updates Analytics
# -----------------------------------------------------------------------------
def test_single_frame_updates_analytics():
    """Verify single PipelineFrameResult correctly increments counters and records telemetry."""
    session = AnalyticsSession()
    res = _make_dummy_frame_result(
        frame_index=0,
        timestamp=0.0417,
        phase=Phase.VEHICLE_GREEN,
        dt=0.0417,
        waiting_peds=1,
        queued_vehs=2,
        ped_wait_s=3.5,
    )

    session.record_frame(res, frame_proc_time_s=0.020, yolo_latency_s=0.012)

    assert session.is_empty is False
    assert session.frames_processed == 1
    assert abs(session.analyzed_duration_s - 0.0417) < 0.001
    assert session.peak_waiting_pedestrians == 1
    assert session.peak_queued_vehicles == 2
    assert session.max_pedestrian_wait_s == 3.5
    assert session.initial_phase == Phase.VEHICLE_GREEN
    assert session.final_phase == Phase.VEHICLE_GREEN
    assert session.num_transitions == 0
    assert session.safety_violations == 0


# -----------------------------------------------------------------------------
# Test 3: Multiple Sequential Frames Produce Correct Demand History
# -----------------------------------------------------------------------------
def test_multiple_frames_produce_demand_history():
    """Verify multiple frames maintain ordered demand histories and calculate peaks accurately."""
    session = AnalyticsSession()

    demands = [(0, 0, 1, 1), (1, 0, 2, 2), (2, 1, 3, 2), (2, 2, 1, 1)]
    for i, (w_peds, c_peds, app_v, q_v) in enumerate(demands):
        res = _make_dummy_frame_result(
            frame_index=i,
            timestamp=i * 0.1,
            phase=Phase.VEHICLE_GREEN,
            dt=0.1,
            waiting_peds=w_peds,
            crossing_peds=c_peds,
            approaching_vehs=app_v,
            queued_vehs=q_v,
        )
        session.record_frame(res, frame_proc_time_s=0.01)

    assert session.frames_processed == 4
    assert session.waiting_pedestrians == [0, 1, 2, 2]
    assert session.crossing_pedestrians == [0, 0, 1, 2]
    assert session.approaching_vehicles == [1, 2, 3, 1]
    assert session.queued_vehicles == [1, 2, 2, 1]
    assert session.peak_waiting_pedestrians == 2
    assert session.peak_crossing_pedestrians == 2
    assert session.peak_approaching_vehicles == 3
    assert session.peak_queued_vehicles == 2


# -----------------------------------------------------------------------------
# Test 4: Controller Phase Transitions Recorded Correctly
# -----------------------------------------------------------------------------
def test_controller_phase_transitions_recorded():
    """Verify phase changes trigger discrete PhaseTransitionEvents with exact from/to phases."""
    session = AnalyticsSession()

    res1 = _make_dummy_frame_result(0, 0.0, Phase.VEHICLE_GREEN, reason="Vehicles moving")
    res2 = _make_dummy_frame_result(1, 1.0, Phase.VEHICLE_GREEN, reason="Vehicles moving")
    res3 = _make_dummy_frame_result(2, 2.0, Phase.VEHICLE_YELLOW, reason="Pedestrian demand East; yellow clearance")
    res4 = _make_dummy_frame_result(3, 3.0, Phase.ALL_RED_CLEARANCE, reason="All-red clearance")
    res5 = _make_dummy_frame_result(4, 4.0, Phase.PEDESTRIAN_WALK, reason="Pedestrian walk active")

    for r in [res1, res2, res3, res4, res5]:
        session.record_frame(r, frame_proc_time_s=0.01)

    assert session.num_transitions == 3
    t1, t2, t3 = session.transitions

    assert t1.from_phase == Phase.VEHICLE_GREEN
    assert t1.to_phase == Phase.VEHICLE_YELLOW

    assert t2.from_phase == Phase.VEHICLE_YELLOW
    assert t2.to_phase == Phase.ALL_RED_CLEARANCE

    assert t3.from_phase == Phase.ALL_RED_CLEARANCE
    assert t3.to_phase == Phase.PEDESTRIAN_WALK


# -----------------------------------------------------------------------------
# Test 5: Transition Reasons Preserved from Controller
# -----------------------------------------------------------------------------
def test_transition_reasons_preserved():
    """Verify transition reasons come directly from FSM decision strings."""
    session = AnalyticsSession()
    custom_reason = "Pedestrians waiting (East approach); switching to pedestrian priority."

    res1 = _make_dummy_frame_result(10, 5.0, Phase.VEHICLE_GREEN)
    res2 = _make_dummy_frame_result(11, 5.5, Phase.VEHICLE_YELLOW, reason=custom_reason)

    session.record_frame(res1)
    session.record_frame(res2)

    assert session.num_transitions == 1
    assert session.transitions[0].reason == custom_reason


# -----------------------------------------------------------------------------
# Test 6: Spatial Demand Aggregation by Approach
# -----------------------------------------------------------------------------
def test_spatial_demand_aggregation():
    """Verify peak approach counts are aggregated correctly across all frames."""
    session = AnalyticsSession()

    res1 = _make_dummy_frame_result(1, 0.1, Phase.VEHICLE_GREEN, waiting_peds=1, approaching_vehs=2)
    res2 = _make_dummy_frame_result(2, 0.2, Phase.VEHICLE_GREEN, waiting_peds=3, approaching_vehs=1)

    session.record_frame(res1)
    session.record_frame(res2)

    east_peak = session.approach_peaks[Approach.EAST]
    south_peak = session.approach_peaks[Approach.SOUTH]
    north_peak = session.approach_peaks[Approach.NORTH]

    assert east_peak.waiting_pedestrians == 3
    assert south_peak.approaching_vehicles == 2
    # Unobserved approach remains strictly 0
    assert north_peak.waiting_pedestrians == 0
    assert north_peak.approaching_vehicles == 0


# -----------------------------------------------------------------------------
# Test 7: Phase Duration Calculation Uses Actual Timestamps
# -----------------------------------------------------------------------------
def test_phase_duration_calculation():
    """Verify phase duration accumulation respects actual per-frame dt values."""
    session = AnalyticsSession()

    # 1.0s of VEHICLE_GREEN (10 frames @ 0.1s dt)
    for i in range(10):
        session.record_frame(_make_dummy_frame_result(i, i * 0.1, Phase.VEHICLE_GREEN, dt=0.1))

    # 0.5s of VEHICLE_YELLOW (5 frames @ 0.1s dt)
    for i in range(10, 15):
        session.record_frame(_make_dummy_frame_result(i, i * 0.1, Phase.VEHICLE_YELLOW, dt=0.1))

    assert abs(session.phase_durations[Phase.VEHICLE_GREEN] - 1.0) < 1e-4
    assert abs(session.phase_durations[Phase.VEHICLE_YELLOW] - 0.5) < 1e-4
    assert session.phase_durations[Phase.PEDESTRIAN_WALK] == 0.0

    # Percentages
    pcts = session.phase_distribution_pct
    assert abs(pcts[Phase.VEHICLE_GREEN] - 66.67) < 0.1
    assert abs(pcts[Phase.VEHICLE_YELLOW] - 33.33) < 0.1


# -----------------------------------------------------------------------------
# Test 8: Performance Metrics Calculated Correctly
# -----------------------------------------------------------------------------
def test_performance_metrics_calculation():
    """Verify measured throughput, latency, and distinction between YOLO and end-to-end time."""
    session = AnalyticsSession(device="CPU")

    # 10 frames taking 0.05s total (each 0.005s) -> 200 FPS
    for i in range(10):
        session.record_frame(
            _make_dummy_frame_result(i, i * 0.1, Phase.VEHICLE_GREEN),
            frame_proc_time_s=0.005,
            yolo_latency_s=0.003,
        )

    assert session.frames_processed == 10
    assert abs(session.total_processing_time_s - 0.05) < 1e-4
    assert abs(session.measured_fps - 200.0) < 1e-2
    assert abs(session.avg_frame_latency_ms - 5.0) < 1e-2
    assert session.avg_yolo_latency_ms is not None
    assert abs(session.avg_yolo_latency_ms - 3.0) < 1e-2


# -----------------------------------------------------------------------------
# Test 9: Safety Violations Counted Correctly
# -----------------------------------------------------------------------------
def test_safety_violations_counted():
    """Verify safety checks detect concurrent GREEN and WALK or unsafe states."""
    session = AnalyticsSession()

    # Normal safe frame
    res_safe = _make_dummy_frame_result(0, 0.0, Phase.VEHICLE_GREEN)
    session.record_frame(res_safe)
    assert session.safety_violations == 0
    assert session.green_walk_conflicts == 0

    # Unsafe synthetic conflict (Vehicle GREEN + Pedestrian WALK)
    res_unsafe = _make_dummy_frame_result(
        1, 0.1, Phase.VEHICLE_GREEN, override_signals=(VehicleSignal.GREEN, PedestrianSignal.WALK)
    )
    session.record_frame(res_unsafe)
    assert session.safety_violations == 1
    assert session.green_walk_conflicts == 1


# -----------------------------------------------------------------------------
# Test 10: Reset Clears Analytics Session
# -----------------------------------------------------------------------------
def test_reset_clears_analytics_session():
    """Verify reset() clears all metrics and restores empty state."""
    session = AnalyticsSession()
    res = _make_dummy_frame_result(0, 0.0, Phase.VEHICLE_GREEN, waiting_peds=2)
    session.record_frame(res, frame_proc_time_s=0.02)

    assert session.frames_processed == 1
    assert session.is_empty is False

    session.reset()
    assert session.is_empty is True
    assert session.frames_processed == 0
    assert session.peak_waiting_pedestrians == 0
    assert len(session.timestamps) == 0
    assert len(session.transitions) == 0


# -----------------------------------------------------------------------------
# Test 11: No Fake Telemetry or Placeholder Values Generated
# -----------------------------------------------------------------------------
def test_no_fake_telemetry_generated():
    """Verify empty session summary produces zero/N/A values without fake defaults."""
    session = AnalyticsSession()
    summary = session.to_executive_summary()

    assert summary["is_empty"] is True
    assert summary["frames_processed"] == 0
    assert summary["analyzed_duration_s"] == 0.0
    assert summary["peak_waiting_pedestrians"] == 0
    assert summary["initial_phase"] == "N/A"
    assert summary["final_phase"] == "N/A"
    assert summary["num_transitions"] == 0
    assert summary["safety_violations"] == 0
