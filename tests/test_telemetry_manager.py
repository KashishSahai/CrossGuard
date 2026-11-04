"""
Unit tests for Telemetry Manager, Event Logging & CSV Export
"""

from core.adaptive_controller import ControllerDecision, SignalPhase
from core.spatial_analysis import SpatialSnapshot
from core.telemetry_manager import TelemetryManager


def test_telemetry_recording_and_csv():
    """Verifies that frame records accumulate and generate valid CSV bytes."""
    mgr = TelemetryManager()

    snap = SpatialSnapshot(
        timestamp=2.0,
        waiting_pedestrians_count=3,
        waiting_track_ids=[1, 2, 3],
        max_pedestrian_waiting_time=4.5,
        crosswalk_occupied=False,
        crossing_pedestrians_count=1,
        crossing_track_ids=[99],
        approach_vehicles_count=2,
        vehicle_track_ids=[10, 11],
        queue_density_level="LOW",
    )

    dec = ControllerDecision(
        timestamp=2.0,
        current_phase=SignalPhase.VEHICLE_YELLOW,
        previous_phase=SignalPhase.VEHICLE_GREEN,
        phase_elapsed_seconds=12.0,
        phase_remaining_seconds=4.0,
        decision_case="CASE_C",
        reason="Group threshold reached",
        pedestrians_waiting=3,
        max_wait_time=4.5,
        crosswalk_occupied=False,
        vehicles_in_approach=2,
        clearance_extensions=0,
    )

    mgr.record_step(snap, dec, frame_idx=60)

    stats = mgr.get_summary_stats()
    assert stats["has_data"] is True
    assert stats["total_records"] == 1
    assert stats["total_served"] == 1
    assert stats["priority_activations"] == 1

    csv_data = mgr.get_csv_bytes()
    assert b"timestamp,wall_time" in csv_data
    assert b"pedestrians_waiting" in csv_data
