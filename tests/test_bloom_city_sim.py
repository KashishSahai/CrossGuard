"""
Unit tests for Bloom City Simulation Engine and Kinematic Synchronization
"""

from core.adaptive_controller import ControllerDecision, SignalPhase
from core.spatial_analysis import SpatialSnapshot
from simulation.bloom_city_engine import BloomCityEngine


def test_bloom_city_synchronization():
    """Verifies that simulation state synchronizes with controller and spatial snapshot."""
    engine = BloomCityEngine()

    snap = SpatialSnapshot(
        timestamp=1.0,
        waiting_pedestrians_count=2,
        waiting_track_ids=[1, 2],
        max_pedestrian_waiting_time=3.0,
        crosswalk_occupied=False,
        crossing_pedestrians_count=0,
        crossing_track_ids=[],
        approach_vehicles_count=3,
        vehicle_track_ids=[10, 11, 12],
        queue_density_level="MEDIUM",
    )

    dec = ControllerDecision(
        timestamp=1.0,
        current_phase=SignalPhase.VEHICLE_GREEN,
        previous_phase=SignalPhase.VEHICLE_GREEN,
        phase_elapsed_seconds=1.0,
        phase_remaining_seconds=14.0,
        decision_case="CASE_A",
        reason="Green active",
        pedestrians_waiting=2,
        max_wait_time=3.0,
        crosswalk_occupied=False,
        vehicles_in_approach=3,
        clearance_extensions=0,
    )

    engine.sync(snap, dec, delta_time=0.1)

    assert engine.current_phase == SignalPhase.VEHICLE_GREEN
    assert engine.countdown_seconds == 14.0
    assert len(engine.pedestrians) >= 2
    assert len(engine.vehicles) >= 2


def test_bloom_city_svg_rendering():
    """Ensures valid and non-empty SVG output is generated."""
    engine = BloomCityEngine()
    svg = engine.render_svg()

    assert "<svg" in svg
    assert "</svg>" in svg
    assert "SECTOR 7" in svg
    assert "WAITING CURB" in svg
