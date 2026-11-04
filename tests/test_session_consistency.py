"""
Unit & Regression Tests for VideoSession Data Consistency & Counter Stability
=============================================================================
Validates that VideoSession attributes, properties, reset operations, and
inference profiles operate with data consistency and zero AttributeError exceptions.
"""

import numpy as np
import pytest

from config.settings import NormalizedZoneConfig, SignalFSMConfig, VisionConfig
from core.session_manager import SessionDiagnostics, SessionStatus, VideoSession
from core.spatial_analysis import SpatialSnapshot
from vision.vision_pipeline import TrackedDetection, VisionFrameResult


def _create_mock_result(
    frame_index: int,
    ped_ids: list,
    veh_ids: list,
    dwell: float = 0.0,
) -> VisionFrameResult:
    """Helper to generate a mock VisionFrameResult with persistent track IDs."""
    dets = []
    for pid in ped_ids:
        dets.append(
            TrackedDetection(
                track_id=pid,
                class_id=0,
                class_name="person",
                confidence=0.35,
                bbox=(100, 200, 130, 260),
                centroid=(115.0, 260.0),
                assigned_zone="WAITING_ZONE",
                dwell_time_seconds=dwell,
                is_pedestrian=True,
                is_vehicle=False,
            )
        )
    for vid in veh_ids:
        dets.append(
            TrackedDetection(
                track_id=vid,
                class_id=2,
                class_name="car",
                confidence=0.65,
                bbox=(300, 400, 380, 500),
                centroid=(340.0, 450.0),
                assigned_zone="APPROACH_QUEUE",
                dwell_time_seconds=0.0,
                is_pedestrian=False,
                is_vehicle=True,
            )
        )
    return VisionFrameResult(
        frame_index=frame_index,
        timestamp_seconds=frame_index / 30.0,
        tracked_detections=dets,
        annotated_frame=np.zeros((100, 100, 3), dtype=np.uint8),
        inference_time_ms=25.0,
        fps=40.0,
        pedestrian_count=len(ped_ids),
        vehicle_count=len(veh_ids),
        device_used="cpu",
    )


def test_video_session_empty_initialization():
    """Verifies that an uninitialized VideoSession safely exposes all counters as zero."""
    session = VideoSession()

    assert session.status == SessionStatus.AWAITING_UPLOAD
    assert session.unique_pedestrians_count == 0
    assert session.unique_vehicles_count == 0
    assert session.total_detections_count == 0
    assert session.unique_tracks_count == 0
    assert session.active_frame_detections_count == 0
    assert session.active_pedestrians_count == 0
    assert session.active_vehicles_count == 0
    assert session.pedestrians_served_count == 0
    assert session.total_source_frames == 0
    assert session.effective_inference_fps == 0.0
    assert session.mean_inference_latency_ms == 0.0
    assert session.is_stale is False


def test_video_session_tracking_counter_consistency():
    """
    Verifies that repeated observations of the same track IDs do NOT multiply unique counts,
    and distinct tracks correctly increment unique pedestrian and vehicle counters.
    """
    session = VideoSession()
    session.video_metadata = {"width": 1920, "height": 1080, "frame_count": 300}

    # Frame 0: Pedestrian #1, Car #10
    res0 = _create_mock_result(frame_index=0, ped_ids=[1], veh_ids=[10])
    session.latest_result = res0
    for det in res0.tracked_detections:
        session.diagnostics.total_detections_count += 1
        session.diagnostics.unique_track_ids.add(det.track_id)
        cname = det.class_name.lower()
        session.diagnostics.detections_by_class[cname] = session.diagnostics.detections_by_class.get(cname, 0) + 1
        session.diagnostics.unique_tracks_by_class[cname].add(det.track_id)

    assert session.unique_pedestrians_count == 1
    assert session.unique_vehicles_count == 1
    assert session.unique_tracks_count == 2
    assert session.total_detections_count == 2
    assert session.active_frame_detections_count == 2
    assert session.active_pedestrians_count == 1
    assert session.active_vehicles_count == 1

    # Frame 1: Same Pedestrian #1, Same Car #10 (Repeated detection)
    res1 = _create_mock_result(frame_index=1, ped_ids=[1], veh_ids=[10])
    session.latest_result = res1
    for det in res1.tracked_detections:
        session.diagnostics.total_detections_count += 1
        session.diagnostics.unique_track_ids.add(det.track_id)
        cname = det.class_name.lower()
        session.diagnostics.detections_by_class[cname] = session.diagnostics.detections_by_class.get(cname, 0) + 1
        session.diagnostics.unique_tracks_by_class[cname].add(det.track_id)

    # Frame detections increased to 4, but unique entities MUST remain 1 and 1
    assert session.total_detections_count == 4
    assert session.unique_pedestrians_count == 1
    assert session.unique_vehicles_count == 1
    assert session.unique_tracks_count == 2

    # Frame 2: Pedestrian #1 + NEW Pedestrian #2, Car #10 + NEW Bus #20
    res2 = _create_mock_result(frame_index=2, ped_ids=[1, 2], veh_ids=[10, 20])
    session.latest_result = res2
    for det in res2.tracked_detections:
        session.diagnostics.total_detections_count += 1
        session.diagnostics.unique_track_ids.add(det.track_id)
        cname = "bus" if det.track_id == 20 else det.class_name.lower()
        session.diagnostics.detections_by_class[cname] = session.diagnostics.detections_by_class.get(cname, 0) + 1
        if cname not in session.diagnostics.unique_tracks_by_class:
            session.diagnostics.unique_tracks_by_class[cname] = set()
        session.diagnostics.unique_tracks_by_class[cname].add(det.track_id)

    assert session.unique_pedestrians_count == 2
    assert session.unique_vehicles_count == 2
    assert session.unique_tracks_count == 4
    assert session.active_pedestrians_count == 2
    assert session.active_vehicles_count == 2


def test_video_session_reset_processing():
    """Verifies that reset_processing resets all counters without AttributeError."""
    session = VideoSession()
    session.video_metadata = {"width": 1920, "height": 1080, "frame_count": 100}
    session.processed_frames_count = 50
    session.inference_latencies_ms = [45.0, 50.0, 55.0]
    session.diagnostics.total_detections_count = 120
    session.diagnostics.unique_tracks_by_class["person"] = {1, 2, 3}
    session.diagnostics.unique_tracks_by_class["car"] = {10, 11}

    assert session.unique_pedestrians_count == 3
    assert session.unique_vehicles_count == 2

    session.reset_processing()

    assert session.processed_frames_count == 0
    assert session.unique_pedestrians_count == 0
    assert session.unique_vehicles_count == 0
    assert session.total_detections_count == 0
    assert session.unique_tracks_count == 0
    assert session.latest_result is None
    assert len(session.inference_latencies_ms) == 0


def test_video_session_stale_configuration_detection():
    """Verifies that changing resolution, stride, or confidence flags stale state."""
    session = VideoSession()
    session.video_path = "mock_traffic.mp4"
    session.processed_frames_count = 10
    session._processed_config_fingerprint = session.config_fingerprint

    assert session.is_stale is False

    # Alter inference resolution profile from 960 to 640
    session.vision_config.inference_size = 640
    assert session.is_stale is True

    # Restore resolution -> not stale
    session.vision_config.inference_size = 960
    assert session.is_stale is False

    # Alter stride from 3 to 1
    session.frame_stride = 1
    assert session.is_stale is True
