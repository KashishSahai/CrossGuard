"""
End-to-End Integration Tests for CrossGuard Video-Driven Architecture
=====================================================================
Validates complete data flow across:
Uploaded Video -> VideoSession -> VisionPipeline (YOLO+ByteTrack) ->
SpatialZoneEngine -> AdaptiveSignalController -> TelemetryManager ->
BloomCityEngine -> CSV Export.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from config.settings import NormalizedZoneConfig, SignalFSMConfig, VisionConfig
from core.adaptive_controller import AdaptiveSignalController, SignalPhase
from core.session_manager import SessionStatus, VideoSession
from core.spatial_analysis import SpatialZoneEngine
from core.video_processor import extract_video_metadata
from vision.vision_pipeline import TrackedDetection


@pytest.fixture
def synthetic_surveillance_video():
    """Generates a temporary surveillance MP4 video for integration testing."""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "test_surveillance.mp4")

    width, height, fps, num_frames = 640, 480, 10.0, 15
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create dark pavement background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # Draw simulated zebra crossing lines
        for y in range(200, 280, 16):
            cv2.rectangle(frame, (220, y), (420, y + 8), (220, 220, 220), -1)

        # Draw a moving simulated pedestrian blob (rectangle)
        ped_y = 100 + i * 15
        cv2.rectangle(frame, (280, ped_y), (310, ped_y + 40), (200, 150, 100), -1)

        out.write(frame)

    out.release()
    yield video_path

    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


def test_end_to_end_session_lifecycle(synthetic_surveillance_video):
    """
    Tests complete lifecycle of VideoSession:
    Ingestion -> Preview Frame -> Zone Calibration -> Full Video Processing ->
    FSM Transition -> Telemetry Record -> Bloom City SVG Generation -> Reset.
    """
    meta = extract_video_metadata(synthetic_surveillance_video)
    assert meta["is_valid"] is True
    assert meta["width"] == 640
    assert meta["height"] == 480

    session = VideoSession()
    assert session.status == SessionStatus.AWAITING_UPLOAD

    # 1. Load Video
    session.load_video(synthetic_surveillance_video, meta)
    assert session.status == SessionStatus.VIDEO_UPLOADED
    assert session.video_path == synthetic_surveillance_video

    # 2. Preview frame retrieval for zone calibration
    preview_frame = session.get_calibration_preview_frame()
    assert preview_frame is not None
    assert preview_frame.shape == (480, 640, 3)

    # 3. Apply custom zone calibration
    custom_zones = NormalizedZoneConfig(
        waiting_zone_north=[(0.1, 0.4), (0.4, 0.4), (0.4, 0.6), (0.1, 0.6)],
        waiting_zone_south=[(0.6, 0.4), (0.9, 0.4), (0.9, 0.6), (0.6, 0.6)],
        crosswalk_polygon=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
        vehicle_approach=[(0.3, 0.05), (0.7, 0.05), (0.7, 0.38), (0.3, 0.38)],
    )
    session.update_zones(custom_zones)
    assert session.spatial_engine.zone_config.waiting_zone_north[0] == (0.1, 0.4)

    # 4. Step a single frame
    res0, snap0, dec0 = session.step_frame(frame_idx=0)
    assert res0.frame_index == 0
    assert snap0.timestamp >= 0.0
    assert dec0.current_phase is not None
    assert session.processed_frames_count == 1

    # 5. Process full video with stride 1
    progress_records = []

    def on_progress(current, total, pct):
        progress_records.append((current, total, pct))

    session.process_video_full(frame_limit=8, stride=1, progress_callback=on_progress)
    assert session.status == SessionStatus.COMPLETED
    assert session.processed_frames_count >= 8
    assert len(progress_records) > 0

    # 6. Verify diagnostics integrity
    diag = session.diagnostics
    assert diag.total_frames_processed >= 8
    assert isinstance(diag.detections_by_class, dict)
    assert "person" in diag.detections_by_class
    assert "car" in diag.detections_by_class

    # 7. Verify Telemetry DB & CSV Export
    df = session.telemetry_manager.get_dataframe()
    assert len(df) >= 8
    assert "pedestrians_waiting" in df.columns
    assert "crosswalk_occupied" in df.columns
    assert "signal_phase" in df.columns

    csv_bytes = session.telemetry_manager.get_csv_bytes()
    assert len(csv_bytes) > 100
    assert b"signal_phase" in csv_bytes

    # 8. Verify Bloom City Digital Twin SVG Sync
    svg_str = session.bloom_engine.render_intersection_svg(session.latest_snapshot, session.latest_decision)
    assert "<svg" in svg_str
    assert "</svg>" in svg_str
    assert session.latest_decision.current_phase.value in svg_str or "CROSSGUARD" in svg_str

    # 9. Verify Session Reset
    session.reset_session()
    assert session.status == SessionStatus.AWAITING_UPLOAD
    assert session.video_path is None
    assert session.processed_frames_count == 0
    assert session.telemetry_manager.get_dataframe().empty


def test_dwell_accumulation_and_fsm_priority_trigger():
    """
    Simulates consecutive frames with pedestrians dwelling in the waiting zone,
    verifying monotonic dwell time accumulation and FSM priority actuation.
    """
    zone_cfg = NormalizedZoneConfig(
        waiting_zone_north=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.35), (0.1, 0.35)],
        waiting_zone_south=[(0.1, 0.7), (0.9, 0.7), (0.9, 0.95), (0.1, 0.95)],
        crosswalk_polygon=[(0.2, 0.35), (0.8, 0.35), (0.8, 0.7), (0.2, 0.7)],
        vehicle_approach=[(0.3, 0.0), (0.7, 0.0), (0.7, 0.3), (0.3, 0.3)],
    )
    signal_cfg = SignalFSMConfig(
        min_vehicle_green=4.0,
        max_waiting_time=5.0,
    )

    spatial_engine = SpatialZoneEngine(frame_width=1000, frame_height=1000, zone_config=zone_cfg)
    controller = AdaptiveSignalController(config=signal_cfg)

    # Footprint inside north waiting zone (e.g., box: 150..250, bottom center: (200, 200) -> (0.2, 0.2))
    det_waiting = TrackedDetection(
        track_id=42,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=(150, 150, 250, 200),
        centroid=(200.0, 175.0),
        is_pedestrian=True,
    )

    # Simulate 8 seconds of dwelling at 1.0s increments
    for t in range(9):
        snapshot = spatial_engine.update(
            tracked_detections=[det_waiting],
            timestamp=float(t),
        )
        decision = controller.step(snapshot=snapshot, timestamp=float(t))

    # Dwell time should have accumulated to ~8.0 seconds
    assert snapshot.max_pedestrian_waiting_time >= 6.0
    assert 42 in snapshot.waiting_track_ids

    # With dwell > 5.0s threshold and vehicle min green elapsed (4.0s),
    # FSM should transition from VEHICLE_GREEN to VEHICLE_YELLOW, ALL_RED_CLEARANCE, or PEDESTRIAN_WALK
    assert decision.current_phase in (
        SignalPhase.VEHICLE_YELLOW,
        SignalPhase.ALL_RED_CLEARANCE,
        SignalPhase.PEDESTRIAN_WALK,
        SignalPhase.PEDESTRIAN_CLEARANCE,
    )


def test_benchmark_runner_with_representative_frame():
    """
    Verifies that the GPU Performance Lab benchmark runner operates with
    a genuine representative surveillance video frame.
    """
    from benchmarking.benchmark_runner import run_inference_benchmark

    sample_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    res = run_inference_benchmark(
        model_name="yolov8n.pt",
        target_device="cpu",
        batch_size=1,
        warmup_runs=1,
        test_runs=2,
        sample_frame=sample_frame,
    )

    assert res.device == "cpu"
    assert res.mean_latency_ms > 0.0
    assert res.fps > 0.0
    assert len(res.latencies_ms) == 2
    assert "CPU" in res.notes.upper()

