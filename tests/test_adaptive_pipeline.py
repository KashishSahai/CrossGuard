"""Unit tests for the End-to-End Adaptive Video Pipeline (Step 7)."""
from __future__ import annotations

import pytest

from core.adaptive_controller import AdaptiveController
from core.models import ControllerState, Decision, DemandSnapshot, PedestrianSignal, Phase, VehicleSignal
from core.perception_adapter import PerceptionAdapter
from core.pipeline import AdaptivePipeline, PipelineFrameResult
from core.spatial_analysis import SpatialZone
from core.telemetry import TelemetryAggregator
from simulation.controller_bridge import BloomCitySignalState, ControllerBridge
from vision.backend import MockBackend
from vision.models import DeviceType, RawDetection, VisionPipelineConfig, resolve_device
from vision.vision_pipeline import VisionPipeline


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(device="cpu")


@pytest.fixture
def pipeline(mock_backend: MockBackend) -> AdaptivePipeline:
    cfg = VisionPipelineConfig(device="cpu")
    vp = VisionPipeline(config=cfg, backend=mock_backend)
    return AdaptivePipeline(
        vision_pipeline=vp,
        default_fps=10.0,
    )


# -----------------------------------------------------------------------------
# Test 1 — Empty frame processing
# -----------------------------------------------------------------------------
def test_empty_frame_processing(pipeline: AdaptivePipeline, mock_backend: MockBackend):
    """Empty frame produces 0 demand, valid controller state, and safe signal state."""
    mock_backend.set_detections([])

    res = pipeline.process_frame("dummy_frame", frame_index=0, dt=0.1)

    assert isinstance(res, PipelineFrameResult)
    assert res.frame_index == 0
    assert res.dt == 0.1
    assert res.timestamp == pytest.approx(0.1)
    assert res.num_tracked_objects == 0
    assert len(res.tracked_detections) == 0
    assert len(res.spatial_detections) == 0

    # Demand
    assert res.waiting_pedestrians == 0
    assert res.crossing_pedestrians == 0
    assert res.queued_vehicles == 0
    assert res.vehicle_demand == 0
    assert res.approaching_vehicles == 0

    # Controller & Signals
    assert isinstance(res.controller_state, ControllerState)
    assert isinstance(res.signal_state, BloomCitySignalState)
    assert res.current_phase == res.controller_state.current_phase
    assert res.current_phase == res.signal_state.phase
    assert res.vehicle_signal == res.signal_state.vehicle_signal
    assert res.pedestrian_signal == res.signal_state.pedestrian_signal
    assert res.signal_state.is_safe is True

    # Serialization summary
    summary = res.to_dict()
    assert summary["frame_index"] == 0
    assert summary["num_tracked"] == 0
    assert summary["waiting_pedestrians"] == 0
    assert summary["queued_vehicles"] == 0
    assert "phase" in summary
    assert "vehicle_signal" in summary
    assert "pedestrian_signal" in summary


# -----------------------------------------------------------------------------
# Test 2 — Single waiting pedestrian
# -----------------------------------------------------------------------------
def test_single_waiting_pedestrian(pipeline: AdaptivePipeline, mock_backend: MockBackend):
    """Tracked person at North waiting curb updates demand and controller."""
    # North curb waiting zone: bottom_center at (416.0, 230.0)
    mock_backend.set_detections([
        RawDetection(
            x1=410.0, y1=190.0, x2=422.0, y2=230.0,
            confidence=0.92, class_id=0, class_name="person",
        )
    ])

    res = pipeline.process_frame("dummy_frame", frame_index=1, dt=0.1)

    assert res.num_tracked_objects == 1
    assert res.waiting_pedestrians == 1
    assert res.crossing_pedestrians == 0
    assert res.demand_snapshot.pedestrians_waiting == 1
    assert len(res.spatial_detections) == 1
    assert res.spatial_detections[0].zone == SpatialZone.NORTH_WAITING

    # Safety invariant
    assert res.signal_state.is_safe is True


# -----------------------------------------------------------------------------
# Test 3 — Single queued vehicle
# -----------------------------------------------------------------------------
def test_single_queued_vehicle(pipeline: AdaptivePipeline, mock_backend: MockBackend):
    """Tracked car in North queue updates vehicle demand in controller."""
    # North queue zone: x=465, y in [106, 206] -> bottom_center at (465.0, 180.0)
    mock_backend.set_detections([
        RawDetection(
            x1=455.0, y1=160.0, x2=475.0, y2=180.0,
            confidence=0.89, class_id=2, class_name="car",
        )
    ])

    res = pipeline.process_frame("dummy_frame", frame_index=1, dt=0.1)

    assert res.num_tracked_objects == 1
    assert res.queued_vehicles == 1
    assert res.vehicle_demand >= 1
    assert res.demand_snapshot.vehicle_queue == 1
    assert len(res.spatial_detections) == 1
    assert res.spatial_detections[0].zone == SpatialZone.NORTH_QUEUE
    assert res.signal_state.is_safe is True


# -----------------------------------------------------------------------------
# Test 4 — Mixed demand
# -----------------------------------------------------------------------------
def test_mixed_demand(pipeline: AdaptivePipeline, mock_backend: MockBackend):
    """Both pedestrians and vehicles are tracked, classified, and aggregated together."""
    mock_backend.set_detections([
        # Person at North waiting curb
        RawDetection(
            x1=410.0, y1=190.0, x2=422.0, y2=230.0,
            confidence=0.95, class_id=0, class_name="person",
        ),
        # Truck in East queue: westbound lane y=315, x in [644, 744] -> (700, 315)
        RawDetection(
            x1=680.0, y1=300.0, x2=720.0, y2=315.0,
            confidence=0.88, class_id=7, class_name="truck",
        ),
    ])

    res = pipeline.process_frame("dummy_frame", frame_index=2, dt=0.1)

    assert res.num_tracked_objects == 2
    assert res.waiting_pedestrians == 1
    assert res.queued_vehicles == 1
    assert res.vehicle_demand >= 1
    assert len(res.telemetry_snapshot.tracked_ids) == 2
    assert res.signal_state.is_safe is True


# -----------------------------------------------------------------------------
# Test 5 — Phase propagation and signal state mapping
# -----------------------------------------------------------------------------
def test_phase_propagation_and_safety(pipeline: AdaptivePipeline, mock_backend: MockBackend):
    """Controller phase directly dictates BloomCitySignalState and satisfies safety invariants."""
    mock_backend.set_detections([])

    # Run for several seconds to observe controller stepping and phase consistency
    for frame_idx in range(50):
        res = pipeline.process_frame("dummy_frame", frame_index=frame_idx, dt=0.1)
        # Ensure Phase matches between controller and signal state
        assert res.current_phase == res.signal_state.phase
        # Ensure signal state safety invariant holds in every frame
        assert res.signal_state.is_safe is True
        # Ensure mutually exclusive green/walk rule
        assert not (
            res.vehicle_signal == VehicleSignal.GREEN
            and res.pedestrian_signal == PedestrianSignal.WALK
        )


# -----------------------------------------------------------------------------
# Test 6 — Consecutive frame processing & track continuity
# -----------------------------------------------------------------------------
def test_consecutive_frame_continuity(pipeline: AdaptivePipeline, mock_backend: MockBackend):
    """State advances continuously over consecutive frames without state drift."""
    # Constant detection over 5 frames
    det = [
        RawDetection(
            x1=410.0, y1=190.0, x2=422.0, y2=230.0,
            confidence=0.91, class_id=0, class_name="person",
        )
    ]
    mock_backend.set_detections(det)

    results = []
    for i in range(5):
        res = pipeline.process_frame("dummy_frame", frame_index=i, dt=0.1)
        results.append(res)

    # Frame indices and timestamps must be strictly monotonic
    for i, res in enumerate(results):
        assert res.frame_index == i
        assert res.timestamp == pytest.approx(0.1 * (i + 1))
        assert res.num_tracked_objects == 1
        # Track ID should remain stable across consecutive identical frames
        assert res.tracked_detections[0].track_id == 1

    assert pipeline.last_result is results[-1]


# -----------------------------------------------------------------------------
# Test 7 — CPU device mode
# -----------------------------------------------------------------------------
def test_cpu_device_mode():
    """Pipeline operates in CPU mode cleanly."""
    cfg = VisionPipelineConfig(device="cpu")
    mb = MockBackend(device="cpu")
    vp = VisionPipeline(config=cfg, backend=mb)
    pipe = AdaptivePipeline(vision_pipeline=vp)

    assert pipe.device == "cpu"
    assert pipe.vision.device == "cpu"

    res = pipe.process_frame("frame", frame_index=0)
    assert res is not None
    assert res.frame_index == 0


# -----------------------------------------------------------------------------
# Test 8 — Device resolution behavior
# -----------------------------------------------------------------------------
def test_device_resolution_behavior():
    """DeviceType auto resolution resolves cleanly."""
    device = resolve_device(DeviceType.AUTO)
    assert device in ("cpu", "cuda")

    forced_cpu = resolve_device(DeviceType.CPU)
    assert forced_cpu == "cpu"


# -----------------------------------------------------------------------------
# Test 9 — Sequence processing
# -----------------------------------------------------------------------------
def test_process_sequence(pipeline: AdaptivePipeline, mock_backend: MockBackend):
    """process_sequence correctly processes lists or generators of frames."""
    mock_backend.set_detections([])
    frames = [f"frame_{i}" for i in range(8)]

    # Full sequence
    results = pipeline.process_sequence(frames, fps=10.0)
    assert len(results) == 8
    assert [r.frame_index for r in results] == list(range(8))
    assert results[-1].timestamp == pytest.approx(0.8)

    # Reset and test max_frames limiting
    pipeline.reset()
    assert pipeline.current_time == 0.0
    assert pipeline.last_result is None

    limited = pipeline.process_sequence(frames, fps=10.0, max_frames=4)
    assert len(limited) == 4
    assert [r.frame_index for r in limited] == [0, 1, 2, 3]


# -----------------------------------------------------------------------------
# Test 10 — Reset functionality
# -----------------------------------------------------------------------------
def test_pipeline_reset(pipeline: AdaptivePipeline, mock_backend: MockBackend):
    """Reset cleanly clears tracking state, clock, and controller state."""
    mock_backend.set_detections([
        RawDetection(x1=10.0, y1=10.0, x2=20.0, y2=20.0, confidence=0.8,
                     class_id=0, class_name="person")
    ])
    pipeline.process_frame("frame_0", frame_index=0, dt=0.5)
    pipeline.process_frame("frame_1", frame_index=1, dt=0.5)

    assert pipeline.current_time == pytest.approx(1.0)
    assert pipeline.last_result is not None

    pipeline.reset()

    assert pipeline.current_time == 0.0
    assert pipeline.last_result is None
    assert pipeline.controller.state.phase_elapsed == 0.0
    assert len(pipeline.vision.tracker._tracks) == 0
    assert pipeline.vision.tracker._next_id == 1

    # Verify that subsequent detection restarts track ID at 1
    res = pipeline.process_frame("frame_after_reset", frame_index=0, dt=0.1)
    assert res.num_tracked_objects == 1
    assert res.tracked_detections[0].track_id == 1


# -----------------------------------------------------------------------------
# Test 11 — Video file handling error behavior
# -----------------------------------------------------------------------------
def test_video_file_handling_nonexistent(pipeline: AdaptivePipeline):
    """Attempting to process a non-existent video file raises FileNotFoundError or RuntimeError if cv2 missing."""
    with pytest.raises((FileNotFoundError, RuntimeError)):
        pipeline.process_video_file("nonexistent_video_path_xyz_123.mp4")
