"""Unit tests for the Vision Detection + ByteTrack Tracking Layer (Step 2)."""
from __future__ import annotations

import pytest

from vision.backend import MockBackend, YOLOBackend
from vision.models import (
    DEFAULT_TRAFFIC_CLASSES,
    DeviceType,
    RawDetection,
    TrackedDetection,
    VisionPipelineConfig,
    is_cuda_available,
    resolve_device,
)
from vision.tracker import ByteTracker, bbox_iou
from vision.vision_pipeline import VisionPipeline


# -----------------------------------------------------------------------------
# Test 1 — Pipeline initialization
# -----------------------------------------------------------------------------
def test_pipeline_initialization_cpu():
    cfg = VisionPipelineConfig(device="cpu", model_name_or_path="yolov8n.pt")
    mock_backend = MockBackend(device="cpu")
    pipeline = VisionPipeline(config=cfg, backend=mock_backend)

    assert pipeline.device == "cpu"
    assert pipeline.is_loaded is True
    assert pipeline.config.get_effective_device() == "cpu"

    # Default YOLO backend initialization without mock also handles CPU gracefully
    default_pipeline = VisionPipeline(config=VisionPipelineConfig(device="cpu"))
    assert default_pipeline.device == "cpu"


# -----------------------------------------------------------------------------
# Test 2 — Output contract
# -----------------------------------------------------------------------------
def test_output_contract():
    mock_backend = MockBackend()
    pipeline = VisionPipeline(backend=mock_backend)

    # Synthetic detection of a person
    mock_backend.set_detections([
        RawDetection(x1=100.0, y1=150.0, x2=200.0, y2=350.0, confidence=0.88,
                     class_id=0, class_name="person")
    ])

    results = pipeline.process_frame("dummy_frame", frame_index=123)

    assert len(results) == 1
    t = results[0]

    assert isinstance(t, TrackedDetection)
    assert t.track_id == 1
    assert t.class_id == 0
    assert t.class_name == "person"
    assert t.confidence == 0.88
    assert t.x1 == 100.0
    assert t.y1 == 150.0
    assert t.x2 == 200.0
    assert t.y2 == 350.0
    assert t.center_x == 150.0
    assert t.center_y == 250.0
    assert t.frame_index == 123
    assert t.bbox == (100.0, 150.0, 200.0, 350.0)
    assert t.center == (150.0, 250.0)
    assert t.bottom_center == (150.0, 350.0)

    # Verify serialization
    d = t.to_dict()
    assert d["track_id"] == 1
    assert d["class_name"] == "person"
    assert d["center_x"] == 150.0
    assert d["center_y"] == 250.0
    assert d["frame_index"] == 123


# -----------------------------------------------------------------------------
# Test 3 — Empty frame
# -----------------------------------------------------------------------------
def test_empty_frame_returns_empty_cleanly():
    mock_backend = MockBackend()
    pipeline = VisionPipeline(backend=mock_backend)

    # None frame
    res_none = pipeline.process_frame(None, frame_index=0)
    assert res_none == []

    # Frame with 0 detections
    mock_backend.set_detections([])
    res_empty = pipeline.process_frame("empty_frame", frame_index=1)
    assert res_empty == []


# -----------------------------------------------------------------------------
# Test 4 — Class filtering
# -----------------------------------------------------------------------------
def test_class_filtering():
    mock_backend = MockBackend()
    pipeline = VisionPipeline(backend=mock_backend)

    # Mix of traffic and non-traffic classes
    mock_backend.set_detections([
        RawDetection(10, 10, 50, 50, 0.9, 0, "person"),
        RawDetection(60, 60, 120, 120, 0.9, 2, "car"),
        RawDetection(130, 130, 180, 180, 0.85, 5, "bus"),
        RawDetection(190, 190, 240, 240, 0.85, 7, "truck"),
        RawDetection(250, 250, 280, 280, 0.80, 3, "motorcycle"),
        RawDetection(300, 300, 340, 340, 0.95, 16, "dog"),
        RawDetection(350, 350, 380, 380, 0.85, 24, "backpack"),
        RawDetection(400, 400, 450, 450, 0.75, 56, "chair"),
    ])

    results = pipeline.process_frame("frame", frame_index=1)
    detected_classes = {r.class_name for r in results}

    # Only traffic classes retained
    assert detected_classes == {"person", "car", "bus", "truck", "motorcycle"}
    assert "dog" not in detected_classes
    assert "backpack" not in detected_classes
    assert "chair" not in detected_classes

    # Config with filtering disabled keeps everything
    pipeline_all = VisionPipeline(
        config=VisionPipelineConfig(filter_classes=False),
        backend=mock_backend,
    )
    all_results = pipeline_all.process_frame("frame", frame_index=1)
    all_classes = {r.class_name for r in all_results}
    assert "dog" in all_classes
    assert "backpack" in all_classes

    # Custom target classes
    pipeline_peds_only = VisionPipeline(
        config=VisionPipelineConfig(target_classes={"person"}),
        backend=mock_backend,
    )
    ped_results = pipeline_peds_only.process_frame("frame", frame_index=1)
    assert len(ped_results) == 1
    assert ped_results[0].class_name == "person"


# -----------------------------------------------------------------------------
# Test 5 — Bounding box center
# -----------------------------------------------------------------------------
def test_bounding_box_center():
    x1, y1, x2, y2 = 40.0, 120.0, 180.0, 360.0
    expected_cx = (x1 + x2) / 2.0  # 110.0
    expected_cy = (y1 + y2) / 2.0  # 240.0

    det = TrackedDetection(
        track_id=1,
        class_id=2,
        class_name="car",
        confidence=0.92,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x=expected_cx,
        center_y=expected_cy,
        frame_index=1,
    )

    assert det.center_x == expected_cx
    assert det.center_y == expected_cy
    assert det.center == (110.0, 240.0)


# -----------------------------------------------------------------------------
# Test 6 — Track ID preservation across consecutive frames
# -----------------------------------------------------------------------------
def test_track_id_preservation():
    mock_backend = MockBackend()
    pipeline = VisionPipeline(backend=mock_backend)

    # Frame 1: Vehicle appears
    mock_backend.set_detections([
        RawDetection(100.0, 100.0, 160.0, 160.0, 0.9, 2, "car")
    ])
    f1 = pipeline.process_frame("f1", frame_index=1)
    assert len(f1) == 1
    first_id = f1[0].track_id

    # Frame 2: Same vehicle slightly moved (high IoU overlap)
    mock_backend.set_detections([
        RawDetection(103.0, 102.0, 163.0, 162.0, 0.9, 2, "car")
    ])
    f2 = pipeline.process_frame("f2", frame_index=2)
    assert len(f2) == 1
    assert f2[0].track_id == first_id

    # Frame 3: Same vehicle moved slightly further
    mock_backend.set_detections([
        RawDetection(106.0, 104.0, 166.0, 164.0, 0.88, 2, "car")
    ])
    f3 = pipeline.process_frame("f3", frame_index=3)
    assert len(f3) == 1
    assert f3[0].track_id == first_id

    # Frame 4: Vehicle disappears
    mock_backend.set_detections([])
    f4 = pipeline.process_frame("f4", frame_index=4)
    assert len(f4) == 0

    # Frame 5: New vehicle appears elsewhere
    mock_backend.set_detections([
        RawDetection(500.0, 500.0, 560.0, 560.0, 0.9, 2, "car")
    ])
    f5 = pipeline.process_frame("f5", frame_index=5)
    assert len(f5) == 1
    assert f5[0].track_id != first_id
    assert f5[0].track_id > first_id


# -----------------------------------------------------------------------------
# Test 7 — Device selection
# -----------------------------------------------------------------------------
def test_device_selection():
    # Explicit CPU request always resolves to 'cpu'
    assert resolve_device("cpu") == "cpu"
    assert resolve_device(DeviceType.CPU) == "cpu"

    # AUTO resolves to 'cuda' only if CUDA is actually available, else 'cpu'
    cuda_avail = is_cuda_available()
    expected_auto = "cuda" if cuda_avail else "cpu"
    assert resolve_device("auto") == expected_auto
    assert resolve_device(DeviceType.AUTO) == expected_auto

    # Verify pipeline configuration adopts resolved device
    pipeline_cpu = VisionPipeline(config=VisionPipelineConfig(device="cpu"),
                                  backend=MockBackend(device="cpu"))
    assert pipeline_cpu.device == "cpu"

    pipeline_auto = VisionPipeline(config=VisionPipelineConfig(device="auto"),
                                   backend=MockBackend(device="auto"))
    assert pipeline_auto.device == expected_auto


# -----------------------------------------------------------------------------
# Test 8 — No-detection frame stability
# -----------------------------------------------------------------------------
def test_no_detection_frame_stability():
    mock_backend = MockBackend()
    mock_backend.set_detections([])
    pipeline = VisionPipeline(backend=mock_backend)

    # Process multiple consecutive empty frames
    for idx in range(10):
        out = pipeline.process_frame(f"frame_{idx}", frame_index=idx)
        assert out == []
        assert isinstance(out, list)


# -----------------------------------------------------------------------------
# Additional Tests: IoU calculation, tracker reset, multi-object frame
# -----------------------------------------------------------------------------
def test_bbox_iou_computation():
    # Identical boxes
    box_a = (10.0, 10.0, 50.0, 50.0)
    assert bbox_iou(box_a, box_a) == 1.0

    # Completely disjoint boxes
    box_b = (100.0, 100.0, 150.0, 150.0)
    assert bbox_iou(box_a, box_b) == 0.0

    # Half overlap: box 1 is [0, 0, 10, 10] (area 100), box 2 is [5, 0, 15, 10] (area 100)
    # Inter area is 5 * 10 = 50. Union area is 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 0.0, 15.0, 10.0)
    assert abs(bbox_iou(box1, box2) - (1.0 / 3.0)) < 1e-6


def test_tracker_reset():
    mock_backend = MockBackend()
    pipeline = VisionPipeline(backend=mock_backend)

    mock_backend.set_detections([
        RawDetection(100.0, 100.0, 150.0, 150.0, 0.9, 0, "person")
    ])
    f1 = pipeline.process_frame("f1", frame_index=1)
    assert f1[0].track_id == 1

    pipeline.reset_tracking()
    # After reset, counter restarts
    f2 = pipeline.process_frame("f2", frame_index=2)
    assert f2[0].track_id == 1


def test_multiple_objects_in_same_frame():
    mock_backend = MockBackend()
    pipeline = VisionPipeline(backend=mock_backend)

    mock_backend.set_detections([
        RawDetection(100.0, 100.0, 150.0, 150.0, 0.9, 0, "person"),
        RawDetection(300.0, 300.0, 380.0, 380.0, 0.92, 2, "car"),
        RawDetection(500.0, 200.0, 600.0, 280.0, 0.85, 5, "bus"),
    ])

    results = pipeline.process_frame("frame", frame_index=1)
    assert len(results) == 3

    track_ids = {r.track_id for r in results}
    assert len(track_ids) == 3  # All distinct IDs

    classes = {r.class_name for r in results}
    assert classes == {"person", "car", "bus"}
