"""
Unit tests for the AI Vision Pipeline (YOLOv8n + ByteTrack)
"""

import numpy as np
import pytest
from vision.vision_pipeline import VisionPipeline, get_vision_model


def test_lazy_model_caching():
    """Verifies that get_vision_model caches the YOLO instance and selects valid device."""
    model1, dev1 = get_vision_model("yolov8n.pt")
    model2, dev2 = get_vision_model("yolov8n.pt")

    assert model1 is model2
    assert dev1 == dev2
    assert dev1 in ("cpu", "cuda")


def test_vision_pipeline_process_frame():
    """Executes inference and tracking on a synthetic surveillance frame."""
    pipeline = VisionPipeline()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = pipeline.process_frame(frame=frame, frame_index=0, fps_video=30.0)

    assert res.frame_index == 0
    assert res.inference_time_ms > 0
    assert res.fps > 0
    assert res.annotated_frame.shape == (480, 640, 3)
    assert isinstance(res.tracked_detections, list)
    assert res.pedestrian_count >= 0
    assert res.vehicle_count >= 0
