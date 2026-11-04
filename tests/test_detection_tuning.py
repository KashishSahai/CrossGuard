"""
Unit Tests for Vision Pipeline Configuration and Small-Object Detection Tuning
=============================================================================
Validates inference resolution settings, class-specific confidence cutoffs,
and tracking format compatibility for small aerial targets.
"""

from config.settings import VisionConfig
from vision.vision_pipeline import VisionPipeline


def test_vision_config_default_parameters():
    """Verifies that defaults are tuned for aerial small-object detection."""
    cfg = VisionConfig()
    assert cfg.inference_size == 960
    assert cfg.pedestrian_conf_threshold == 0.18
    assert cfg.vehicle_conf_threshold == 0.22
    assert cfg.confidence_threshold <= 0.20


def test_vision_pipeline_class_threshold_discrimination():
    """Verifies class-specific confidence thresholding logic in VisionPipeline."""
    cfg = VisionConfig(
        pedestrian_conf_threshold=0.18,
        vehicle_conf_threshold=0.30,
        confidence_threshold=0.18,
    )
    pipeline = VisionPipeline(config=cfg)

    # In YOLO COCO classes: 0 is person, 2 is car
    assert pipeline._get_class_threshold(0) == 0.18
    assert pipeline._get_class_threshold(2) == 0.30
    assert pipeline._get_class_threshold(3) == 0.30  # motorcycle
    assert pipeline._get_class_threshold(5) == 0.30  # bus
    assert pipeline._get_class_threshold(7) == 0.30  # truck


def test_custom_inference_resolutions():
    """Verifies that arbitrary standard inference sizes are supported."""
    for res in [640, 960, 1280]:
        cfg = VisionConfig(inference_size=res)
        assert cfg.inference_size == res
