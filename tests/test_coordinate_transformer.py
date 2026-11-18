"""Focused unit tests for Camera-Coordinate Normalization (Step 13)."""
from __future__ import annotations

import pytest

from core.perception_adapter import CoordinateTransformer, PerceptionAdapter, SpatialDetection
from core.spatial_analysis import Approach, SpatialAnalyzer, SpatialZone, ZoneType
from vision.models import TrackedDetection


# -----------------------------------------------------------------------------
# Test 1 — 1920x1080 to 1000x800 Scaling
# -----------------------------------------------------------------------------
def test_coordinate_transformer_scaling_1920_to_1000():
    """Verify linear scaling from 1080p camera to canonical 1000x800 coordinates."""
    ct = CoordinateTransformer(
        source_width=1920.0,
        source_height=1080.0,
        target_width=1000.0,
        target_height=800.0,
    )

    assert ct.is_identity is False
    assert ct.scale_x == pytest.approx(1000.0 / 1920.0)
    assert ct.scale_y == pytest.approx(800.0 / 1080.0)

    # Origin
    assert ct.transform_point((0.0, 0.0)) == (0.0, 0.0)

    # Image center (960, 540) -> Canonical center (500, 400)
    cx, cy = ct.transform_point((960.0, 540.0))
    assert cx == pytest.approx(500.0)
    assert cy == pytest.approx(400.0)

    # Bottom-right corner (1920, 1080) -> (1000, 800)
    br_x, br_y = ct.transform_point((1920.0, 1080.0))
    assert br_x == pytest.approx(1000.0)
    assert br_y == pytest.approx(800.0)

    # Bounding box
    bbox_in = (0.0, 0.0, 1920.0, 1080.0)
    bbox_out = ct.transform_bbox(bbox_in)
    assert bbox_out[0] == pytest.approx(0.0)
    assert bbox_out[1] == pytest.approx(0.0)
    assert bbox_out[2] == pytest.approx(1000.0)
    assert bbox_out[3] == pytest.approx(800.0)


# -----------------------------------------------------------------------------
# Test 2 — 1000x800 to 1000x800 Identity
# -----------------------------------------------------------------------------
def test_coordinate_transformer_identity():
    """When source equals target dimensions, transform is strictly identity."""
    ct = CoordinateTransformer(
        source_width=1000.0,
        source_height=800.0,
        target_width=1000.0,
        target_height=800.0,
    )

    assert ct.is_identity is True
    assert ct.scale_x == 1.0
    assert ct.scale_y == 1.0

    pt = (416.0, 230.0)
    assert ct.transform_point(pt) == pt

    bbox = (100.0, 200.0, 300.0, 400.0)
    assert ct.transform_bbox(bbox) == bbox


# -----------------------------------------------------------------------------
# Test 3 — Points Near Each Edge
# -----------------------------------------------------------------------------
def test_coordinate_transformer_edges():
    """Verify points near all four frame edges transform cleanly within bounds."""
    ct = CoordinateTransformer(
        source_width=1920.0,
        source_height=1080.0,
        target_width=1000.0,
        target_height=800.0,
    )

    # Left edge
    lx, ly = ct.transform_point((0.0, 540.0))
    assert lx == pytest.approx(0.0)
    assert ly == pytest.approx(400.0)

    # Right edge
    rx, ry = ct.transform_point((1920.0, 540.0))
    assert rx == pytest.approx(1000.0)
    assert ry == pytest.approx(400.0)

    # Top edge
    tx, ty = ct.transform_point((960.0, 0.0))
    assert tx == pytest.approx(500.0)
    assert ty == pytest.approx(0.0)

    # Bottom edge
    bx, by = ct.transform_point((960.0, 1080.0))
    assert bx == pytest.approx(500.0)
    assert by == pytest.approx(800.0)

    # Inset corners
    top_left = ct.transform_point((19.2, 10.8))
    assert top_left[0] == pytest.approx(10.0)
    assert top_left[1] == pytest.approx(8.0)


# -----------------------------------------------------------------------------
# Test 4 — Bottom-Center Transformation
# -----------------------------------------------------------------------------
def test_bottom_center_transformation():
    """Verify that bottom_center is scaled consistently and used as the spatial reference."""
    adapter = PerceptionAdapter(
        source_width=1920.0,
        source_height=1080.0,
        target_width=1000.0,
        target_height=800.0,
    )

    # Raw detection in 1920x1080 space
    # Box from x1=900, y1=400 to x2=1020, y2=540
    # bottom_center is ((900+1020)/2, 540) = (960.0, 540.0)
    raw_det = TrackedDetection(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.9,
        x1=900.0,
        y1=400.0,
        x2=1020.0,
        y2=540.0,
        center_x=960.0,
        center_y=470.0,
        frame_index=0,
    )

    sd = adapter.classify_detection(raw_det)

    assert isinstance(sd, SpatialDetection)
    # The normalized bottom_center should map to (960 * 1000/1920, 540 * 800/1080) = (500.0, 400.0)
    assert sd.bottom_center[0] == pytest.approx(500.0)
    assert sd.bottom_center[1] == pytest.approx(400.0)
    assert sd.spatial_point[0] == pytest.approx(500.0)
    assert sd.spatial_point[1] == pytest.approx(400.0)


# -----------------------------------------------------------------------------
# Test 5 — Transformed Detection Enters Expected Spatial Zone
# -----------------------------------------------------------------------------
def test_transformed_detection_enters_expected_spatial_zone():
    """Points that would be outside canonical canvas map to correct zone after scaling."""
    # North curb waiting zone in canonical 1000x800 is at (416.0, 230.0).
    # In 1920x1080 source resolution, this corresponds to:
    #   x_src = 416.0 * 1920 / 1000 = 798.72
    #   y_src = 230.0 * 1080 / 800 = 310.5
    src_x = 798.72
    src_y = 310.5

    # 1. Without scaling (source=1000x800):
    # Point (798.72, 310.5) has x=798.72, which in canonical space is East Approach, NOT North waiting!
    unscaled_adapter = PerceptionAdapter(source_width=1000.0, source_height=800.0)
    det_unscaled = TrackedDetection(
        track_id=10,
        class_id=0,
        class_name="person",
        confidence=0.95,
        x1=src_x - 10,
        y1=src_y - 20,
        x2=src_x + 10,
        y2=src_y,
        center_x=src_x,
        center_y=src_y - 10,
        frame_index=1,
    )
    sd_unscaled = unscaled_adapter.classify_detection(det_unscaled)
    # Without scaling, this does not land in NORTH_WAITING
    assert sd_unscaled.zone != SpatialZone.NORTH_WAITING

    # 2. With 1920x1080 scaling:
    scaled_adapter = PerceptionAdapter(source_width=1920.0, source_height=1080.0)
    sd_scaled = scaled_adapter.classify_detection(det_unscaled)
    # With scaling, bottom-center becomes (416.0, 230.0) -> correctly classified as NORTH_WAITING!
    assert sd_scaled.zone == SpatialZone.NORTH_WAITING
    assert sd_scaled.approach == Approach.NORTH
    assert sd_scaled.zone_type == ZoneType.WAITING


# -----------------------------------------------------------------------------
# Test 6 — Dynamic Resolution Update
# -----------------------------------------------------------------------------
def test_adapter_dynamic_resolution_update():
    """Verify set_source_resolution updates transformer without breaking instance state."""
    adapter = PerceptionAdapter()
    assert adapter.transformer.is_identity is True

    adapter.set_source_resolution(1920.0, 1080.0)
    assert adapter.transformer.is_identity is False
    assert adapter.transformer.source_width == 1920.0
    assert adapter.transformer.source_height == 1080.0

    # Reset to identity
    adapter.set_source_resolution(1000.0, 800.0)
    assert adapter.transformer.is_identity is True
