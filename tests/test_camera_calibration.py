"""Unit tests for Camera Homography Calibration (Step 16)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from core.camera_calibration import (
    DEFAULT_CALIBRATION_CORRESPONDENCES,
    INTERSECTION_50S_CALIBRATION_CORRESPONDENCES,
    CameraCalibration,
)
from core.models import Phase
from core.perception_adapter import PerceptionAdapter, SpatialDetection
from core.pipeline import AdaptivePipeline
from core.spatial_analysis import Approach, SpatialAnalyzer, SpatialZone, ZoneType
from vision.models import TrackedDetection


# -----------------------------------------------------------------------------
# Test 1 — Identity / Disabled Calibration
# -----------------------------------------------------------------------------
def test_calibration_identity_and_disabled():
    """Verify identity and uncalibrated modes preserve coordinates."""
    # Identity calibration
    ident = CameraCalibration.identity()
    assert ident.is_calibrated is True
    assert ident.is_identity is True
    assert ident.is_valid is True

    pt = (123.4, 567.8)
    assert ident.transform_point(pt) == pt

    bbox = (10.0, 20.0, 50.0, 80.0)
    assert ident.transform_bbox(bbox) == bbox

    # Empty / uncalibrated calibration
    empty_calib = CameraCalibration(correspondences=[])
    assert empty_calib.is_calibrated is False
    assert empty_calib.transform_point(pt) == pt
    assert empty_calib.transform_bbox(bbox) == bbox


# -----------------------------------------------------------------------------
# Test 2 — Known Correspondence Mapping
# -----------------------------------------------------------------------------
def test_known_correspondences_mapping():
    """Verify that landmark correspondences map close to their canonical targets."""
    calib = CameraCalibration.for_drone_intersection()
    assert calib.is_calibrated is True
    assert calib.is_valid is True
    assert calib.homography_matrix is not None
    assert calib.homography_matrix.shape == (3, 3)

    # Check inner box corners:
    # Real NW (620, 180) -> Canonical NW (430, 280)
    nw_x, nw_y = calib.transform_point((620.0, 180.0))
    assert abs(nw_x - 430.0) < 65.0
    assert abs(nw_y - 280.0) < 65.0

    # Real SE (1300, 690) -> Canonical SE (570, 420)
    se_x, se_y = calib.transform_point((1300.0, 690.0))
    assert abs(se_x - 570.0) < 65.0
    assert abs(se_y - 420.0) < 65.0


# -----------------------------------------------------------------------------
# Test 3 — Center Mapping
# -----------------------------------------------------------------------------
def test_center_mapping_precision():
    """Verify that real center (980, 460) maps within tolerance of canonical (500, 350)."""
    calib = CameraCalibration.for_drone_intersection()

    cx, cy = calib.transform_point((980.0, 460.0))
    # Within 20 pixels of ideal canonical center (500, 350)
    assert abs(cx - 500.0) < 20.0
    assert abs(cy - 350.0) < 20.0


# -----------------------------------------------------------------------------
# Test 4 — Approach Mapping
# -----------------------------------------------------------------------------
def test_four_approaches_map_to_intended_zones():
    """Verify that points along each real approach arm map into corresponding canonical zones."""
    calib = CameraCalibration.for_drone_intersection()
    analyzer = SpatialAnalyzer()

    # Real West Approach (300, 540)
    west_pt = calib.transform_point((300.0, 540.0))
    west_zone = analyzer.classify(west_pt, "car")
    assert west_zone in (SpatialZone.WEST_APPROACH, SpatialZone.WEST_QUEUE)
    assert west_zone.approach == Approach.WEST

    # Real North Approach (850, 40)
    north_pt = calib.transform_point((850.0, 40.0))
    north_zone = analyzer.classify(north_pt, "car")
    assert north_zone in (SpatialZone.NORTH_APPROACH, SpatialZone.NORTH_QUEUE)
    assert north_zone.approach == Approach.NORTH

    # Real South Approach (1050, 950)
    south_pt = calib.transform_point((1050.0, 950.0))
    south_zone = analyzer.classify(south_pt, "car")
    assert south_zone in (SpatialZone.SOUTH_APPROACH, SpatialZone.SOUTH_QUEUE)
    assert south_zone.approach == Approach.SOUTH

    # Real East Approach (1600, 330)
    east_pt = calib.transform_point((1600.0, 330.0))
    east_zone = analyzer.classify(east_pt, "car")
    assert east_zone in (SpatialZone.EAST_APPROACH, SpatialZone.EAST_QUEUE)
    assert east_zone.approach == Approach.EAST


# -----------------------------------------------------------------------------
# Test 5 — PerceptionAdapter Integration
# -----------------------------------------------------------------------------
def test_perception_adapter_with_calibration():
    """Verify PerceptionAdapter uses calibration homography when configured."""
    calib = CameraCalibration.for_drone_intersection()
    adapter = PerceptionAdapter(calibration=calib)

    det = TrackedDetection(
        track_id=4,
        class_id=2,
        class_name="car",
        confidence=0.9,
        x1=275.0,
        y1=500.0,
        x2=365.0,
        y2=540.0,
        center_x=320.0,
        center_y=520.0,
        frame_index=1,
    )

    spatial = adapter.classify_detection(det)
    assert spatial.is_in_zone is True
    assert spatial.approach == Approach.WEST
    assert spatial.zone in (SpatialZone.WEST_APPROACH, SpatialZone.WEST_QUEUE)


# -----------------------------------------------------------------------------
# Test 6 — AdaptivePipeline Integration and Backward Compatibility
# -----------------------------------------------------------------------------
def test_adaptive_pipeline_calibration_integration():
    """Verify AdaptivePipeline executes end-to-end with CameraCalibration."""
    calib = CameraCalibration.for_drone_intersection()
    pipeline = AdaptivePipeline(calibration=calib)

    assert pipeline.calibration is not None
    assert pipeline.calibration.is_calibrated is True

    # Empty frame runs cleanly
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res = pipeline.process_frame(dummy_frame, frame_index=0)
    assert res.frame_index == 0
    assert res.demand_snapshot.is_valid()
    assert res.current_phase == Phase.VEHICLE_GREEN


# -----------------------------------------------------------------------------
# Test 7 — 50s Intersection Calibration Properties (Step 20)
# -----------------------------------------------------------------------------
def test_intersection_50s_calibration_properties():
    """Verify that CameraCalibration.for_traffic_intersection_50s initializes properly."""
    calib = CameraCalibration.for_traffic_intersection_50s()
    assert calib.is_calibrated is True
    assert calib.is_valid is True
    assert calib.is_identity is False
    assert len(calib.correspondences) == 17
    assert calib.homography_matrix is not None
    assert calib.homography_matrix.shape == (3, 3)
    assert np.all(np.isfinite(calib.homography_matrix))


# -----------------------------------------------------------------------------
# Test 8 — 50s Intersection Center Mapping (Step 20)
# -----------------------------------------------------------------------------
def test_intersection_50s_center_mapping():
    """Verify real center (960, 715) maps within 20 px of canonical (500, 350)."""
    calib = CameraCalibration.for_traffic_intersection_50s()
    cx, cy = calib.transform_point((960.0, 715.0))
    dist = math.hypot(cx - 500.0, cy - 350.0)
    assert dist < 20.0, f"Center distance {dist:.2f}px exceeds 20px tolerance: ({cx:.1f}, {cy:.1f})"


# -----------------------------------------------------------------------------
# Test 9 — 50s Intersection Pedestrian Waiting Mapping (Step 20)
# -----------------------------------------------------------------------------
def test_intersection_50s_pedestrian_waiting_mapping():
    """Verify pedestrian curbs map into EAST_WAITING and WEST_WAITING zones."""
    calib = CameraCalibration.for_traffic_intersection_50s()
    analyzer = SpatialAnalyzer()

    # East waiting curb (1770, 750)
    east_pt = calib.transform_point((1770.0, 750.0))
    east_zone = analyzer.classify(east_pt, "person")
    assert east_zone == SpatialZone.EAST_WAITING, (
        f"East ped point {east_pt} mapped to {east_zone}, expected EAST_WAITING"
    )

    # West waiting curb (693, 847)
    west_pt = calib.transform_point((693.0, 847.0))
    west_zone = analyzer.classify(west_pt, "person")
    assert west_zone == SpatialZone.WEST_WAITING, (
        f"West ped point {west_pt} mapped to {west_zone}, expected WEST_WAITING"
    )


# -----------------------------------------------------------------------------
# Test 10 — 50s Intersection Zebra Crossing Mapping (Step 20)
# -----------------------------------------------------------------------------
def test_intersection_50s_zebra_crossing_mapping():
    """Verify crosswalk centroids map into SOUTH_CROSSING and NORTH_CROSSING zones."""
    calib = CameraCalibration.for_traffic_intersection_50s()
    analyzer = SpatialAnalyzer()

    # South zebra crossing (1010, 870)
    south_pt = calib.transform_point((1010.0, 870.0))
    south_zone = analyzer.classify(south_pt, "person")
    assert south_zone == SpatialZone.SOUTH_CROSSING, (
        f"South cross point {south_pt} mapped to {south_zone}, expected SOUTH_CROSSING"
    )

    # North zebra crossing (880, 540)
    north_pt = calib.transform_point((880.0, 540.0))
    north_zone = analyzer.classify(north_pt, "person")
    assert north_zone == SpatialZone.NORTH_CROSSING, (
        f"North cross point {north_pt} mapped to {north_zone}, expected NORTH_CROSSING"
    )


# -----------------------------------------------------------------------------
# Test 11 — 50s Intersection Approach Corridors Mapping (Step 20)
# -----------------------------------------------------------------------------
def test_intersection_50s_approach_corridors_mapping():
    """Verify approach corridors map into appropriate approach or queue zones."""
    calib = CameraCalibration.for_traffic_intersection_50s()
    analyzer = SpatialAnalyzer()

    # North approach corridor (765, 415)
    north_pt = calib.transform_point((765.0, 415.0))
    north_zone = analyzer.classify(north_pt, "car")
    assert north_zone in (SpatialZone.NORTH_APPROACH, SpatialZone.NORTH_QUEUE), (
        f"North veh point {north_pt} mapped to {north_zone}"
    )

    # South approach corridor (1080, 1050)
    south_pt = calib.transform_point((1080.0, 1050.0))
    south_zone = analyzer.classify(south_pt, "car")
    assert south_zone in (SpatialZone.SOUTH_APPROACH, SpatialZone.SOUTH_QUEUE), (
        f"South veh point {south_pt} mapped to {south_zone}"
    )

    # West approach corridor (600, 800)
    west_pt = calib.transform_point((600.0, 800.0))
    west_zone = analyzer.classify(west_pt, "car")
    assert west_zone in (SpatialZone.WEST_APPROACH, SpatialZone.WEST_QUEUE), (
        f"West veh point {west_pt} mapped to {west_zone}"
    )


# -----------------------------------------------------------------------------
# Test 12 — Drone Calibration Backward Compatibility (Step 20)
# -----------------------------------------------------------------------------
def test_drone_calibration_backward_compatibility():
    """Verify drone calibration remains completely backward compatible and unchanged."""
    drone_calib = CameraCalibration.for_drone_intersection()
    assert drone_calib.is_calibrated is True
    assert drone_calib.is_valid is True
    assert len(drone_calib.correspondences) == 17

    # Center mapping for drone video: (980, 460) -> (500, 350)
    dcx, dcy = drone_calib.transform_point((980.0, 460.0))
    assert abs(dcx - 500.0) < 20.0
    assert abs(dcy - 350.0) < 20.0


# -----------------------------------------------------------------------------
# Test 13 — Urban Four-Way Intersection Calibration
# -----------------------------------------------------------------------------
def test_urban_intersection_calibration():
    """Verify urban intersection calibration maps landmarks and pedestrian curbs."""
    calib = CameraCalibration.for_urban_intersection()
    assert calib.is_calibrated is True
    assert calib.is_valid is True
    assert calib.homography_matrix is not None
    assert calib.homography_matrix.shape == (3, 3)

    analyzer = SpatialAnalyzer()

    # North zebra (520, 335) -> NORTH_CROSSING
    nx, ny = calib.transform_point((520.0, 335.0))
    assert analyzer.classify((nx, ny), "person") == SpatialZone.NORTH_CROSSING

    # East zebra (1000, 400) -> EAST_CROSSING
    ex, ey = calib.transform_point((1000.0, 400.0))
    assert analyzer.classify((ex, ey), "person") == SpatialZone.EAST_CROSSING

    # West zebra (160, 500) -> WEST_CROSSING
    wx, wy = calib.transform_point((160.0, 500.0))
    assert analyzer.classify((wx, wy), "person") == SpatialZone.WEST_CROSSING

    # East waiting curb (1120, 510) -> EAST_WAITING
    ew_x, ew_y = calib.transform_point((1120.0, 510.0))
    assert analyzer.classify((ew_x, ew_y), "person") == SpatialZone.EAST_WAITING

    # West waiting curb (220, 550) -> WEST_WAITING
    ww_x, ww_y = calib.transform_point((220.0, 550.0))
    assert analyzer.classify((ww_x, ww_y), "person") == SpatialZone.WEST_WAITING


