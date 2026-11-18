"""Camera Perspective Homography Calibration for CrossGuard (Step 16).

Calibrates real-world overhead camera coordinates (e.g. 1920x1080) directly into
CrossGuard's canonical 1000x800 coordinate system using planar perspective homography.
Operates before spatial zone classification and replaces simple linear scaling
for calibrated camera pathways while preserving backward-compatible identity
behavior when uncalibrated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Validated Default Correspondences (from Step 15)
# -----------------------------------------------------------------------------
# Pairs of ((real_x, real_y), (canonical_x, canonical_y))
DEFAULT_CALIBRATION_CORRESPONDENCES: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    # 1. Intersection Center
    ((980.0, 460.0), (500.0, 350.0)),
    # 2. Inner Intersection Box Corners
    ((620.0, 180.0), (430.0, 280.0)),   # NW
    ((1300.0, 180.0), (570.0, 280.0)),  # NE
    ((1300.0, 690.0), (570.0, 420.0)),  # SE
    ((620.0, 690.0), (430.0, 420.0)),   # SW
    # 3. Zebra Crossings (Centroids)
    ((950.0, 85.0), (500.0, 230.0)),    # North (Crosswalk 1)
    ((960.0, 785.0), (500.0, 470.0)),   # South (Crosswalk 2)
    ((1335.0, 440.0), (620.0, 350.0)),  # East (Crosswalk 3)
    ((585.0, 435.0), (380.0, 350.0)),   # West (Crosswalk 4)
    # 4. Stop Lines (Centroids)
    ((815.0, 115.0), (465.0, 206.0)),   # North Stop
    ((1065.0, 755.0), (535.0, 494.0)),  # South Stop
    ((1385.0, 325.0), (644.0, 315.0)),  # East Stop
    ((540.0, 545.0), (356.0, 385.0)),   # West Stop
    # 5. Approach Corridors (Ingress Centroids)
    ((850.0, 40.0), (465.0, 50.0)),     # North Approach
    ((1050.0, 950.0), (535.0, 650.0)),  # South Approach
    ((1600.0, 330.0), (850.0, 315.0)),  # East Approach
    ((300.0, 540.0), (150.0, 385.0)),   # West Approach
]


# -----------------------------------------------------------------------------
# Validated Correspondences for 50s Traffic Video (Step 20)
# (tools/intersection_50s.mp4)
# -----------------------------------------------------------------------------
INTERSECTION_50S_CALIBRATION_CORRESPONDENCES: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    # 1. Intersection Center
    ((960.0, 715.0), (500.0, 350.0)),
    # 2. Inner Intersection Box Corners
    ((750.0, 655.0), (430.0, 280.0)),   # NW
    ((1150.0, 515.0), (570.0, 280.0)),  # NE
    ((1315.0, 780.0), (570.0, 420.0)),  # SE
    ((785.0, 830.0), (430.0, 420.0)),   # SW
    # 3. Zebra Crossings (Centroids)
    ((880.0, 540.0), (500.0, 230.0)),   # North (Crosswalk 1)
    ((1010.0, 870.0), (500.0, 470.0)),  # South (Crosswalk 2)
    ((1560.0, 550.0), (620.0, 350.0)),  # East (Crosswalk 3)
    ((680.0, 755.0), (380.0, 350.0)),   # West (Crosswalk 4)
    # 4. Stop Lines (Centroids)
    ((800.0, 550.0), (465.0, 206.0)),   # North Stop
    ((1040.0, 930.0), (535.0, 494.0)),  # South Stop
    ((1750.0, 400.0), (644.0, 315.0)),  # East Stop
    ((650.0, 805.0), (356.0, 385.0)),   # West Stop
    # 5. Approach / Pedestrian Corridors
    ((765.0, 415.0), (465.0, 50.0)),    # North Approach Ingress
    ((1080.0, 1050.0), (535.0, 650.0)), # South Approach Ingress
    ((1770.0, 750.0), (620.0, 435.0)),  # East Pedestrian Waiting Curb
    ((693.0, 847.0), (380.0, 435.0)),   # West Pedestrian Waiting Curb
]


# -----------------------------------------------------------------------------
# Validated Correspondences for Urban Four-Way Intersection (1280x720)
# (tools/uploads/gemini_generated_video_f9c39c5e.mp4)
# -----------------------------------------------------------------------------
URBAN_INTERSECTION_CALIBRATION_CORRESPONDENCES: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    # 1. Intersection Center
    ((540.0, 430.0), (500.0, 350.0)),
    # 2. Zebra Crossings (Centroids)
    ((520.0, 335.0), (500.0, 230.0)),   # North zebra
    ((550.0, 560.0), (500.0, 470.0)),   # South zebra
    ((1000.0, 400.0), (620.0, 350.0)),  # East zebra
    ((160.0, 500.0), (380.0, 350.0)),   # West zebra
    # 3. Inner Box Corners
    ((380.0, 370.0), (430.0, 280.0)),   # NW
    ((710.0, 370.0), (570.0, 280.0)),   # NE
    ((780.0, 580.0), (570.0, 420.0)),   # SE
    ((340.0, 580.0), (430.0, 420.0)),   # SW
    # 4. Stop Lines / Corridors
    ((430.0, 270.0), (465.0, 100.0)),   # North approach incoming
    ((590.0, 715.0), (535.0, 620.0)),   # South approach incoming
    # 5. Pedestrian Waiting Curbs
    ((1120.0, 510.0), (620.0, 435.0)),  # East waiting curb
    ((220.0, 550.0), (380.0, 435.0)),   # West waiting curb
]


# -----------------------------------------------------------------------------
# Camera Calibration Component
# -----------------------------------------------------------------------------
@dataclass
class CameraCalibration:
    """Manages perspective homography from camera pixels to canonical 1000x800 coordinates."""
    source_width: float = 1920.0
    source_height: float = 1080.0
    target_width: float = 1000.0
    target_height: float = 800.0
    correspondences: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    _homography_matrix: Optional[np.ndarray] = None
    _inv_homography_matrix: Optional[np.ndarray] = None
    _is_identity: bool = False

    def __post_init__(self) -> None:
        if self._is_identity:
            self._homography_matrix = np.eye(3, dtype=np.float64)
            self._inv_homography_matrix = np.eye(3, dtype=np.float64)
        elif self.correspondences and self._homography_matrix is None:
            self.compute_homography()

    @property
    def is_calibrated(self) -> bool:
        """True if a usable homography matrix has been computed."""
        return self._homography_matrix is not None and self.is_valid

    @property
    def is_identity(self) -> bool:
        """True if this calibration represents an identity pass-through."""
        return self._is_identity

    @property
    def is_valid(self) -> bool:
        """Validates that the homography matrix is 3x3, finite, and non-singular."""
        if self._homography_matrix is None:
            return False
        if self._homography_matrix.shape != (3, 3):
            return False
        if not np.all(np.isfinite(self._homography_matrix)):
            return False
        det = float(np.linalg.det(self._homography_matrix))
        return abs(det) > 1e-12

    @property
    def homography_matrix(self) -> Optional[np.ndarray]:
        """Returns the computed 3x3 perspective transform matrix."""
        return self._homography_matrix

    @property
    def source_resolution(self) -> Tuple[float, float]:
        return (self.source_width, self.source_height)

    @property
    def target_resolution(self) -> Tuple[float, float]:
        return (self.target_width, self.target_height)

    def compute_homography(self) -> np.ndarray:
        """Compute the 3x3 perspective homography matrix using least-squares."""
        if len(self.correspondences) < 4:
            raise ValueError(
                f"At least 4 correspondence pairs required for homography, got {len(self.correspondences)}"
            )

        src_pts = np.float32([pair[0] for pair in self.correspondences])
        dst_pts = np.float32([pair[1] for pair in self.correspondences])

        H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
        if H is None:
            raise RuntimeError("Failed to compute homography matrix from correspondences")

        self._homography_matrix = H.astype(np.float64)
        try:
            self._inv_homography_matrix = np.linalg.inv(self._homography_matrix)
        except np.linalg.LinAlgError:
            self._inv_homography_matrix = None

        return self._homography_matrix

    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a 2D camera point (x, y) into canonical coordinates.

        Guarantees returned coordinates are standard Python floats for compatibility
        with SpatialAnalyzer.
        """
        if not self.is_calibrated or self._is_identity:
            return (float(point[0]), float(point[1]))

        x, y = float(point[0]), float(point[1])
        H = self._homography_matrix
        w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        if abs(w) < 1e-12:
            return (0.0, 0.0)

        dst_x = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
        dst_y = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w

        return (float(dst_x), float(dst_y))

    def transform_bottom_center(self, bottom_center: Tuple[float, float]) -> Tuple[float, float]:
        """Convenience alias for transforming the physical ground-contact point."""
        return self.transform_point(bottom_center)

    def transform_bbox(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Transform a bounding box (x1, y1, x2, y2) into canonical coordinate space.

        Maps all 4 bounding corners through the perspective homography and returns
        the tight axis-aligned bounding box enclosing the transformed corners.
        """
        if not self.is_calibrated or self._is_identity:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        x1, y1, x2, y2 = bbox
        corners = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]
        tx_corners = [self.transform_point(c) for c in corners]

        xs = [pt[0] for pt in tx_corners]
        ys = [pt[1] for pt in tx_corners]

        return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

    # ---------------------------------------------------------------- Factory
    @classmethod
    def from_correspondences(
        cls,
        correspondences: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
        source_width: float = 1920.0,
        source_height: float = 1080.0,
        target_width: float = 1000.0,
        target_height: float = 800.0,
    ) -> CameraCalibration:
        """Create a CameraCalibration from custom landmark correspondence pairs."""
        calib = cls(
            source_width=float(source_width),
            source_height=float(source_height),
            target_width=float(target_width),
            target_height=float(target_height),
            correspondences=list(correspondences),
        )
        calib.compute_homography()
        return calib

    @classmethod
    def for_drone_intersection(
        cls,
        source_width: float = 1920.0,
        source_height: float = 1080.0,
        target_width: float = 1000.0,
        target_height: float = 800.0,
    ) -> CameraCalibration:
        """Factory for the validated overhead drone intersection video (97885-649761107.mp4)."""
        return cls.from_correspondences(
            correspondences=DEFAULT_CALIBRATION_CORRESPONDENCES,
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
        )

    @classmethod
    def for_traffic_intersection_50s(
        cls,
        source_width: float = 1920.0,
        source_height: float = 1080.0,
        target_width: float = 1000.0,
        target_height: float = 800.0,
    ) -> CameraCalibration:
        """Factory for the validated 50s intersection video (tools/intersection_50s.mp4)."""
        return cls.from_correspondences(
            correspondences=INTERSECTION_50S_CALIBRATION_CORRESPONDENCES,
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
        )

    @classmethod
    def for_urban_intersection(
        cls,
        source_width: float = 1280.0,
        source_height: float = 720.0,
        target_width: float = 1000.0,
        target_height: float = 800.0,
    ) -> CameraCalibration:
        """Factory for the validated urban four-way intersection video with crosswalks."""
        return cls.from_correspondences(
            correspondences=URBAN_INTERSECTION_CALIBRATION_CORRESPONDENCES,
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
        )

    @classmethod
    def identity(
        cls,
        target_width: float = 1000.0,
        target_height: float = 800.0,
    ) -> CameraCalibration:
        """Create an identity/pass-through calibration."""
        return cls(
            source_width=float(target_width),
            source_height=float(target_height),
            target_width=float(target_width),
            target_height=float(target_height),
            correspondences=[],
            _is_identity=True,
        )
