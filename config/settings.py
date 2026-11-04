"""
CrossGuard Configuration and Calibration Settings
=================================================
Centralized parameters for AI vision detection, ByteTrack tracking,
spatial polygon calibration, adaptive signal controller FSM, and Bloom City simulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Target COCO Classes for Traffic Surveillance
# 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASS_IDS = [0, 2, 3, 5, 7]
CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
PEDESTRIAN_CLASS_IDS = [0]


@dataclass
class VisionConfig:
    """YOLO and ByteTrack configuration."""
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.20
    pedestrian_conf_threshold: float = 0.18
    vehicle_conf_threshold: float = 0.22
    inference_size: int = 960  # 640 Fast, 960 Balanced (Recommended), 1280 Precision
    iou_threshold: float = 0.45
    tracker_type: str = "bytetrack.yaml"
    slicing_mode: str = "OFF"  # "OFF", "STREET_ROI", "2_TILE"
    max_history_frames: int = 300
    target_classes: List[int] = field(default_factory=lambda: list(TARGET_CLASS_IDS))


@dataclass
class NormalizedZoneConfig:
    """
    Normalized polygon coordinates (x, y in [0.0, 1.0]) for cross-resolution scaling.
    Calibrated accurately to cover the roadway corridor, curbs, and pedestrian crossings.
    """
    # Pedestrian waiting zones (curbsides / sidewalks)
    waiting_zone_north: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.05, 0.45),
            (0.30, 0.45),
            (0.30, 0.78),
            (0.05, 0.78),
        ]
    )
    waiting_zone_south: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.70, 0.42),
            (0.96, 0.42),
            (0.96, 0.75),
            (0.70, 0.75),
        ]
    )
    # Zebra crosswalk active corridor (street crossing zone)
    crosswalk_polygon: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.28, 0.42),
            (0.72, 0.42),
            (0.72, 0.76),
            (0.28, 0.76),
        ]
    )
    # Vehicular approach lane / stop queue zone (covering roadway lanes, not rooftops)
    vehicle_approach: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.20, 0.30),
            (0.80, 0.30),
            (0.80, 0.68),
            (0.20, 0.68),
        ]
    )
    # Hysteresis and spatial tolerance parameters
    consecutive_observations_required: int = 3
    dwell_time_alert_seconds: float = 15.0


@dataclass
class SignalFSMConfig:
    """Adaptive Pedestrian-Priority Signal Controller Parameters."""
    # Safety Bounds (MUTCD standards)
    min_vehicle_green: float = 12.0
    max_vehicle_green: float = 60.0
    vehicle_yellow_duration: float = 4.0
    all_red_clearance_duration: float = 2.0
    pedestrian_walk_duration: float = 7.0
    pedestrian_clearance_duration: float = 12.0

    # Pedestrian Priority & Accumulation Policies
    pedestrian_group_threshold: int = 3
    accumulation_interval: float = 8.0
    max_waiting_time: float = 25.0

    # Clearance Extension for Occupied Crosswalk (Safety Policy)
    max_clearance_extensions: int = 3
    extension_seconds: float = 3.0
    consecutive_clear_observations: int = 3


@dataclass
class BloomCityPalette:
    """Color styling for Bloom City pastel digital twin."""
    background: str = "#FDF2F8"       # Soft pastel pink
    road: str = "#CBD5E1"             # Soft slate road
    lane_marking: str = "#F8FAFC"     # Off-white markings
    crosswalk_stripe: str = "#FFFFFF" # Crisp white zebra stripes
    curb: str = "#DDD6FE"             # Soft lilac curb
    waiting_zone: str = "rgba(167, 139, 250, 0.35)" # Lavender translucent fill
    crossing_zone: str = "rgba(244, 114, 182, 0.25)" # Soft rose translucent fill
    vehicle_stop_bar: str = "#F43F5E" # Coral red stop line
    pedestrian_color: str = "#7C3AED" # Deep violet pedestrian
    car_mint: str = "#6EE7B7"
    car_coral: str = "#FDA4AF"
    car_sky: str = "#93C5FD"
    car_cream: str = "#FEF08A"
