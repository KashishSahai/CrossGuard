"""
CrossGuard - Bloom City: Core Configuration
===========================================
Defines global application metadata, navigation specifications, supported media
formats, and architectural subsystem definitions for Stage 1.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


APP_NAME = "CrossGuard"
PROJECT_TITLE = "CrossGuard // Bloom City"
PROJECT_SUBTITLE = "AI-Powered Adaptive Pedestrian-Priority Traffic Management Research Simulator"
STAGE_NAME = "Stage 1: The CrossGuard AI Command Centre"
VERSION = "1.0.0-stage1"

# Supported media extensions for video ingestion
SUPPORTED_VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
MAX_VIDEO_SIZE_MB: int = 300

# Navigation Page Registry
NAV_PAGES: Dict[str, Dict[str, str]] = {
    "overview": {
        "title": "Command Overview",
        "icon": "activity",
        "description": "System health, live traffic summary, and recent audit events.",
    },
    "vision_monitor": {
        "title": "AI Vision Monitor",
        "icon": "camera",
        "description": "Real traffic video ingestion, OpenCV preview, and detection viewport.",
    },
    "bloom_city": {
        "title": "Bloom City Twin",
        "icon": "map",
        "description": "Digital twin intersection simulation workspace and connection telemetry.",
    },
    "signal_control": {
        "title": "Adaptive Signal Control",
        "icon": "sliders",
        "description": "Signal state visualization, pedestrian demand policy, and phase timing.",
    },
    "analytics": {
        "title": "Traffic Analytics",
        "icon": "bar-chart-2",
        "description": "Historical spatial density, waiting-time trends, and queue analytics.",
    },
    "gpu_performance": {
        "title": "GPU Performance Lab",
        "icon": "cpu",
        "description": "Hardware telemetry, PyTorch acceleration status, and inference benchmarking.",
    },
}

# Subsystem Registry for Live Health Diagnostics
@dataclass(frozen=True)
class SubsystemDef:
    id: str
    name: str
    target_stage: str
    default_status: str  # "READY", "STANDBY", "STAGE_PENDING", "OFFLINE"
    description: str


SUBSYSTEM_REGISTRY: List[SubsystemDef] = [
    SubsystemDef(
        id="video_ingest",
        name="Video Stream Ingestion Engine",
        target_stage="Stage 1",
        default_status="READY",
        description="Local OpenCV video buffer & format validation harness."
    ),
    SubsystemDef(
        id="yolo_detector",
        name="YOLOv8/v11 Object Detector",
        target_stage="Stage 2",
        default_status="STAGE_PENDING",
        description="Pedestrian & vehicular bounding box inference pipeline."
    ),
    SubsystemDef(
        id="bytetrack_tracker",
        name="ByteTrack Association Filter",
        target_stage="Stage 2",
        default_status="STAGE_PENDING",
        description="Multi-target temporal persistence and trajectory tracking."
    ),
    SubsystemDef(
        id="zone_analyzer",
        name="Spatial Zone & Curb Occupancy Engine",
        target_stage="Stage 2",
        default_status="STAGE_PENDING",
        description="Polygon intersection logic for curb waiting and crosswalk occupancy."
    ),
    SubsystemDef(
        id="adaptive_controller",
        name="Adaptive Signal Priority Controller",
        target_stage="Stage 3",
        default_status="STANDBY",
        description="Dynamic green extension and pedestrian safety preemption engine."
    ),
    SubsystemDef(
        id="bloom_city_bridge",
        name="Bloom City Digital Twin Bridge",
        target_stage="Stage 3",
        default_status="STANDBY",
        description="Real-time WebSocket / shared-memory intersection state link."
    ),
    SubsystemDef(
        id="telemetry_logger",
        name="Telemetry Event Store",
        target_stage="Stage 1",
        default_status="READY",
        description="In-memory audit log and operational event stream."
    ),
]
