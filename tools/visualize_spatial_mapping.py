"""Spatial Mapping Visualization Tool for CrossGuard (Step 14).

Visualizes the canonical 1000x800 CrossGuard intersection zones overlaid on
a representative frame from the real traffic video (97885-649761107.mp4).
Renders all real YOLO detections with bounding boxes, confidence, class,
track ID, transformed bottom-center ground contact point, and assigned spatial zone
(explicitly labeled ZONE_NONE if unmapped).

Also aggregates summary statistics across the complete 125-frame video:
- Vehicle detections: total, mapped, unmapped
- Pedestrian detections: total, mapped, unmapped
- Zone distribution: NORTH, SOUTH, EAST, WEST, NONE
- Unique track IDs per class
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from core.models import DemandSnapshot
from core.perception_adapter import CoordinateTransformer, SpatialDetection
from core.pipeline import AdaptivePipeline, PipelineFrameResult
from core.spatial_analysis import (
    Approach,
    RectRegion,
    SpatialAnalyzer,
    SpatialZone,
    ZoneDefinition,
    ZoneType,
    is_pedestrian,
    is_vehicle,
)
from simulation import geometry as G

# Canonical target resolution
TARGET_WIDTH = 1000
TARGET_HEIGHT = 800

# Color palette for zones (BGR)
COLOR_APPROACH = (220, 120, 30)      # Muted Blue/Slate for Approach
COLOR_QUEUE = (245, 160, 50)         # Bright Light Blue for Queue
COLOR_CROSSING = (0, 220, 255)       # Gold / Yellow for Zebra crossing
COLOR_WAITING = (0, 165, 255)        # Orange for Pedestrian Waiting Curbs
COLOR_STOP_LINE = (0, 0, 255)        # Red for Stop lines
COLOR_BOX = (200, 200, 200)          # Light Grey for intersection boundary

# Colors for detections (BGR)
COLOR_DET_MAPPED = (0, 255, 0)       # Bright Green for mapped detections
COLOR_DET_UNMAPPED = (0, 0, 255)     # Bright Red for unmapped detections (ZONE_NONE)
COLOR_SPATIAL_PT = (0, 255, 255)     # Yellow for ground-contact spatial point


def draw_intersection_zones(
    canvas: np.ndarray,
    analyzer: SpatialAnalyzer,
) -> np.ndarray:
    """Draw semi-transparent colored overlays for all 16 canonical CrossGuard zones."""
    overlay = canvas.copy()
    h, w = canvas.shape[:2]

    # Draw all zones from SpatialAnalyzer
    for zone, zone_def in analyzer.all_zones().items():
        if zone.zone_type == ZoneType.APPROACH:
            fill_color = COLOR_APPROACH
            border_color = (255, 140, 50)
            alpha_val = 0.18
        elif zone.zone_type == ZoneType.QUEUE:
            fill_color = COLOR_QUEUE
            border_color = (255, 180, 80)
            alpha_val = 0.28
        elif zone.zone_type == ZoneType.CROSSING:
            fill_color = COLOR_CROSSING
            border_color = (0, 240, 255)
            alpha_val = 0.35
        elif zone.zone_type == ZoneType.WAITING:
            fill_color = COLOR_WAITING
            border_color = (0, 190, 255)
            alpha_val = 0.30
        else:
            continue

        for r in zone_def.regions:
            x1 = int(max(0, min(w - 1, r.x_min)))
            y1 = int(max(0, min(h - 1, r.y_min)))
            x2 = int(max(0, min(w - 1, r.x_max)))
            y2 = int(max(0, min(h - 1, r.y_max)))

            # Filled rectangle on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
            # Solid border on base canvas
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 1)

            # Zone label text
            label = zone.value
            # Abbreviate for compact display
            short_label = (
                label.replace("_APPROACH", "_APP")
                .replace("_CROSSING", "_ZEBRA")
                .replace("_WAITING", "_CURB")
                .replace("_QUEUE", "_Q")
            )
            cx_text = (x1 + x2) // 2
            cy_text = (y1 + y2) // 2
            cv2.putText(
                canvas,
                short_label,
                (cx_text - len(short_label) * 4, cy_text + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

    # Blend overlay with canvas for transparency
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    # Draw Stop Lines (solid red lines at stop line boundaries)
    # North stop: y = 206, x in [430, 500]
    cv2.line(canvas, (430, 206), (500, 206), COLOR_STOP_LINE, 2)
    # South stop: y = 494, x in [500, 570]
    cv2.line(canvas, (500, 494), (570, 494), COLOR_STOP_LINE, 2)
    # East stop: x = 644, y in [280, 350]
    cv2.line(canvas, (644, 280), (644, 350), COLOR_STOP_LINE, 2)
    # West stop: x = 356, y in [350, 420]
    cv2.line(canvas, (356, 350), (356, 420), COLOR_STOP_LINE, 2)

    # Draw Intersection Center Box outline: [430, 280] to [570, 420]
    cv2.rectangle(canvas, (430, 280), (570, 420), COLOR_BOX, 1, lineType=cv2.LINE_AA)

    return canvas


def draw_detections(
    canvas: np.ndarray,
    detections: List[SpatialDetection],
) -> np.ndarray:
    """Draw bounding boxes, spatial points, and classification labels on the canvas."""
    for det in detections:
        is_mapped = det.zone != SpatialZone.NONE
        box_color = COLOR_DET_MAPPED if is_mapped else COLOR_DET_UNMAPPED

        # Transformed bounding box in 1000x800 space
        x1, y1, x2, y2 = [int(round(c)) for c in det.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        # Transformed spatial point (bottom-center ground contact)
        px, py = int(round(det.spatial_point[0])), int(round(det.spatial_point[1]))
        cv2.circle(canvas, (px, py), 5, COLOR_SPATIAL_PT, -1)
        cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1)

        # Label: track ID, class, confidence, and assigned zone (explicit ZONE_NONE if unmapped)
        zone_str = det.zone.value if is_mapped else "ZONE_NONE"
        label_text = f"#{det.track_id} {det.class_name} {det.confidence:.2f} -> {zone_str}"

        # Position label above or below bbox
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        label_y = y1 - 8 if y1 - 8 > lh + 4 else y2 + lh + 8
        label_x = max(4, min(canvas.shape[1] - lw - 8, x1))

        # Text background box
        cv2.rectangle(
            canvas,
            (label_x - 3, label_y - lh - 3),
            (label_x + lw + 3, label_y + 3),
            (20, 20, 20),
            -1,
        )
        cv2.rectangle(
            canvas,
            (label_x - 3, label_y - lh - 3),
            (label_x + lw + 3, label_y + 3),
            box_color,
            1,
        )
        cv2.putText(
            canvas,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def draw_hud(
    canvas: np.ndarray,
    frame_index: int,
    detections: List[SpatialDetection],
    source_res: Tuple[int, int] = (1920, 1080),
    target_res: Tuple[int, int] = (1000, 800),
) -> np.ndarray:
    """Draw top header HUD and bottom legend bar."""
    w = canvas.shape[1]

    # Top HUD Banner (semi-transparent dark bar)
    hud_top = canvas[0:60, 0:w].copy()
    cv2.rectangle(hud_top, (0, 0), (w, 60), (15, 15, 20), -1)
    cv2.addWeighted(hud_top, 0.82, canvas[0:60, 0:w], 0.18, 0, canvas[0:60, 0:w])

    mapped_count = sum(1 for d in detections if d.zone != SpatialZone.NONE)
    unmapped_count = len(detections) - mapped_count

    title = f"CROSSGUARD SPATIAL MAPPING VISUALIZATION | Frame #{frame_index}"
    meta = (
        f"Video: 97885-649761107.mp4 ({source_res[0]}x{source_res[1]} -> Canonical {target_res[0]}x{target_res[1]}) "
        f"| Pipeline: YOLOv8n (CPU) + ByteTrack"
    )
    stats_str = f"Frame Detections: {len(detections)} (Mapped: {mapped_count}, Unmapped: {unmapped_count})"

    cv2.putText(canvas, title, (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, meta, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, stats_str, (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1, cv2.LINE_AA)

    # Bottom Legend Banner
    h = canvas.shape[0]
    hud_bot = canvas[h - 36 : h, 0:w].copy()
    cv2.rectangle(hud_bot, (0, 0), (w, 36), (15, 15, 20), -1)
    cv2.addWeighted(hud_bot, 0.85, canvas[h - 36 : h, 0:w], 0.15, 0, canvas[h - 36 : h, 0:w])

    legend_items = [
        ("ZONES:", (255, 255, 255)),
        ("Approach", COLOR_APPROACH),
        ("Queue", COLOR_QUEUE),
        ("Crossing", COLOR_CROSSING),
        ("Waiting Curb", COLOR_WAITING),
        ("Stop Line", COLOR_STOP_LINE),
        ("| DETECTIONS:", (255, 255, 255)),
        ("Mapped", COLOR_DET_MAPPED),
        ("ZONE_NONE", COLOR_DET_UNMAPPED),
        ("Ground Point", COLOR_SPATIAL_PT),
    ]

    lx = 14
    for text, color in legend_items:
        if text.startswith("|") or text.endswith(":"):
            cv2.putText(canvas, text, (lx, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            lx += tw + 10
        else:
            cv2.rectangle(canvas, (lx, h - 22), (lx + 10, h - 12), color, -1)
            cv2.putText(canvas, text, (lx + 14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            lx += tw + 24

    return canvas


def render_spatial_visualization(
    frame_raw: np.ndarray,
    spatial_detections: List[SpatialDetection],
    frame_index: int,
    source_width: int = 1920,
    source_height: int = 1080,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> np.ndarray:
    """Render the full spatial mapping visualization on a canonical 1000x800 canvas."""
    # 1. Resize raw video frame to canonical target resolution (1000x800)
    canvas = cv2.resize(frame_raw, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # 2. Overlay canonical CrossGuard intersection zones
    analyzer = SpatialAnalyzer()
    canvas = draw_intersection_zones(canvas, analyzer)

    # 3. Draw real detections (using Step 13 transformed coordinates)
    canvas = draw_detections(canvas, spatial_detections)

    # 4. Draw HUD header and legend
    canvas = draw_hud(
        canvas,
        frame_index=frame_index,
        detections=spatial_detections,
        source_res=(source_width, source_height),
        target_res=(target_width, target_height),
    )

    return canvas


def run_full_spatial_evaluation(
    video_path: str = "97885-649761107.mp4",
    representative_frame_index: int = 35,
    output_path: str = "tools/spatial_mapping_debug.jpg",
    artifact_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full video evaluation across all 125 frames and generate visualization."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print("=" * 70)
    print("CROSSGUARD STEP 14 SPATIAL MAPPING DIAGNOSIS & VISUALIZATION")
    print("=" * 70)
    print(f"Video file:       {video_path}")
    print(f"Resolution:       {width} x {height}")
    print(f"Target Canonical: {TARGET_WIDTH} x {TARGET_HEIGHT}")
    print(f"Frames:           {total_frames} ({total_frames / fps:.2f}s at {fps} FPS)")
    print("-" * 70)

    # Initialize AdaptivePipeline with coordinate transformation
    pipeline = AdaptivePipeline(
        default_fps=fps,
        source_resolution=(float(width), float(height)),
    )

    veh_total = 0
    veh_mapped = 0
    veh_unmapped = 0

    ped_total = 0
    ped_mapped = 0
    ped_unmapped = 0

    zone_counts: Dict[str, int] = {
        "NORTH": 0,
        "SOUTH": 0,
        "EAST": 0,
        "WEST": 0,
        "NONE": 0,
    }

    veh_tracks: Set[int] = set()
    ped_tracks: Set[int] = set()

    rep_frame_raw: Optional[np.ndarray] = None
    rep_detections: List[SpatialDetection] = []

    frame_idx = 0
    start_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        res = pipeline.process_frame(frame, frame_index=frame_idx, dt=1.0 / fps)

        # Capture representative frame
        if frame_idx == representative_frame_index:
            rep_frame_raw = frame.copy()
            rep_detections = list(res.spatial_detections)

        # Aggregate statistics
        for s in res.spatial_detections:
            cls = s.class_name.strip().lower()

            if is_vehicle(cls):
                veh_total += 1
                veh_tracks.add(s.track_id)
                if s.zone != SpatialZone.NONE:
                    veh_mapped += 1
                else:
                    veh_unmapped += 1
            elif is_pedestrian(cls):
                ped_total += 1
                ped_tracks.add(s.track_id)
                if s.zone != SpatialZone.NONE:
                    ped_mapped += 1
                else:
                    ped_unmapped += 1

            if s.zone == SpatialZone.NONE:
                zone_counts["NONE"] += 1
            elif s.approach:
                zone_counts[s.approach.value] += 1

        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - start_time

    # Render debug visualization
    if rep_frame_raw is not None:
        vis_image = render_spatial_visualization(
            frame_raw=rep_frame_raw,
            spatial_detections=rep_detections,
            frame_index=representative_frame_index,
            source_width=width,
            source_height=height,
            target_width=TARGET_WIDTH,
            target_height=TARGET_HEIGHT,
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, vis_image)
        print(f"Debug visualization saved to: {output_path}")

        if artifact_path:
            os.makedirs(os.path.dirname(os.path.abspath(artifact_path)), exist_ok=True)
            cv2.imwrite(artifact_path, vis_image)
            print(f"Debug visualization artifact saved to: {artifact_path}")
    else:
        print(f"Warning: Representative frame {representative_frame_index} could not be captured.")

    print("-" * 70)
    print("DETECTION & SPATIAL SUMMARY:")
    print(f"  Vehicles total:          {veh_total}")
    print(f"  Vehicles mapped:         {veh_mapped}")
    print(f"  Vehicles unmapped:       {veh_unmapped}")
    print(f"  Pedestrians total:       {ped_total}")
    print(f"  Pedestrians mapped:      {ped_mapped}")
    print(f"  Pedestrians unmapped:    {ped_unmapped}")
    print("ZONE DISTRIBUTION:")
    for z_name, count in zone_counts.items():
        print(f"  {z_name}: {count}")
    print(f"UNIQUE TRACKS:")
    print(f"  Unique vehicle tracks:    {len(veh_tracks)}")
    print(f"  Unique pedestrian tracks: {len(ped_tracks)}")
    print(f"Total processing time:     {elapsed:.2f}s ({frame_idx / elapsed:.2f} FPS)")
    print("=" * 70)

    return {
        "video_path": video_path,
        "width": width,
        "height": height,
        "frames": frame_idx,
        "output_path": output_path,
        "artifact_path": artifact_path,
        "veh_total": veh_total,
        "veh_mapped": veh_mapped,
        "veh_unmapped": veh_unmapped,
        "ped_total": ped_total,
        "ped_mapped": ped_mapped,
        "ped_unmapped": ped_unmapped,
        "zone_counts": zone_counts,
        "unique_veh_tracks": len(veh_tracks),
        "unique_ped_tracks": len(ped_tracks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CrossGuard Spatial Mapping Visualizer")
    parser.add_argument("--video", default="97885-649761107.mp4", help="Video file path")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=35,
        help="Representative frame index to visualize (default: 35)",
    )
    parser.add_argument(
        "--output",
        default="tools/spatial_mapping_debug.jpg",
        help="Output image path",
    )
    parser.add_argument(
        "--artifact",
        default=r"C:\Users\KIIT0001\.gemini\antigravity\brain\81376bcc-8a14-4cb5-84d7-e0b7a7bfdb56\spatial_mapping_debug.jpg",
        help="Artifact image path",
    )
    args = parser.parse_args()

    run_full_spatial_evaluation(
        video_path=args.video,
        representative_frame_index=args.frame_index,
        output_path=args.output,
        artifact_path=args.artifact,
    )


if __name__ == "__main__":
    main()
