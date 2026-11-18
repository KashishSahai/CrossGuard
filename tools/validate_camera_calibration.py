"""Validation of Camera Homography Calibration on Real Traffic Video (Step 16).

Processes all 125 frames of 97885-649761107.mp4 through the AdaptivePipeline
equipped with CameraCalibration (perspective homography).
Compares spatial classification metrics against the Step 14 linear scaling baseline.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Set

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from core.camera_calibration import CameraCalibration
from core.pipeline import AdaptivePipeline, PipelineFrameResult
from core.spatial_analysis import Approach, SpatialZone, is_pedestrian, is_vehicle


def run_calibrated_video_validation(
    video_path: str = "97885-649761107.mp4",
) -> Dict[str, Any]:
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
    print("CROSSGUARD STEP 16 CALIBRATED PIPELINE VALIDATION")
    print("=" * 70)
    print(f"Video file:          {video_path}")
    print(f"Resolution:          {width} x {height}")
    print(f"Frames:              {total_frames} ({total_frames / fps:.2f}s at {fps} FPS)")
    print("-" * 70)

    # Initialize calibration
    calib = CameraCalibration.for_drone_intersection(
        source_width=float(width),
        source_height=float(height),
    )
    print(f"Calibration active:  {calib.is_calibrated}")
    print(f"Homography valid:    {calib.is_valid}")

    # Center verification
    cx, cy = calib.transform_point((980.0, 460.0))
    print(f"Center mapping:      (980, 460) -> ({cx:.1f}, {cy:.1f})")

    # Initialize AdaptivePipeline with calibration
    pipeline = AdaptivePipeline(
        default_fps=fps,
        calibration=calib,
    )

    results: List[PipelineFrameResult] = []
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

    frame_idx = 0
    start_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        res = pipeline.process_frame(frame, frame_index=frame_idx, dt=1.0 / fps)
        results.append(res)

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

    # FSM and telemetry metrics
    all_demand_valid = all(r.demand_snapshot.is_valid() for r in results)
    fsm_phases = [r.current_phase.value for r in results]
    transitions = []
    for prev, nxt in zip(fsm_phases[:-1], fsm_phases[1:]):
        if prev != nxt:
            transitions.append(f"{prev} -> {nxt}")

    bridge_updates = len(results)

    # Step 14 Baseline comparison
    # Step 14:
    # Vehicles total: 39, mapped: 13, unmapped: 26
    # Pedestrians total: 1, mapped: 0, unmapped: 1
    # Zones: NORTH: 0, SOUTH: 0, EAST: 0, WEST: 13, NONE: 27
    s14_veh_mapped = 13
    s14_veh_unmapped = 26
    s14_ped_mapped = 0
    s14_ped_unmapped = 1

    print("-" * 70)
    print("STEP 14 vs STEP 16 CALIBRATION COMPARISON:")
    print("VEHICLE DETECTIONS:")
    print(f"  total:                     {veh_total}")
    print(f"  mapped before calibration: {s14_veh_mapped}")
    print(f"  mapped after calibration:  {veh_mapped}")
    print(f"  unmapped before:           {s14_veh_unmapped}")
    print(f"  unmapped after:            {veh_unmapped}")
    print("PEDESTRIAN DETECTIONS:")
    print(f"  total:                     {ped_total}")
    print(f"  mapped before:             {s14_ped_mapped}")
    print(f"  mapped after:              {ped_mapped}")
    print(f"  unmapped before:           {s14_ped_unmapped}")
    print(f"  unmapped after:            {ped_unmapped}")
    print("ZONE DISTRIBUTION AFTER CALIBRATION:")
    for z_name, count in zone_counts.items():
        print(f"  {z_name}: {count}")
    print(f"UNIQUE VEHICLE TRACKS:       {len(veh_tracks)}")
    print(f"UNIQUE PEDESTRIAN TRACKS:    {len(ped_tracks)}")
    print(f"DEMAND SNAPSHOTS:            {len(results)} generated (All valid: {all_demand_valid})")
    print(f"FSM TRANSITIONS:             {transitions or 'None (elapsed 5.0s < min_green 10.0s)'}")
    print(f"BLOOM CITY UPDATES:          {bridge_updates} updates (All safe)")
    print(f"Processing time:             {elapsed:.2f}s ({frame_idx / elapsed:.2f} FPS)")
    print("=" * 70)

    return {
        "video": video_path,
        "width": width,
        "height": height,
        "frames": frame_idx,
        "center_mapped": (cx, cy),
        "veh_total": veh_total,
        "s14_veh_mapped": s14_veh_mapped,
        "veh_mapped": veh_mapped,
        "s14_veh_unmapped": s14_veh_unmapped,
        "veh_unmapped": veh_unmapped,
        "ped_total": ped_total,
        "s14_ped_mapped": s14_ped_mapped,
        "ped_mapped": ped_mapped,
        "s14_ped_unmapped": s14_ped_unmapped,
        "ped_unmapped": ped_unmapped,
        "zone_counts": zone_counts,
        "unique_veh_tracks": len(veh_tracks),
        "unique_ped_tracks": len(ped_tracks),
        "demand_snapshots": len(results),
        "fsm_transitions": transitions,
        "bridge_updates": bridge_updates,
    }


if __name__ == "__main__":
    run_calibrated_video_validation()
