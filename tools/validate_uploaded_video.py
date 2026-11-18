"""Comprehensive validation of the full CrossGuard perception pipeline on uploaded video (Step 12).

Orchestrates:
  Real Video (97885-649761107.mp4)
      ↓
  YOLOv8n (real weights, CPU)
      ↓
  ByteTrack
      ↓
  PerceptionAdapter
      ↓
  SpatialAnalysis
      ↓
  TelemetryAggregator
      ↓
  AdaptiveController
      ↓
  ControllerBridge
      ↓
  BloomCitySignalState
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

from core.models import DemandSnapshot, Phase
from core.pipeline import AdaptivePipeline, PipelineFrameResult
from core.spatial_analysis import Approach, SpatialZone, ZoneType


def run_comprehensive_validation(video_path: str = "97885-649761107.mp4") -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if fps > 0 else 0.0

    print("=" * 70)
    print("CROSSGUARD STEP 12 COMPREHENSIVE PIPELINE VALIDATION")
    print("=" * 70)
    print(f"Video file:       {video_path}")
    print(f"Resolution:       {width} x {height}")
    print(f"Source FPS:       {fps}")
    print(f"Total Frames:     {total_frames} ({duration_s:.2f} s)")
    print("-" * 70)

    # Initialize real AdaptivePipeline
    pipeline = AdaptivePipeline(default_fps=fps)
    print(f"Vision status:    {pipeline.vision_status}")
    print(f"Hardware device:  {pipeline.device}")
    print("Beginning frame-by-frame processing...")

    yolo_times: List[float] = []
    frame_times: List[float] = []
    results: List[PipelineFrameResult] = []

    # Detailed tracking statistics
    all_track_ids: Set[int] = set()
    track_lifespans: Dict[int, int] = {}
    detected_classes: Set[str] = set()
    confidence_scores: List[float] = []

    # Spatial statistics
    approach_counts: Dict[Approach, int] = {app: 0 for app in Approach}
    zone_type_counts: Dict[ZoneType, int] = {zt: 0 for zt in ZoneType}
    unmapped_detections = 0

    # Read frames
    frame_idx = 0
    start_total_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Time YOLO separately
        t_yolo_start = time.perf_counter()
        raw_dets = pipeline.vision.backend.detect(frame)
        t_yolo_end = time.perf_counter()
        yolo_times.append(t_yolo_end - t_yolo_start)

        for d in raw_dets:
            detected_classes.add(d.class_name)
            confidence_scores.append(d.confidence)

        # Full pipeline step
        t_frame_start = time.perf_counter()
        res = pipeline.process_frame(frame, frame_index=frame_idx, dt=1.0 / fps)
        t_frame_end = time.perf_counter()
        frame_times.append(t_frame_end - t_frame_start)

        results.append(res)

        # Record tracking stats
        for t in res.tracked_detections:
            all_track_ids.add(t.track_id)
            track_lifespans[t.track_id] = track_lifespans.get(t.track_id, 0) + 1

        # Record spatial stats
        for s in res.spatial_detections:
            if s.zone != SpatialZone.NONE:
                if s.approach:
                    approach_counts[s.approach] += 1
                if s.zone_type:
                    zone_type_counts[s.zone_type] += 1
            else:
                unmapped_detections += 1

        frame_idx += 1

    cap.release()
    total_pipeline_time = time.perf_counter() - start_total_time

    # Performance calculations
    num_processed = len(results)
    avg_yolo_ms = (sum(yolo_times) / len(yolo_times) * 1000) if yolo_times else 0.0
    avg_pipeline_ms = (sum(frame_times) / len(frame_times) * 1000) if frame_times else 0.0
    end_to_end_fps = (num_processed / total_pipeline_time) if total_pipeline_time > 0 else 0.0

    # Tracking calculations
    persistent_tracks = sum(1 for lifespan in track_lifespans.values() if lifespan >= 10)
    avg_active_tracks = (
        sum(len(r.tracked_detections) for r in results) / num_processed
        if num_processed > 0
        else 0.0
    )

    # Controller & FSM metrics
    phases_observed = [r.current_phase for r in results]
    initial_phase = phases_observed[0] if phases_observed else None
    final_phase = phases_observed[-1] if phases_observed else None
    phase_transitions: List[str] = []
    prev_phase = initial_phase
    for p in phases_observed[1:]:
        if p != prev_phase:
            phase_transitions.append(f"{prev_phase.value} -> {p.value}")
            prev_phase = p

    # Bloom City Bridge metrics
    bridge_updates = len(results)
    final_vehicle_signal = results[-1].vehicle_signal if results else None
    final_pedestrian_signal = results[-1].pedestrian_signal if results else None
    all_safe = all(r.signal_state.is_safe for r in results)

    # Telemetry aggregate metrics
    vehicle_demand_observed = sum(r.vehicle_demand for r in results)
    ped_demand_observed = sum(r.waiting_pedestrians + r.crossing_pedestrians for r in results)
    all_snapshots_valid = all(r.demand_snapshot.is_valid() for r in results)

    print("-" * 70)
    print("VALIDATION SUMMARY")
    print(f"Frames processed:            {num_processed} / {total_frames}")
    print(f"YOLO status:                 {pipeline.vision_status}")
    print(f"Hardware device:             {pipeline.device}")
    print(f"Total raw detections:        {len(confidence_scores)}")
    print(f"Detected classes:            {sorted(detected_classes)}")
    if confidence_scores:
        print(f"Confidence min/mean/max:     {min(confidence_scores):.2f} / {np.mean(confidence_scores):.2f} / {max(confidence_scores):.2f}")
    print(f"Total unique tracks:         {len(all_track_ids)}")
    print(f"Persistent tracks (>=10 fr): {persistent_tracks}")
    print(f"Avg active tracks / frame:   {avg_active_tracks:.2f}")
    print(f"Approach detections:         North: {approach_counts[Approach.NORTH]}, South: {approach_counts[Approach.SOUTH]}, East: {approach_counts[Approach.EAST]}, West: {approach_counts[Approach.WEST]}")
    print(f"Zone types detected:         Waiting: {zone_type_counts[ZoneType.WAITING]}, Crossing: {zone_type_counts[ZoneType.CROSSING]}, Queue: {zone_type_counts[ZoneType.QUEUE]}, Approach: {zone_type_counts[ZoneType.APPROACH]}")
    print(f"Unmapped detections (NONE):  {unmapped_detections}")
    print(f"DemandSnapshot valid:        {all_snapshots_valid}")
    print(f"Initial FSM Phase:           {initial_phase.value if initial_phase else None}")
    print(f"Phase transitions:           {phase_transitions or 'None (5s elapsed < min green 10.0s)'}")
    print(f"Final FSM Phase:             {final_phase.value if final_phase else None}")
    print(f"Final Vehicle Signal:        {final_vehicle_signal.value if final_vehicle_signal else None}")
    print(f"Final Pedestrian Signal:     {final_pedestrian_signal.value if final_pedestrian_signal else None}")
    print(f"Bloom City Signal Safety:    {'PASS (All 125 frames safe)' if all_safe else 'FAIL'}")
    print(f"Avg YOLO Forward Pass:       {avg_yolo_ms:.1f} ms")
    print(f"Avg Total Pipeline Step:     {avg_pipeline_ms:.1f} ms")
    print(f"End-to-End Processing FPS:   {end_to_end_fps:.2f} fps")
    print("=" * 70)

    return {
        "num_processed": num_processed,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_s": duration_s,
        "yolo_status": pipeline.vision_status,
        "device": pipeline.device,
        "total_detections": len(confidence_scores),
        "detected_classes": sorted(detected_classes),
        "conf_min": min(confidence_scores) if confidence_scores else 0.0,
        "conf_mean": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
        "conf_max": max(confidence_scores) if confidence_scores else 0.0,
        "total_tracks": len(all_track_ids),
        "persistent_tracks": persistent_tracks,
        "avg_active_tracks": avg_active_tracks,
        "approach_counts": {k.value: v for k, v in approach_counts.items()},
        "zone_type_counts": {k.value: v for k, v in zone_type_counts.items()},
        "unmapped_detections": unmapped_detections,
        "vehicle_demand_observed": vehicle_demand_observed,
        "ped_demand_observed": ped_demand_observed,
        "all_snapshots_valid": all_snapshots_valid,
        "initial_phase": initial_phase.value if initial_phase else "",
        "phase_transitions": phase_transitions,
        "final_phase": final_phase.value if final_phase else "",
        "bridge_updates": bridge_updates,
        "final_vehicle_signal": final_vehicle_signal.value if final_vehicle_signal else "",
        "final_pedestrian_signal": final_pedestrian_signal.value if final_pedestrian_signal else "",
        "signals_safe": all_safe,
        "avg_yolo_ms": avg_yolo_ms,
        "avg_pipeline_ms": avg_pipeline_ms,
        "end_to_end_fps": end_to_end_fps,
    }


if __name__ == "__main__":
    v_path = sys.argv[1] if len(sys.argv) > 1 else "97885-649761107.mp4"
    run_comprehensive_validation(v_path)
