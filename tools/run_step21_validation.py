"""Step 21 Full Real-Video Adaptive Behavior Validation Script for CrossGuard.

Executes the complete closed-loop pipeline on tools/intersection_50s.mp4:
Video -> Vision (YOLOv8n) -> ByteTrack -> CameraCalibration (for_traffic_intersection_50s)
-> PerceptionAdapter -> Spatial Analysis -> TelemetryAggregator -> AdaptiveController
-> ControllerBridge -> Bloom City signal state.

Collects and records all metrics requested by CrossGuard Step 21.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from core.camera_calibration import CameraCalibration
from core.models import ControllerState, Decision, DemandSnapshot, PedestrianSignal, Phase, VehicleSignal
from core.pipeline import AdaptivePipeline, PipelineFrameResult
from core.spatial_analysis import Approach, SpatialZone, ZoneType, is_pedestrian, is_vehicle


def run_step21_validation(
    video_path: str = "tools/intersection_50s.mp4",
    output_csv: str = "tools/step21_frame_log.csv",
    output_json: str = "tools/step21_summary.json",
) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if fps > 0 else 0.0

    print("=" * 70, flush=True)
    print("CROSSGUARD STEP 21: FULL REAL-VIDEO ADAPTIVE BEHAVIOR VALIDATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Video Path:        {video_path}", flush=True)
    print(f"Resolution:        {width} x {height}", flush=True)
    print(f"FPS:               {fps:.3f}", flush=True)
    print(f"Total Frames:      {total_frames}", flush=True)
    print(f"Duration:          {duration_s:.2f} s", flush=True)
    print("-" * 70, flush=True)

    # 1. Instantiate Camera Calibration using NEW Step 20 camera-specific homography
    calib = CameraCalibration.for_traffic_intersection_50s(
        source_width=float(width),
        source_height=float(height),
    )

    # 2. Instantiate AdaptivePipeline with validated 50s camera calibration
    pipeline = AdaptivePipeline(
        default_fps=fps,
        calibration=calib,
    )

    print(f"Vision Backend:    {pipeline.vision_status} (device: {pipeline.device})", flush=True)
    print(f"YOLO Model:        {getattr(pipeline.vision.backend, 'model_name_or_path', 'yolov8n.pt')}", flush=True)
    print(f"Calibration:       ACTIVE (for_traffic_intersection_50s, is_valid: {calib.is_valid})", flush=True)
    print(f"Controller FSM:    ACTIVE (initial: {pipeline.controller.state.current_phase.value})", flush=True)
    print(f"Controller Bridge: ACTIVE (Bloom City)", flush=True)
    print("-" * 70, flush=True)

    # Metrics storage
    yolo_latencies: List[float] = []
    frame_latencies: List[float] = []

    detection_counts_by_class: Dict[str, int] = defaultdict(int)
    confidence_scores: List[float] = []
    total_raw_detections = 0
    total_pedestrian_detections = 0
    total_vehicle_detections = 0

    # Tracking metrics
    track_first_seen: Dict[int, int] = {}
    track_last_seen: Dict[int, int] = {}
    track_classes: Dict[int, str] = {}
    active_track_counts: List[int] = []

    # Pedestrian waiting tracking (track_id -> frame_first_waiting)
    waiting_peds_tracker: Dict[int, int] = {}
    waiting_peds_last_seen: Dict[int, int] = {}

    # Spatial metrics for Pedestrians (exact required zones)
    pedestrian_zone_counts: Dict[str, int] = {
        "NORTH_WAITING": 0,
        "SOUTH_WAITING": 0,
        "EAST_WAITING": 0,
        "WEST_WAITING": 0,
        "NORTH_CROSSING": 0,
        "SOUTH_CROSSING": 0,
        "EAST_CROSSING": 0,
        "WEST_CROSSING": 0,
        "NONE": 0,
    }

    # Spatial metrics for Vehicles
    vehicle_approach_counts: Dict[str, int] = {
        "NORTH_APPROACH": 0,
        "SOUTH_APPROACH": 0,
        "EAST_APPROACH": 0,
        "WEST_APPROACH": 0,
    }
    vehicle_queue_counts: Dict[str, int] = {
        "NORTH_QUEUE": 0,
        "SOUTH_QUEUE": 0,
        "EAST_QUEUE": 0,
        "WEST_QUEUE": 0,
    }
    vehicle_none_count = 0

    # Telemetry metrics
    demand_snapshots: List[DemandSnapshot] = []
    valid_snapshots_count = 0
    peak_waiting_ped = 0
    peak_crossing_ped = 0
    peak_queued_veh = 0
    peak_approaching_veh = 0
    approaches_with_demand: Set[str] = set()

    # FSM tracking
    fsm_phases: List[str] = []
    transitions: List[Dict[str, Any]] = []
    phase_durations: Dict[str, float] = defaultdict(float)
    max_observed_ped_wait = 0.0
    vehicle_protection_occurred = False
    max_green_occurred = False

    # Safety metrics
    safety_violations: List[str] = []

    # Bloom City Bridge metrics
    bridge_updates = 0
    signal_transitions = 0
    mismatches = 0
    prev_signal_state: Optional[Tuple[str, str]] = None

    frame_logs: List[Dict[str, Any]] = []

    start_wall_time = time.perf_counter()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp = frame_idx / fps if fps > 0 else 0.0

        # Measure YOLO inference latency separately
        t0_yolo = time.perf_counter()
        raw_dets = pipeline.vision.backend.detect(frame)
        t1_yolo = time.perf_counter()
        yolo_ms = (t1_yolo - t0_yolo) * 1000.0
        yolo_latencies.append(yolo_ms)

        total_raw_detections += len(raw_dets)
        for d in raw_dets:
            detection_counts_by_class[d.class_name] += 1
            confidence_scores.append(float(d.confidence))
            cls_lower = d.class_name.lower().strip()
            if is_pedestrian(cls_lower):
                total_pedestrian_detections += 1
            elif is_vehicle(cls_lower):
                total_vehicle_detections += 1

        # Determine longest waiting pedestrian from active tracking
        longest_wait_s = 0.0
        if waiting_peds_tracker:
            max_frames_waiting = max(frame_idx - start_f for start_f in waiting_peds_tracker.values())
            longest_wait_s = max_frames_waiting / fps if fps > 0 else 0.0

        # Execute full pipeline step
        t0_frame = time.perf_counter()
        res: PipelineFrameResult = pipeline.process_frame(
            frame,
            frame_index=frame_idx,
            dt=1.0 / fps,
            longest_pedestrian_wait_s=longest_wait_s,
        )
        t1_frame = time.perf_counter()
        frame_ms = (t1_frame - t0_frame) * 1000.0
        frame_latencies.append(frame_ms)

        # Update tracking records
        active_track_counts.append(len(res.tracked_detections))
        for t in res.tracked_detections:
            tid = t.track_id
            if tid not in track_first_seen:
                track_first_seen[tid] = frame_idx
                track_classes[tid] = t.class_name
            track_last_seen[tid] = frame_idx

        # Update spatial classification metrics
        current_waiting_tracks_in_frame: Set[int] = set()
        frame_mapped = 0
        frame_unmapped = 0

        for s in res.spatial_detections:
            cls_lower = s.class_name.lower().strip()
            zone_str = s.zone.value if s.zone else "NONE"

            if is_pedestrian(cls_lower):
                if zone_str in pedestrian_zone_counts:
                    pedestrian_zone_counts[zone_str] += 1
                else:
                    pedestrian_zone_counts["NONE"] += 1

                if s.zone_type == ZoneType.WAITING:
                    current_waiting_tracks_in_frame.add(s.track_id)
            elif is_vehicle(cls_lower):
                if zone_str in vehicle_approach_counts:
                    vehicle_approach_counts[zone_str] += 1
                elif zone_str in vehicle_queue_counts:
                    vehicle_queue_counts[zone_str] += 1
                else:
                    vehicle_none_count += 1

            if s.zone != SpatialZone.NONE:
                frame_mapped += 1
                if s.approach:
                    approaches_with_demand.add(s.approach.value)
            else:
                frame_unmapped += 1

        # Maintain pedestrian wait tracker with 5-frame grace period for tracking continuity
        for tid in current_waiting_tracks_in_frame:
            if tid not in waiting_peds_tracker:
                waiting_peds_tracker[tid] = frame_idx
            waiting_peds_last_seen[tid] = frame_idx

        for tid in list(waiting_peds_tracker.keys()):
            if frame_idx - waiting_peds_last_seen.get(tid, 0) > 5:
                del waiting_peds_tracker[tid]
                del waiting_peds_last_seen[tid]

        # Telemetry metrics
        ds = res.demand_snapshot
        demand_snapshots.append(ds)
        if ds.is_valid():
            valid_snapshots_count += 1

        if ds.pedestrians_waiting > peak_waiting_ped:
            peak_waiting_ped = ds.pedestrians_waiting
        if ds.pedestrians_in_crosswalk > peak_crossing_ped:
            peak_crossing_ped = ds.pedestrians_in_crosswalk
        if ds.vehicle_queue > peak_queued_veh:
            peak_queued_veh = ds.vehicle_queue
        if res.approaching_vehicles > peak_approaching_veh:
            peak_approaching_veh = res.approaching_vehicles
        if ds.longest_pedestrian_wait_s > max_observed_ped_wait:
            max_observed_ped_wait = ds.longest_pedestrian_wait_s

        # Check vehicle protection and max-green in controller decision reason
        reason_str = res.controller_state.decision_reason
        if "minimum vehicle green protects" in reason_str:
            vehicle_protection_occurred = True
        if "maximum vehicle green" in reason_str:
            max_green_occurred = True

        # FSM and Phase tracking
        current_phase = res.current_phase.value
        fsm_phases.append(current_phase)
        phase_durations[current_phase] += (1.0 / fps)

        if frame_idx > 0 and fsm_phases[-1] != fsm_phases[-2]:
            prev_p = fsm_phases[-2]
            new_p = fsm_phases[-1]
            trans_info = {
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "from_phase": prev_p,
                "to_phase": new_p,
                "reason": res.controller_state.decision_reason,
                "waiting_pedestrians": ds.pedestrians_waiting,
                "longest_ped_wait_s": round(ds.longest_pedestrian_wait_s, 2),
                "vehicle_demand": ds.vehicle_demand,
            }
            transitions.append(trans_info)
            print(f"  >>> FSM TRANSITION at frame {frame_idx} (t={timestamp:.2f}s): {prev_p} -> {new_p}", flush=True)
            print(f"      Reason: {trans_info['reason']}", flush=True)

        # Safety verification
        veh_sig = res.vehicle_signal
        ped_sig = res.pedestrian_signal

        # Invariant 1: No vehicle green simultaneous with pedestrian walk/clearance
        if veh_sig == VehicleSignal.GREEN and ped_sig in (PedestrianSignal.WALK, PedestrianSignal.CLEARING):
            safety_violations.append(f"Frame {frame_idx}: Green vehicle + Walking/Clearing ped")

        # Invariant 2: Vehicle yellow implies pedestrian dont-walk
        if veh_sig == VehicleSignal.YELLOW and ped_sig != PedestrianSignal.DONT_WALK:
            safety_violations.append(f"Frame {frame_idx}: Yellow vehicle + non-DONT_WALK ped")

        # Invariant 3: All-red implies vehicle red + pedestrian dont-walk
        if res.current_phase == Phase.ALL_RED_CLEARANCE:
            if veh_sig != VehicleSignal.RED or ped_sig != PedestrianSignal.DONT_WALK:
                safety_violations.append(f"Frame {frame_idx}: All-red mismatch")

        # Bloom City Bridge updates
        bridge_updates += 1
        curr_sig = (veh_sig.value, ped_sig.value)
        if prev_signal_state is not None and curr_sig != prev_signal_state:
            signal_transitions += 1
        prev_signal_state = curr_sig

        # Bridge match check
        if res.signal_state.vehicle_signal != res.controller_state.vehicle_signal:
            mismatches += 1
        if res.signal_state.pedestrian_signal != res.controller_state.pedestrian_signal:
            mismatches += 1

        # Frame log row
        frame_logs.append({
            "frame_index": frame_idx,
            "timestamp": round(timestamp, 3),
            "yolo_dets": len(raw_dets),
            "active_tracks": len(res.tracked_detections),
            "mapped": frame_mapped,
            "unmapped": frame_unmapped,
            "ped_waiting": ds.pedestrians_waiting,
            "ped_crossing": ds.pedestrians_in_crosswalk,
            "veh_approach": res.approaching_vehicles,
            "veh_queue": ds.vehicle_queue,
            "veh_demand": ds.vehicle_demand,
            "longest_ped_wait": round(ds.longest_pedestrian_wait_s, 2),
            "phase": current_phase,
            "decision": res.controller_state.decision.value,
            "decision_reason": res.controller_state.decision_reason,
            "veh_signal": veh_sig.value,
            "ped_signal": ped_sig.value,
            "yolo_ms": round(yolo_ms, 2),
            "frame_ms": round(frame_ms, 2),
        })

        if frame_idx % 100 == 0 or frame_idx == total_frames - 1:
            elapsed = time.perf_counter() - start_wall_time
            proc_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  [Frame {frame_idx:4d}/{total_frames}] {proc_fps:.2f} proc FPS | "
                  f"tracks={len(res.tracked_detections)} | mapped={frame_mapped} | "
                  f"phase={current_phase} | decision={res.controller_state.decision.value}", flush=True)

        frame_idx += 1

    cap.release()
    total_wall_time = time.perf_counter() - start_wall_time

    # Compute track durations
    veh_track_durations = []
    ped_track_durations = []
    unique_veh_tracks = 0
    unique_ped_tracks = 0

    for tid, cls_name in track_classes.items():
        dur_frames = track_last_seen[tid] - track_first_seen[tid] + 1
        dur_s = dur_frames / fps if fps > 0 else 0.0
        cls_norm = cls_name.lower().strip()
        if is_vehicle(cls_norm):
            unique_veh_tracks += 1
            veh_track_durations.append(dur_s)
        elif is_pedestrian(cls_norm):
            unique_ped_tracks += 1
            ped_track_durations.append(dur_s)

    # Performance calculations
    avg_yolo_ms = float(np.mean(yolo_latencies)) if yolo_latencies else 0.0
    avg_frame_ms = float(np.mean(frame_latencies)) if frame_latencies else 0.0
    end_to_end_fps = frame_idx / total_wall_time if total_wall_time > 0 else 0.0
    avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
    avg_active_tracks = float(np.mean(active_track_counts)) if active_track_counts else 0.0

    valid_snapshots_pct = (valid_snapshots_count / frame_idx * 100.0) if frame_idx > 0 else 0.0

    # Write CSV
    if output_csv and frame_logs:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(frame_logs[0].keys()))
            writer.writeheader()
            writer.writerows(frame_logs)
        print(f"Wrote frame log CSV: {output_csv}", flush=True)

    summary_data = {
        "video": {
            "path": video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": frame_idx,
            "duration_s": round(duration_s, 2),
        },
        "vision": {
            "status": pipeline.vision_status,
            "model": getattr(pipeline.vision.backend, "model_name_or_path", "yolov8n.pt"),
            "device": pipeline.device,
            "total_raw_detections": total_raw_detections,
            "pedestrian_detections": total_pedestrian_detections,
            "vehicle_detections": total_vehicle_detections,
            "detection_counts_by_class": dict(detection_counts_by_class),
            "average_confidence": round(avg_confidence, 4),
            "average_yolo_ms": round(avg_yolo_ms, 2),
        },
        "tracking": {
            "unique_vehicle_tracks": unique_veh_tracks,
            "unique_pedestrian_tracks": unique_ped_tracks,
            "longest_pedestrian_track_s": round(max(ped_track_durations, default=0.0), 2),
            "longest_vehicle_track_s": round(max(veh_track_durations, default=0.0), 2),
            "average_active_tracks_per_frame": round(avg_active_tracks, 2),
        },
        "calibration": {
            "factory": "for_traffic_intersection_50s",
            "is_valid": calib.is_valid,
            "source_resolution": [width, height],
            "target_resolution": [1000, 800],
        },
        "spatial": {
            "pedestrian_zones": pedestrian_zone_counts,
            "vehicle_approach_zones": vehicle_approach_counts,
            "vehicle_queue_zones": vehicle_queue_counts,
            "vehicle_none": vehicle_none_count,
        },
        "telemetry": {
            "total_snapshots": len(demand_snapshots),
            "valid_snapshots_pct": round(valid_snapshots_pct, 2),
            "peak_waiting_pedestrians": peak_waiting_ped,
            "peak_crossing_pedestrians": peak_crossing_ped,
            "peak_queued_vehicles": peak_queued_veh,
            "peak_approaching_vehicles": peak_approaching_veh,
            "approaches_with_demand": sorted(list(approaches_with_demand)),
        },
        "fsm": {
            "initial_state": fsm_phases[0] if fsm_phases else "UNKNOWN",
            "final_state": fsm_phases[-1] if fsm_phases else "UNKNOWN",
            "total_transitions": len(transitions),
            "transitions": transitions,
            "phase_durations": {k: round(v, 2) for k, v in phase_durations.items()},
            "longest_pedestrian_wait_s": round(max_observed_ped_wait, 2),
            "pedestrian_priority_triggered": any(t["to_phase"] == "VEHICLE_YELLOW" for t in transitions),
            "vehicle_protection_occurred": vehicle_protection_occurred,
            "max_green_occurred": max_green_occurred,
            "safety_violations_count": len(safety_violations),
            "safety_violations": safety_violations,
        },
        "bloom_city": {
            "bridge_updates": bridge_updates,
            "signal_transitions": signal_transitions,
            "fsm_bloom_mismatches": mismatches,
            "safety_violations_count": len(safety_violations),
        },
        "performance": {
            "avg_yolo_ms": round(avg_yolo_ms, 2),
            "avg_frame_ms": round(avg_frame_ms, 2),
            "end_to_end_fps": round(end_to_end_fps, 2),
            "total_wall_time_s": round(total_wall_time, 2),
        },
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Wrote summary JSON: {output_json}", flush=True)

    print("=" * 70, flush=True)
    print("STEP 21 VALIDATION COMPLETE", flush=True)
    print("=" * 70, flush=True)
    return summary_data


if __name__ == "__main__":
    run_step21_validation()
