"""End-to-end Adaptive Traffic Management Pipeline for CrossGuard (Step 7).

Orchestrates the complete perception-to-control lifecycle:
  Video Frame
      ↓
  VisionPipeline (YOLOv8n + ByteTrack)
      ↓
  TrackedDetection[]
      ↓
  PerceptionAdapter
      ↓
  SpatialDetection[]
      ↓
  TelemetryAggregator
      ↓
  DemandSnapshot
      ↓
  AdaptiveController
      ↓
  ControllerBridge
      ↓
  BloomCitySignalState

Pure Python; no UI dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.adaptive_controller import AdaptiveController
from core.camera_calibration import CameraCalibration
from core.models import ControllerState, DemandSnapshot, PedestrianSignal, Phase, VehicleSignal
from core.perception_adapter import PerceptionAdapter, SpatialDetection
from core.telemetry import TelemetryAggregator, TelemetrySnapshot
from simulation.controller_bridge import BloomCitySignalState, ControllerBridge
from vision.models import DeviceType, TrackedDetection, VisionPipelineConfig, is_cuda_available, resolve_device
from vision.overlay import draw_perception_overlay
from vision.vision_pipeline import VisionPipeline


@dataclass(frozen=True)
class PipelineFrameResult:
    """Structured result produced for each processed video frame."""
    frame_index: int
    timestamp: float
    dt: float
    tracked_detections: List[TrackedDetection]
    spatial_detections: List[SpatialDetection]
    telemetry_snapshot: TelemetrySnapshot
    demand_snapshot: DemandSnapshot
    controller_state: ControllerState
    signal_state: BloomCitySignalState
    annotated_frame: Optional[np.ndarray] = None

    @property
    def num_tracked_objects(self) -> int:
        return len(self.tracked_detections)

    @property
    def waiting_pedestrians(self) -> int:
        return self.demand_snapshot.pedestrians_waiting

    @property
    def crossing_pedestrians(self) -> int:
        return self.demand_snapshot.pedestrians_in_crosswalk

    @property
    def vehicle_demand(self) -> int:
        return self.demand_snapshot.vehicle_demand

    @property
    def queued_vehicles(self) -> int:
        return self.demand_snapshot.vehicle_queue

    @property
    def approaching_vehicles(self) -> int:
        return self.telemetry_snapshot.approaching_vehicles

    @property
    def current_phase(self) -> Phase:
        return self.controller_state.current_phase

    @property
    def vehicle_signal(self) -> VehicleSignal:
        return self.signal_state.vehicle_signal

    @property
    def pedestrian_signal(self) -> PedestrianSignal:
        return self.signal_state.pedestrian_signal

    def to_dict(self) -> Dict[str, Any]:
        """Lightweight serializable summary for observability and debugging."""
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "dt": round(self.dt, 3),
            "num_tracked": self.num_tracked_objects,
            "waiting_pedestrians": self.waiting_pedestrians,
            "crossing_pedestrians": self.crossing_pedestrians,
            "approaching_vehicles": self.approaching_vehicles,
            "queued_vehicles": self.queued_vehicles,
            "vehicle_demand": self.vehicle_demand,
            "phase": self.current_phase.value,
            "vehicle_signal": self.vehicle_signal.value,
            "pedestrian_signal": self.pedestrian_signal.value,
            "decision": self.controller_state.decision.value,
            "decision_reason": self.controller_state.decision_reason,
            "time_remaining": round(self.controller_state.time_remaining, 1),
            "fail_safe_active": self.controller_state.fail_safe_active,
        }


class AdaptivePipeline:
    """Orchestrates perception, spatial analysis, telemetry, FSM, and Bloom City bridge."""

    def __init__(
        self,
        vision_pipeline: Optional[VisionPipeline] = None,
        perception_adapter: Optional[PerceptionAdapter] = None,
        telemetry_aggregator: Optional[TelemetryAggregator] = None,
        controller: Optional[AdaptiveController] = None,
        controller_bridge: Optional[ControllerBridge] = None,
        default_fps: float = 10.0,
        source_resolution: Optional[Tuple[float, float]] = None,
        calibration: Optional[CameraCalibration] = None,
        device: Union[str, DeviceType] = DeviceType.AUTO,
        gpu_worker_url: Optional[str] = None,
        gpu_worker_client: Optional[Any] = None,
    ) -> None:
        self.gpu_worker_url = gpu_worker_url
        self.gpu_worker_client = gpu_worker_client

        if vision_pipeline is not None:
            self.vision = vision_pipeline
        else:
            self.vision = self._create_vision_pipeline(
                device=device,
                gpu_worker_url=gpu_worker_url,
                gpu_worker_client=gpu_worker_client,
            )
        self.adapter = perception_adapter or PerceptionAdapter(calibration=calibration)
        if calibration is not None:
            self.adapter.set_calibration(calibration)
        if source_resolution is not None and len(source_resolution) == 2:
            self.adapter.set_source_resolution(source_resolution[0], source_resolution[1])
        self.telemetry = telemetry_aggregator or TelemetryAggregator(source_name="vision")
        self.controller = controller or AdaptiveController()
        self.bridge = controller_bridge or ControllerBridge()
        self.default_fps = max(1.0, float(default_fps))
        self.current_time: float = 0.0
        self.last_result: Optional[PipelineFrameResult] = None

    @staticmethod
    def _create_vision_pipeline(
        device: Union[str, DeviceType],
        gpu_worker_url: Optional[str] = None,
        gpu_worker_client: Optional[Any] = None,
    ) -> VisionPipeline:
        req_str = device.value.lower() if isinstance(device, DeviceType) else str(device).strip().lower()

        # 1. Explicit CPU
        if req_str == "cpu":
            return VisionPipeline(config=VisionPipelineConfig(device=DeviceType.CPU))

        def _resolve_client():
            if gpu_worker_client is not None:
                return gpu_worker_client
            from core.gpu_config import get_configured_worker_url
            url = gpu_worker_url or get_configured_worker_url()
            if url and url.strip():
                from gpu_worker.client import GPUWorkerClient
                return GPUWorkerClient(base_url=url.strip())
            return None

        # 2. Explicit GPU (CUDA)
        if req_str in ("cuda", "gpu", "cuda:0"):
            client = _resolve_client()
            if client is not None:
                from gpu_worker.backend import RemoteGPUBackend
                try:
                    backend = RemoteGPUBackend(client=client, verify_on_init=True)
                except Exception as e:
                    raise RuntimeError(
                        f"Remote CUDA worker is not currently reachable. Failed to connect: {e}"
                    ) from e
                return VisionPipeline(
                    config=VisionPipelineConfig(device=DeviceType.CUDA),
                    backend=backend,
                )
            elif is_cuda_available():
                return VisionPipeline(config=VisionPipelineConfig(device=DeviceType.CUDA))
            else:
                raise RuntimeError(
                    "Remote CUDA worker is not currently reachable. (No remote GPU worker configured and local CUDA unavailable)."
                )

        # 3. Auto
        client = _resolve_client()
        if client is not None:
            try:
                from gpu_worker.backend import RemoteGPUBackend
                backend = RemoteGPUBackend(client=client, verify_on_init=True)
                return VisionPipeline(
                    config=VisionPipelineConfig(device=DeviceType.CUDA),
                    backend=backend,
                )
            except Exception:
                pass

        if is_cuda_available():
            return VisionPipeline(config=VisionPipelineConfig(device=DeviceType.CUDA))

        return VisionPipeline(config=VisionPipelineConfig(device=DeviceType.CPU))

    @property
    def calibration(self) -> Optional[CameraCalibration]:
        """Active camera calibration on the perception adapter, if any."""
        return self.adapter.calibration

    def set_calibration(self, calibration: Optional[CameraCalibration]) -> None:
        """Configure or clear camera homography calibration."""
        self.adapter.set_calibration(calibration)

    @property
    def device(self) -> str:
        """Effective hardware device from the vision pipeline ('cpu' or 'cuda')."""
        return self.vision.device

    @property
    def requested_device(self) -> str:
        """Requested device specification from the vision pipeline ('auto', 'cpu', or 'cuda')."""
        return getattr(self.vision, "requested_device", "auto")

    @property
    def execution_mode(self) -> str:
        """Observable execution mode: 'Remote CUDA', 'Local CUDA', or 'Local CPU'."""
        backend = getattr(self.vision, "backend", None)
        if backend is not None and type(backend).__name__ == "RemoteGPUBackend":
            return "Remote CUDA"
        elif self.device == "cuda":
            return "Local CUDA"
        return "Local CPU"

    @property
    def gpu_name(self) -> Optional[str]:
        """Hardware GPU model name if executing on GPU, else None."""
        return getattr(self.vision, "gpu_name", None)

    @property
    def last_inference_time_ms(self) -> float:
        """Measured inference latency in ms from the most recent frame."""
        return getattr(self.vision, "last_inference_time_ms", 0.0)

    @property
    def vision_status(self) -> str:
        """Observable status of vision perception: 'ACTIVE', 'UNAVAILABLE', or 'MOCK'."""
        return self.vision.status

    @property
    def vision_load_error(self) -> Optional[str]:
        """Description of vision backend load failure, if any."""
        return self.vision.load_error

    def reset(self) -> None:
        """Reset internal pipeline state (tracker, controller, and clock)."""
        self.vision.reset_tracking()
        self.controller.reset()
        self.current_time = 0.0
        self.last_result = None

    def process_frame(
        self,
        frame: Any,
        frame_index: int = 0,
        dt: Optional[float] = None,
        longest_pedestrian_wait_s: float = 0.0,
    ) -> PipelineFrameResult:
        """Process a single video frame through the complete pipeline.

        Flow:
          Frame -> Vision (YOLO + ByteTrack)
                -> Perception Adapter
                -> Telemetry Aggregator
                -> Adaptive FSM
                -> Bloom City Signal State
        """
        step_dt = float(dt) if dt is not None and dt > 0 else (1.0 / self.default_fps)

        # Auto-detect camera frame resolution if frame has 2D shape, transformer is currently at default identity, and calibration is NOT active
        if hasattr(frame, "shape") and len(frame.shape) >= 2:
            fh, fw = frame.shape[:2]
            if (
                (self.adapter.calibration is None or not self.adapter.calibration.is_calibrated)
                and self.adapter.transformer.is_identity
                and (float(fw), float(fh)) != (self.adapter.transformer.target_width, self.adapter.transformer.target_height)
            ):
                self.adapter.set_source_resolution(float(fw), float(fh))

        # 1. Vision: Video Frame -> TrackedDetection[]
        tracked = self.vision.process_frame(frame, frame_index=frame_index)

        # 2. Perception Adapter: TrackedDetection[] -> SpatialDetection[]
        spatial = self.adapter.adapt(tracked)

        # 3. Telemetry: SpatialDetection[] -> DemandSnapshot
        telemetry_snap = self.telemetry.aggregate(
            spatial,
            frame_index=frame_index,
            longest_pedestrian_wait_s=longest_pedestrian_wait_s,
        )
        demand = telemetry_snap.demand

        # 4. Adaptive FSM: DemandSnapshot -> ControllerState
        ctrl_state = self.controller.step(dt=step_dt, demand=demand)

        # 5. Bridge: ControllerState -> BloomCitySignalState
        signal_state = self.bridge.map_state(ctrl_state)

        # Update clock
        self.current_time += step_dt

        # 6. Perception Overlay: Render visual bounding boxes & telemetry labels
        annotated = None
        if frame is not None and isinstance(frame, np.ndarray) and frame.ndim >= 2:
            annotated = draw_perception_overlay(
                frame,
                spatial_detections=spatial,
                tracked_detections=tracked,
            )

        result = PipelineFrameResult(
            frame_index=frame_index,
            timestamp=self.current_time,
            dt=step_dt,
            tracked_detections=tracked,
            spatial_detections=spatial,
            telemetry_snapshot=telemetry_snap,
            demand_snapshot=demand,
            controller_state=ctrl_state,
            signal_state=signal_state,
            annotated_frame=annotated,
        )
        self.last_result = result
        return result

    def get_last_frame_diagnostics(self) -> Dict[str, Any]:
        """Return end-to-end diagnostic instrumentation for the most recent frame."""
        res = self.last_result
        raw = getattr(self.vision, "last_raw_detections", [])
        filtered = getattr(self.vision, "last_filtered_detections", [])
        raw_classes: Dict[str, int] = {}
        for d in raw:
            raw_classes[d.class_name] = raw_classes.get(d.class_name, 0) + 1

        return {
            "requested_device": self.requested_device,
            "actual_device": self.device,
            "execution_mode": self.execution_mode,
            "vision_backend": type(self.vision.backend).__name__,
            "vision_status": self.vision_status,
            "yolo_loaded": self.vision.is_loaded,
            "raw_detection_count": len(raw),
            "raw_class_counts": raw_classes,
            "bytetrack_input_count": len(filtered),
            "tracked_detection_count": len(res.tracked_detections) if res else 0,
            "spatial_detection_count": len(res.spatial_detections) if res else 0,
            "waiting_pedestrians": res.waiting_pedestrians if res else 0,
            "crossing_pedestrians": res.crossing_pedestrians if res else 0,
            "approaching_vehicles": res.approaching_vehicles if res else 0,
            "queued_vehicles": res.queued_vehicles if res else 0,
            "controller_phase": res.current_phase.value if res else "N/A",
        }

    def process_sequence(
        self,
        frames: Iterable[Any],
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[PipelineFrameResult]:
        """Process an iterable or generator of frames sequentially."""
        effective_fps = float(fps) if fps is not None and fps > 0 else self.default_fps
        step_dt = 1.0 / effective_fps
        results: List[PipelineFrameResult] = []

        for idx, frame in enumerate(frames):
            if max_frames is not None and idx >= max_frames:
                break
            res = self.process_frame(frame, frame_index=idx, dt=step_dt)
            results.append(res)

        return results

    def process_video_file(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
    ) -> List[PipelineFrameResult]:
        """Process a local video file sequentially if OpenCV is available."""
        try:
            import cv2
        except ImportError:
            raise RuntimeError(
                "OpenCV (cv2) is required to process video files directly. "
                "Install opencv-python or pass pre-decoded frames to process_sequence()."
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file at: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        src_w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if src_w > 0 and src_h > 0 and self.adapter.transformer.is_identity:
            if (src_w, src_h) != (self.adapter.transformer.target_width, self.adapter.transformer.target_height):
                self.adapter.set_source_resolution(src_w, src_h)

        fps = target_fps if target_fps is not None and target_fps > 0 else (src_fps if src_fps > 0 else self.default_fps)
        step_dt = 1.0 / fps

        results: List[PipelineFrameResult] = []
        frame_idx = 0

        try:
            while cap.isOpened():
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                res = self.process_frame(frame, frame_index=frame_idx, dt=step_dt)
                results.append(res)
                frame_idx += 1
        finally:
            cap.release()

        return results
