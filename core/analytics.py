"""Analytics & Research Evaluation Session for CrossGuard.

Consumes stream of PipelineFrameResult objects produced by AdaptivePipeline.
Accumulates actual observed demand, controller phase transitions, execution throughput,
and safety invariant checks. Pure Python; no UI dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.models import Decision, PedestrianSignal, Phase, VehicleSignal
from core.pipeline import PipelineFrameResult
from core.spatial_analysis import Approach


@dataclass(frozen=True)
class PhaseTransitionEvent:
    """Record of a discrete FSM phase transition."""
    timestamp: float
    frame_index: int
    from_phase: Phase
    to_phase: Phase
    reason: str
    waiting_pedestrians: int
    crossing_pedestrians: int
    vehicle_demand: int
    queued_vehicles: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 2),
            "frame_index": self.frame_index,
            "from_phase": self.from_phase.value,
            "to_phase": self.to_phase.value,
            "reason": self.reason,
            "waiting_pedestrians": self.waiting_pedestrians,
            "crossing_pedestrians": self.crossing_pedestrians,
            "vehicle_demand": self.vehicle_demand,
            "queued_vehicles": self.queued_vehicles,
        }


@dataclass
class ApproachPeakDemand:
    """Peak observed traffic and pedestrian counts for a single approach."""
    approach: Approach
    waiting_pedestrians: int = 0
    crossing_pedestrians: int = 0
    approaching_vehicles: int = 0
    queued_vehicles: int = 0

    def update(
        self,
        waiting: int,
        crossing: int,
        approaching: int,
        queued: int,
    ) -> None:
        if waiting > self.waiting_pedestrians:
            self.waiting_pedestrians = waiting
        if crossing > self.crossing_pedestrians:
            self.crossing_pedestrians = crossing
        if approaching > self.approaching_vehicles:
            self.approaching_vehicles = approaching
        if queued > self.queued_vehicles:
            self.queued_vehicles = queued

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approach": self.approach.value,
            "waiting_pedestrians": self.waiting_pedestrians,
            "crossing_pedestrians": self.crossing_pedestrians,
            "approaching_vehicles": self.approaching_vehicles,
            "queued_vehicles": self.queued_vehicles,
        }


class AnalyticsSession:
    """Aggregates and computes session-level research metrics from PipelineFrameResult.

    Consumes pipeline outputs sequentially. Zero inference recalculation.
    Zero fabricated data points.
    """

    def __init__(
        self,
        video_name: str = "",
        video_duration_s: float = 0.0,
        device: str = "CPU",
        vision_status: str = "ACTIVE",
        gpu_name: Optional[str] = None,
    ) -> None:
        self.video_name = video_name
        self.video_duration_s = float(video_duration_s)
        self.device = str(device)
        self.vision_status = str(vision_status)
        self.gpu_name = gpu_name
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated session metrics to empty state."""
        self.frames_processed: int = 0
        self.total_processing_time_s: float = 0.0
        self.yolo_latency_total_s: float = 0.0
        self.yolo_latency_count: int = 0

        # Time series telemetry
        self.timestamps: List[float] = []
        self.frame_indices: List[int] = []
        self.waiting_pedestrians: List[int] = []
        self.crossing_pedestrians: List[int] = []
        self.approaching_vehicles: List[int] = []
        self.queued_vehicles: List[int] = []
        self.vehicle_demand: List[int] = []
        self.phases: List[Phase] = []
        self.decisions: List[Decision] = []
        self.pedestrian_waits: List[float] = []

        # Controller phase durations (seconds)
        self.phase_durations: Dict[Phase, float] = {p: 0.0 for p in Phase}
        self.phase_frame_counts: Dict[Phase, int] = {p: 0 for p in Phase}

        # Transitions
        self.transitions: List[PhaseTransitionEvent] = []
        self._last_phase: Optional[Phase] = None

        # Spatial approach peaks
        self.approach_peaks: Dict[Approach, ApproachPeakDemand] = {
            app: ApproachPeakDemand(approach=app) for app in Approach
        }

        # Safety invariant checks
        self.safety_violations: int = 0
        self.green_walk_conflicts: int = 0
        self.total_safety_checks: int = 0

    @property
    def is_empty(self) -> bool:
        """True if no frames have been analyzed in this session."""
        return self.frames_processed == 0

    @property
    def analyzed_duration_s(self) -> float:
        """Total duration of analyzed video segment in seconds."""
        if not self.timestamps:
            return 0.0
        return sum(self.phase_durations.values())

    @property
    def measured_fps(self) -> float:
        """Measured throughput in processed frames per second."""
        if self.total_processing_time_s <= 0 or self.frames_processed == 0:
            return 0.0
        return self.frames_processed / self.total_processing_time_s

    @property
    def avg_frame_latency_ms(self) -> float:
        """Average end-to-end processing latency per frame in milliseconds."""
        if self.frames_processed == 0:
            return 0.0
        return (self.total_processing_time_s / self.frames_processed) * 1000.0

    @property
    def avg_yolo_latency_ms(self) -> Optional[float]:
        """Average YOLO inference latency in milliseconds if measured."""
        if self.yolo_latency_count == 0:
            return None
        return (self.yolo_latency_total_s / self.yolo_latency_count) * 1000.0

    @property
    def peak_waiting_pedestrians(self) -> int:
        return max(self.waiting_pedestrians, default=0)

    @property
    def peak_crossing_pedestrians(self) -> int:
        return max(self.crossing_pedestrians, default=0)

    @property
    def peak_approaching_vehicles(self) -> int:
        return max(self.approaching_vehicles, default=0)

    @property
    def peak_queued_vehicles(self) -> int:
        return max(self.queued_vehicles, default=0)

    @property
    def max_pedestrian_wait_s(self) -> float:
        return max(self.pedestrian_waits, default=0.0)

    @property
    def initial_phase(self) -> Optional[Phase]:
        return self.phases[0] if self.phases else None

    @property
    def final_phase(self) -> Optional[Phase]:
        return self.phases[-1] if self.phases else None

    @property
    def num_transitions(self) -> int:
        return len(self.transitions)

    @property
    def pedestrian_priority_activations(self) -> int:
        """Count of transitions triggered by or serving pedestrian priority."""
        return sum(
            1
            for t in self.transitions
            if t.to_phase in (Phase.VEHICLE_YELLOW, Phase.PEDESTRIAN_WALK)
            and ("pedestrian" in t.reason.lower() or "walk" in t.reason.lower())
        )

    @property
    def phase_distribution_pct(self) -> Dict[Phase, float]:
        """Percentage of total analyzed time spent in each phase."""
        total = sum(self.phase_durations.values())
        if total <= 0:
            return {p: 0.0 for p in Phase}
        return {p: (dur / total) * 100.0 for p, dur in self.phase_durations.items()}

    def record_frame(
        self,
        res: PipelineFrameResult,
        frame_proc_time_s: float = 0.0,
        yolo_latency_s: Optional[float] = None,
    ) -> None:
        """Record telemetry, controller state, and timing from a single PipelineFrameResult."""
        if res is None:
            return

        self.frames_processed += 1
        self.total_processing_time_s += max(0.0, float(frame_proc_time_s))

        if yolo_latency_s is not None and yolo_latency_s > 0:
            self.yolo_latency_total_s += float(yolo_latency_s)
            self.yolo_latency_count += 1

        # Track sequential time series
        ts = float(res.timestamp)
        self.timestamps.append(ts)
        self.frame_indices.append(int(res.frame_index))
        self.waiting_pedestrians.append(int(res.waiting_pedestrians))
        self.crossing_pedestrians.append(int(res.crossing_pedestrians))
        self.approaching_vehicles.append(int(res.approaching_vehicles))
        self.queued_vehicles.append(int(res.queued_vehicles))
        self.vehicle_demand.append(int(res.vehicle_demand))

        current_phase = res.current_phase
        self.phases.append(current_phase)
        self.decisions.append(res.controller_state.decision)

        ped_wait = float(res.demand_snapshot.longest_pedestrian_wait_s)
        self.pedestrian_waits.append(ped_wait)

        # Accumulate phase duration
        dt_val = float(res.dt) if res.dt > 0 else 0.0417
        self.phase_durations[current_phase] += dt_val
        self.phase_frame_counts[current_phase] += 1

        # Detect discrete FSM phase transitions
        if self._last_phase is not None and self._last_phase != current_phase:
            event = PhaseTransitionEvent(
                timestamp=ts,
                frame_index=res.frame_index,
                from_phase=self._last_phase,
                to_phase=current_phase,
                reason=res.controller_state.decision_reason,
                waiting_pedestrians=res.waiting_pedestrians,
                crossing_pedestrians=res.crossing_pedestrians,
                vehicle_demand=res.vehicle_demand,
                queued_vehicles=res.queued_vehicles,
            )
            self.transitions.append(event)

        self._last_phase = current_phase

        # Spatial approach peak demand updates
        if res.telemetry_snapshot and res.telemetry_snapshot.by_approach:
            for app, app_demand in res.telemetry_snapshot.by_approach.items():
                if app in self.approach_peaks:
                    self.approach_peaks[app].update(
                        waiting=app_demand.waiting_pedestrians,
                        crossing=app_demand.crossing_pedestrians,
                        approaching=app_demand.approaching_vehicles,
                        queued=app_demand.queued_vehicles,
                    )

        # Safety Invariant Checks
        self.total_safety_checks += 1

        # Invariant 1: Vehicle GREEN and Pedestrian WALK must NEVER occur concurrently
        v_sig = res.vehicle_signal
        p_sig = res.pedestrian_signal
        ctrl_v = res.controller_state.vehicle_signal if res.controller_state else v_sig
        ctrl_p = res.controller_state.pedestrian_signal if res.controller_state else p_sig

        if (
            (v_sig == VehicleSignal.GREEN and p_sig == PedestrianSignal.WALK)
            or (ctrl_v == VehicleSignal.GREEN and ctrl_p == PedestrianSignal.WALK)
        ):
            self.green_walk_conflicts += 1
            self.safety_violations += 1

        # Invariant 2: Explicit bridge signal safety
        if res.signal_state and not res.signal_state.is_safe:
            if not ((v_sig == VehicleSignal.GREEN and p_sig == PedestrianSignal.WALK)
                    or (ctrl_v == VehicleSignal.GREEN and ctrl_p == PedestrianSignal.WALK)):
                self.safety_violations += 1

    def to_executive_summary(self) -> Dict[str, Any]:
        """Return executive summary dictionary for dashboard display."""
        return {
            "is_empty": self.is_empty,
            "video_name": self.video_name,
            "frames_processed": self.frames_processed,
            "analyzed_duration_s": round(self.analyzed_duration_s, 2),
            "video_duration_s": round(self.video_duration_s, 2),
            "measured_fps": round(self.measured_fps, 2),
            "avg_latency_ms": round(self.avg_frame_latency_ms, 2),
            "device": self.device,
            "gpu_name": self.gpu_name,
            "vision_status": self.vision_status,
            "peak_waiting_pedestrians": self.peak_waiting_pedestrians,
            "peak_crossing_pedestrians": self.peak_crossing_pedestrians,
            "peak_approaching_vehicles": self.peak_approaching_vehicles,
            "peak_queued_vehicles": self.peak_queued_vehicles,
            "initial_phase": self.initial_phase.value if self.initial_phase else "N/A",
            "final_phase": self.final_phase.value if self.final_phase else "N/A",
            "num_transitions": self.num_transitions,
            "pedestrian_priority_activations": self.pedestrian_priority_activations,
            "max_pedestrian_wait_s": round(self.max_pedestrian_wait_s, 1),
            "safety_violations": self.safety_violations,
            "green_walk_conflicts": self.green_walk_conflicts,
        }
