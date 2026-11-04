"""
CrossGuard Telemetry and Session Data Logger
============================================
Collects longitudinal records from the AI vision pipeline, spatial zone analyzer,
and adaptive signal controller for real-time charting, CSV export, and audit trails.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import io
import pandas as pd

from core.adaptive_controller import ControllerDecision, SignalPhase
from core.spatial_analysis import SpatialSnapshot


class TelemetryManager:
    """
    In-memory session telemetry aggregator and CSV data export engine.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.events: List[Dict[str, str]] = []
        self._served_pedestrians: set = set()
        self._observed_pedestrians: set = set()
        self._priority_activations: int = 0
        self._clearance_extensions: int = 0
        self._last_phase: Optional[SignalPhase] = None

    def record_step(
        self,
        snapshot: SpatialSnapshot,
        decision: ControllerDecision,
        frame_idx: int = 0,
    ) -> None:
        """Records a single synchronous processing step into session telemetry."""
        now_str = datetime.now().strftime("%H:%M:%S")

        # Track priority activations
        if decision.decision_case in ("CASE_C", "CASE_D"):
            if decision.previous_phase == SignalPhase.VEHICLE_GREEN and decision.current_phase == SignalPhase.VEHICLE_YELLOW:
                self._priority_activations += 1

        # Track clearance extensions
        if decision.decision_case == "CASE_E" and decision.clearance_extensions > 0:
            self._clearance_extensions = max(self._clearance_extensions, decision.clearance_extensions)

        # Track unique pedestrians who entered crossing / were served or waited
        for pid in snapshot.crossing_track_ids:
            self._served_pedestrians.add(pid)
            self._observed_pedestrians.add(pid)

        for pid in snapshot.waiting_track_ids:
            self._observed_pedestrians.add(pid)

        self._last_phase = decision.current_phase

        # Append structured row
        record = {
            "timestamp": snapshot.timestamp,
            "wall_time": now_str,
            "frame_idx": frame_idx,
            "pedestrians_waiting": snapshot.waiting_pedestrians_count,
            "pedestrians_crossing": snapshot.crossing_pedestrians_count,
            "max_wait_seconds": snapshot.max_pedestrian_waiting_time,
            "vehicles_approach": snapshot.approach_vehicles_count,
            "queue_density": snapshot.queue_density_level,
            "crosswalk_occupied": 1 if snapshot.crosswalk_occupied else 0,
            "signal_phase": decision.current_phase.value,
            "phase_remaining_sec": decision.phase_remaining_seconds,
            "decision_case": decision.decision_case,
            "clearance_extensions": decision.clearance_extensions,
        }
        self.records.append(record)
        if len(self.records) > 2000:
            self.records.pop(0)

        # Append to audit event stream if phase transitioned or significant case occurred
        if decision.current_phase != decision.previous_phase or decision.decision_case in ("CASE_C", "CASE_D", "CASE_E"):
            self.add_event(
                level="DECISION" if decision.decision_case.startswith("CASE") else "PHASE",
                message=decision.reason,
            )

    def add_event(self, level: str, message: str) -> None:
        """Appends a timestamped audit log event."""
        self.events.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "msg": message,
        })
        if len(self.events) > 300:
            self.events.pop(0)

    def get_dataframe(self) -> pd.DataFrame:
        """Returns recorded telemetry as a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame(self.records)

    def get_csv_bytes(self) -> bytes:
        """Encodes session telemetry to CSV bytes for user download."""
        df = self.get_dataframe()
        if df.empty:
            return b"timestamp,message\n0.0,No telemetry recorded\n"
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates aggregated metrics from session history."""
        if not self.records:
            return {
                "has_data": False,
                "total_records": 0,
                "total_served": 0,
                "total_observed_peds": 0,
                "avg_wait_time": 0.0,
                "priority_activations": 0,
                "clearance_extensions": 0,
                "peak_queue": 0,
            }

        df = pd.DataFrame(self.records)
        avg_wait = df["max_wait_seconds"].mean() if "max_wait_seconds" in df else 0.0
        peak_queue = df["vehicles_approach"].max() if "vehicles_approach" in df else 0

        return {
            "has_data": True,
            "total_records": len(df),
            "total_served": len(self._served_pedestrians),
            "total_observed_peds": len(self._observed_pedestrians),
            "avg_wait_time": round(float(avg_wait), 1),
            "priority_activations": self._priority_activations,
            "clearance_extensions": self._clearance_extensions,
            "peak_queue": int(peak_queue),
        }
