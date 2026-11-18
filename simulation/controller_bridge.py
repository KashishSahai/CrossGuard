"""Controller to Bloom City Integration Bridge (Step 6).

Translates AdaptiveController decisions and phases into Bloom City visual signal
representations, cleanly separating controller decision logic from renderer visualization.
Pure Python; no UI dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.adaptive_controller import AdaptiveController
from core.models import (
    PHASE_ORDER,
    PHASE_OUTPUTS,
    ControllerState,
    Decision,
    DemandSnapshot,
    PedestrianSignal,
    Phase,
    VehicleSignal,
)
from simulation.bloom_city_engine import BloomCityEngine, DT
from simulation.bloom_city_renderer import render_bloom_city_html


class SimulationMode(str, Enum):
    SCENARIO_DEMO = "SCENARIO DEMO"
    LIVE_CONTROLLER = "LIVE CONTROLLER"


# Visual color constants matching bloom_city_renderer
VEHICLE_LENS_COLORS = {
    VehicleSignal.GREEN: "#38d889",
    VehicleSignal.YELLOW: "#ffd23f",
    VehicleSignal.RED: "#ff5d6c",
}

PEDESTRIAN_LENS_COLORS = {
    PedestrianSignal.DONT_WALK: "#ff5d6c",
    PedestrianSignal.WALK: "#38d889",
    PedestrianSignal.CLEARING: "#ffb020",
}

PEDESTRIAN_GLYPHS = {
    PedestrianSignal.DONT_WALK: "✋",
    PedestrianSignal.WALK: "🚶",
    PedestrianSignal.CLEARING: "⏳",
}


@dataclass(frozen=True)
class BloomCitySignalState:
    """Explicit visual signal state for Bloom City signal heads."""
    phase: Phase
    vehicle_signal: VehicleSignal
    pedestrian_signal: PedestrianSignal
    decision: Decision
    decision_reason: str
    phase_index: int
    vehicle_signal_index: int
    pedestrian_signal_index: int
    decision_index: int
    time_remaining: float
    fail_safe_active: bool
    vehicle_lens_color: str
    pedestrian_lens_color: str
    pedestrian_glyph: str

    @property
    def is_safe(self) -> bool:
        """Safety invariant: vehicles cannot have green while pedestrians have walk."""
        return not (
            self.vehicle_signal == VehicleSignal.GREEN
            and self.pedestrian_signal == PedestrianSignal.WALK
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "vehicle_signal": self.vehicle_signal.value,
            "pedestrian_signal": self.pedestrian_signal.value,
            "decision": self.decision.value,
            "decision_reason": self.decision_reason,
            "phase_index": self.phase_index,
            "vehicle_signal_index": self.vehicle_signal_index,
            "pedestrian_signal_index": self.pedestrian_signal_index,
            "decision_index": self.decision_index,
            "time_remaining": round(self.time_remaining, 1),
            "fail_safe_active": self.fail_safe_active,
            "vehicle_lens_color": self.vehicle_lens_color,
            "pedestrian_lens_color": self.pedestrian_lens_color,
            "pedestrian_glyph": self.pedestrian_glyph,
        }


def map_controller_state(state: Optional[ControllerState]) -> BloomCitySignalState:
    """Deterministically map a ControllerState into BloomCitySignalState.

    If state is missing, unknown, or corrupted, fails safely to ALL_RED_CLEARANCE.
    """
    if state is None or not isinstance(state, ControllerState):
        # Fail-safe default: all-red clearance, don't walk
        safe_phase = Phase.ALL_RED_CLEARANCE
        v_sig, p_sig = PHASE_OUTPUTS[safe_phase]
        return BloomCitySignalState(
            phase=safe_phase,
            vehicle_signal=v_sig,
            pedestrian_signal=p_sig,
            decision=Decision.FIXED_TIME_FALLBACK,
            decision_reason="Invalid controller state; fail-safe all-red clearance active.",
            phase_index=PHASE_ORDER.index(safe_phase),
            vehicle_signal_index=("GREEN", "YELLOW", "RED").index(v_sig.value),
            pedestrian_signal_index=("DONT_WALK", "WALK", "CLEARING").index(p_sig.value),
            decision_index=2,
            time_remaining=0.0,
            fail_safe_active=True,
            vehicle_lens_color=VEHICLE_LENS_COLORS[v_sig],
            pedestrian_lens_color=PEDESTRIAN_LENS_COLORS[p_sig],
            pedestrian_glyph=PEDESTRIAN_GLYPHS[p_sig],
        )

    phase = state.current_phase
    if phase not in PHASE_OUTPUTS:
        # Unknown phase: fail safely
        phase = Phase.ALL_RED_CLEARANCE

    v_sig, p_sig = PHASE_OUTPUTS[phase]

    vsig_idx = ("GREEN", "YELLOW", "RED").index(v_sig.value)
    psig_idx = ("DONT_WALK", "WALK", "CLEARING").index(p_sig.value)
    dec_idx = ("VEHICLE_PRIORITY", "PEDESTRIAN_PRIORITY", "FIXED_TIME_FALLBACK").index(
        state.decision.value if state.decision in Decision else Decision.FIXED_TIME_FALLBACK.value
    )

    return BloomCitySignalState(
        phase=phase,
        vehicle_signal=v_sig,
        pedestrian_signal=p_sig,
        decision=state.decision,
        decision_reason=state.decision_reason,
        phase_index=PHASE_ORDER.index(phase),
        vehicle_signal_index=vsig_idx,
        pedestrian_signal_index=psig_idx,
        decision_index=dec_idx,
        time_remaining=max(0.0, state.time_remaining),
        fail_safe_active=state.fail_safe_active,
        vehicle_lens_color=VEHICLE_LENS_COLORS[v_sig],
        pedestrian_lens_color=PEDESTRIAN_LENS_COLORS[p_sig],
        pedestrian_glyph=PEDESTRIAN_GLYPHS[p_sig],
    )


class ControllerBridge:
    """Bridge coordinating AdaptiveController decisions with Bloom City visualization.

    Supports both SCENARIO_DEMO and LIVE_CONTROLLER modes without mixing
    controller logic into the renderer.
    """

    def __init__(self, mode: SimulationMode = SimulationMode.SCENARIO_DEMO) -> None:
        self.mode = mode

    def map_state(self, state: Optional[ControllerState]) -> BloomCitySignalState:
        """Translate a ControllerState into the Bloom City signal state."""
        return map_controller_state(state)

    def apply_to_engine(self, engine: BloomCityEngine, state: ControllerState) -> None:
        """Synchronize a running BloomCityEngine with an external ControllerState."""
        engine.ctrl = state
        # Agents react to the signal state immediately
        engine._update_vehicles(0.0)
        engine._update_pedestrians(0.0)

    def build_controller_frames(
        self,
        engine: BloomCityEngine,
        controller: AdaptiveController,
        demand_sequence: Sequence[DemandSnapshot],
        dt: float = DT,
        fps: int = 10,
    ) -> Tuple[List[dict], List[str]]:
        """Record simulation frames driven directly by an AdaptiveController and demand snapshots."""
        reasons: List[str] = []
        frames: List[dict] = []
        steps_per_frame = max(1, int(round(1.0 / fps / dt)))

        # Assign external controller to the engine
        engine.controller = controller
        t0 = engine.time

        step_idx = 0
        for demand in demand_sequence:
            engine.step(dt=dt, demand_override=demand)
            if step_idx % steps_per_frame == 0:
                frames.append(engine.frame(reasons, t0))
            step_idx += 1

        return frames, reasons

    def render_live_html(
        self,
        frames: Sequence[dict],
        reasons: Sequence[str],
        theme: str = "day",
        title: str = "Live Adaptive Control",
    ) -> str:
        """Render self-contained HTML for live controller mode."""
        return render_bloom_city_html(
            frames,
            reasons,
            theme=theme,
            mode=SimulationMode.LIVE_CONTROLLER.value,
            scenario_title=title,
            source_note="Signals driven directly by AdaptiveController",
        )
