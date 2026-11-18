"""Safety-aware finite-state controller. Pure Python; no UI dependencies.

VEHICLE_GREEN -> VEHICLE_YELLOW -> ALL_RED_CLEARANCE -> PEDESTRIAN_WALK
              -> PEDESTRIAN_CLEARANCE -> VEHICLE_GREEN

Safety invariants (enforced structurally, not by demand):
  * the sequence above is the only way phases advance - yellow and all-red are never skipped;
  * VEHICLE_GREEN lasts at least `min_vehicle_green`, so vehicles are always released;
  * pedestrian clearance can only be extended (bounded), never cut short;
  * invalid/missing telemetry falls back to a fixed-time cycle, never to an unsafe state.
"""
from __future__ import annotations

import math
from typing import Optional

from .models import (PHASE_OUTPUTS, ControllerConfig, ControllerState, Decision,
                     DemandSnapshot, Phase)


class AdaptiveController:
    MAX_SUBSTEP = 0.5

    def __init__(self, config: Optional[ControllerConfig] = None,
                 initial_phase: Phase = Phase.VEHICLE_GREEN) -> None:
        self.config = config or ControllerConfig()
        self.reset(initial_phase)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, phase: Phase = Phase.VEHICLE_GREEN) -> None:
        self._phase = phase
        self._elapsed = 0.0
        self._total = 0.0
        self._cycles = 0
        self._ped_served = 0
        self._ext = 0.0
        self._hold = False
        self._invalid_for = 0.0
        self._fail_safe = False
        self._decision = (Decision.VEHICLE_PRIORITY if phase == Phase.VEHICLE_GREEN
                          else Decision.PEDESTRIAN_PRIORITY)
        self._reason = "Vehicles released; no pedestrian request yet."

    def force_phase(self, phase: Phase, hold: bool = False) -> None:
        """Debug/demo helper: jump to a phase (optionally freezing it)."""
        self._enter(phase)
        self._hold = hold
        self._decision = (Decision.VEHICLE_PRIORITY if phase == Phase.VEHICLE_GREEN
                          else Decision.PEDESTRIAN_PRIORITY)
        self._reason = f"Phase forced for inspection ({phase.value})."

    def release_hold(self) -> None:
        self._hold = False

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> ControllerState:
        v, p = PHASE_OUTPUTS[self._phase]
        target = self._target_duration()
        return ControllerState(
            current_phase=self._phase, vehicle_signal=v, pedestrian_signal=p,
            decision=self._decision, decision_reason=self._reason,
            phase_elapsed=self._elapsed, phase_target=target,
            time_remaining=max(0.0, target - self._elapsed),
            fail_safe_active=self._fail_safe, total_time=self._total,
            cycles_completed=self._cycles, pedestrian_phases_served=self._ped_served)

    def _target_duration(self) -> float:
        c = self.config
        return {
            Phase.VEHICLE_GREEN: c.fixed_green_time if self._fail_safe else c.min_vehicle_green,
            Phase.VEHICLE_YELLOW: c.yellow_time,
            Phase.ALL_RED_CLEARANCE: c.all_red_time,
            Phase.PEDESTRIAN_WALK: c.pedestrian_walk_time,
            Phase.PEDESTRIAN_CLEARANCE: c.pedestrian_clearance_time + self._ext,
        }[self._phase]

    # ------------------------------------------------------------------ stepping
    def step(self, dt: float, demand: Optional[DemandSnapshot] = None) -> ControllerState:
        if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            return self.state
        if demand is not None and hasattr(demand, "demand") and isinstance(getattr(demand, "demand"), DemandSnapshot):
            demand = getattr(demand, "demand")
        while dt > 1e-9:
            chunk = min(dt, self.MAX_SUBSTEP)
            self._tick(chunk, demand)
            dt -= chunk
        return self.state

    def _tick(self, dt: float, demand: Optional[DemandSnapshot]) -> None:
        self._elapsed += dt
        self._total += dt
        self._update_failsafe(dt, demand)
        if self._hold:
            return
        c, ph = self.config, self._phase
        if ph == Phase.VEHICLE_GREEN:
            self._tick_green(demand)
        elif ph == Phase.VEHICLE_YELLOW:
            if self._elapsed >= c.yellow_time:
                self._enter(Phase.ALL_RED_CLEARANCE)
                self._reason = "Yellow complete; all-red clearance empties the intersection."
        elif ph == Phase.ALL_RED_CLEARANCE:
            if self._elapsed >= c.all_red_time:
                self._enter(Phase.PEDESTRIAN_WALK)
                self._reason = "Intersection clear; pedestrians may cross."
        elif ph == Phase.PEDESTRIAN_WALK:
            if self._elapsed >= c.pedestrian_walk_time:
                self._enter(Phase.PEDESTRIAN_CLEARANCE)
                self._reason = "Walk interval over; pedestrians already crossing finish."
        elif ph == Phase.PEDESTRIAN_CLEARANCE:
            self._tick_clearance(dt, demand)

    def _update_failsafe(self, dt: float, demand: Optional[DemandSnapshot]) -> None:
        if demand is None or not demand.is_valid():
            self._invalid_for += dt
            if self._invalid_for >= self.config.failsafe_after:
                self._fail_safe = True
        else:
            self._invalid_for = 0.0
            self._fail_safe = False

    def _tick_green(self, demand: Optional[DemandSnapshot]) -> None:
        c = self.config
        if self._fail_safe:
            self._decision = Decision.FIXED_TIME_FALLBACK
            remaining = c.fixed_green_time - self._elapsed
            if remaining <= 0:
                self._start_pedestrian_sequence("fixed-time fallback, telemetry unavailable; "
                                                "pedestrian phase scheduled.")
            else:
                self._reason = (f"Telemetry unavailable - fixed-time fallback "
                                f"({remaining:.0f}s of vehicle green left).")
            return

        self._decision = Decision.VEHICLE_PRIORITY
        d = demand
        if d is None or not d.is_valid():   # brief dropout: keep vehicles moving
            self._reason = "Telemetry briefly unavailable; holding vehicle green."
            return
        waiting = int(d.pedestrians_waiting)
        crossing = int(d.pedestrians_in_crosswalk)
        ped_demand = waiting + crossing
        wait = float(d.longest_pedestrian_wait_s)
        vdem = int(d.vehicle_demand)
        heavy = vdem >= c.vehicle_release_threshold
        if ped_demand == 0:
            self._reason = "No pedestrians waiting; vehicles keep the green."
            return
        left = c.min_vehicle_green - self._elapsed
        if left > 0:
            self._reason = (f"{waiting} waiting; minimum vehicle green protects vehicle release "
                            f"({left:.0f}s left).")
            return
        who = f"{waiting} pedestrian{'s' if waiting != 1 else ''} waiting"
        if wait >= c.max_pedestrian_wait:
            self._start_pedestrian_sequence(f"{who}; longest wait {wait:.0f}s reached the "
                                            f"{c.max_pedestrian_wait:.0f}s limit.")
        elif self._elapsed >= c.max_vehicle_green:
            self._start_pedestrian_sequence(f"{who}; maximum vehicle green "
                                            f"({c.max_vehicle_green:.0f}s) reached.")
        elif ped_demand >= c.pedestrian_demand_threshold and not heavy:
            self._start_pedestrian_sequence(f"{who} at the crossing while vehicle release "
                                            f"demand is low ({vdem}).")
        elif (ped_demand < c.pedestrian_demand_threshold and wait >= c.single_pedestrian_wait
              and not heavy):
            self._start_pedestrian_sequence(f"{who}, waiting {wait:.0f}s; vehicle demand "
                                            f"is manageable ({vdem}).")
        elif heavy:
            self._reason = (f"{who}, but vehicle demand is heavy ({vdem}); vehicles keep "
                            f"the green until wait reaches {c.max_pedestrian_wait:.0f}s "
                            f"(now {wait:.0f}s).")
        else:
            self._reason = f"{who}; below the demand threshold, still observing."

    def _tick_clearance(self, dt: float, demand: Optional[DemandSnapshot]) -> None:
        c = self.config
        if self._elapsed < c.pedestrian_clearance_time + self._ext:
            return
        occupied = (demand is not None and demand.is_valid()
                    and demand.pedestrians_in_crosswalk > 0)
        if occupied and self._ext < c.max_clearance_extension:
            self._ext = min(c.max_clearance_extension, self._ext + dt)
            self._reason = "Pedestrians still in the crosswalk; extending clearance."
            return
        self._enter(Phase.VEHICLE_GREEN)
        self._cycles += 1
        self._decision = Decision.VEHICLE_PRIORITY
        self._reason = "Crosswalk clear; vehicles released."

    # ------------------------------------------------------------------ helpers
    def _start_pedestrian_sequence(self, reason: str) -> None:
        self._enter(Phase.VEHICLE_YELLOW)
        self._decision = Decision.PEDESTRIAN_PRIORITY
        self._ped_served += 1
        self._reason = reason[0].upper() + reason[1:]

    def _enter(self, phase: Phase) -> None:
        self._phase = phase
        self._elapsed = 0.0
        self._ext = 0.0
