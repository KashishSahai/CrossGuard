"""Deterministic, controller-driven Bloom City simulation.

The engine owns representative agents and consumes the controller's output state. It never
decides signals itself. Python is authoritative; the renderer only draws recorded frames.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.adaptive_controller import AdaptiveController
from core.models import (PHASE_ORDER, ControllerConfig, ControllerState, DemandSnapshot,
                         Decision, PedestrianSignal, Phase, VehicleSignal)

from . import geometry as G
from .simulation_agents import (KIND_ORDER, PedestrianAgent, PedestrianState, VehicleAgent,
                                VehicleKind, VehicleState)

DT = 0.05
BRAKE = 130.0              # comfortable braking (px/s^2)
HARD_BRAKE_FACTOR = 1.6
MIN_GAP = 14.0
ZONE = 260.0               # approach zone length before the stop line
PED_SPEED = 26.0           # px/s
TOKEN_MAX = 9.0            # seconds one axis may hold the box while the other waits
VEHICLE_MIX = [VehicleKind.CAR, VehicleKind.CAR, VehicleKind.VAN, VehicleKind.CAR, VehicleKind.BUS,
               VehicleKind.CAR, VehicleKind.MOTORCYCLE, VehicleKind.TRUCK, VehicleKind.CAR,
               VehicleKind.CAR]


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str
    vehicle_rates: Tuple[float, float, float, float]   # vehicles / minute per approach N,E,S,W
    pedestrian_rate: float                              # arrivals / minute, all crossings
    initial_pedestrians: int = 0


SCENARIOS: Dict[str, Scenario] = {
    s.key: s for s in (
        Scenario("balanced", "Balanced commute", "Moderate vehicles, steady pedestrians.",
                 (10, 8, 10, 8), 9, 2),
        Scenario("ped_surge", "Pedestrian surge", "Many pedestrians, light traffic.",
                 (5, 4, 5, 4), 24, 4),
        Scenario("heavy_traffic", "Heavy traffic", "Dense vehicles, few pedestrians.",
                 (24, 20, 24, 20), 5, 1),
        Scenario("quiet", "Quiet street", "Very little of either.",
                 (3, 2, 3, 2), 2, 0),
    )
}


class BloomCityEngine:
    def __init__(self, controller: Optional[AdaptiveController] = None,
                 scenario: str | Scenario = "balanced", seed: int = 7) -> None:
        self.controller = controller or AdaptiveController(ControllerConfig())
        self.scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.seed = seed
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.controller.reset()
        self.rng = random.Random(self.seed)
        self.time = 0.0
        self.vehicles: List[VehicleAgent] = []
        self.pedestrians: List[PedestrianAgent] = []
        self._next_vid = 1
        self._next_pid = 1
        self._veh_credit = [0.4 * i for i in range(4)]   # staggered start
        self._ped_credit = 0.0
        self._ped_end_rr = 0
        self._mix_i = 0
        self.token_axis = 0
        self.token_time = 0.0
        self.yielding = False
        self.ctrl: ControllerState = self.controller.state
        self.demand = DemandSnapshot(source="simulation")
        for _ in range(self.scenario.initial_pedestrians):
            self._spawn_pedestrian()

    def set_scenario(self, scenario: str | Scenario) -> None:
        self.scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario

    # ------------------------------------------------------------------ main step
    def step(self, dt: float = DT, demand_override: Optional[DemandSnapshot] = None) -> ControllerState:
        demand = demand_override if demand_override is not None else self._measure_demand()
        self.ctrl = self.controller.step(dt, demand)
        self.demand = demand
        self._update_pedestrians(dt)
        self._update_vehicles(dt)
        self._spawn(dt)
        self.time += dt
        return self.ctrl

    def run(self, seconds: float, demand_override: Optional[DemandSnapshot] = None) -> None:
        for _ in range(int(round(seconds / DT))):
            self.step(DT, demand_override)

    # ------------------------------------------------------------------ demand
    def _measure_demand(self) -> DemandSnapshot:
        waiting = [p for p in self.pedestrians if p.state == PedestrianState.WAITING]
        crossing = [p for p in self.pedestrians if p.state in (PedestrianState.CROSSING, PedestrianState.ENTERING, PedestrianState.CLEARING)]
        zone = [v for v in self.vehicles if self._in_zone(v)]
        return DemandSnapshot(
            pedestrians_waiting=len(waiting),
            longest_pedestrian_wait_s=max((p.wait_time for p in waiting), default=0.0),
            pedestrians_in_crosswalk=len(crossing),
            vehicle_demand=len(zone),
            vehicle_queue=sum(1 for v in zone if v.speed < 5.0),
            source="simulation")

    def _in_zone(self, v: VehicleAgent) -> bool:
        line = G.STOP_LINE_S[v.approach]
        return line - ZONE <= v.front <= line + 2

    # ------------------------------------------------------------------ pedestrians
    def _crosswalk_has_vehicle(self, cw: int) -> bool:
        z = G.ZEBRAS[cw]
        return any(G.rects_overlap(v.bbox(), z) for v in self.vehicles)

    def _spawn_pedestrian(self) -> None:
        ends = [(cw, d) for cw in range(4) for d in (1, -1)]
        for _ in range(8):
            cw, d = ends[self._ped_end_rr % 8]
            self._ped_end_rr += 1
            used = {p.slot for p in self.pedestrians
                    if p.crosswalk == cw and p.direction == d and p.state == PedestrianState.WAITING}
            free = [s for s in range(G.MAX_SLOTS) if s not in used]
            if free:
                self.pedestrians.append(PedestrianAgent(self._next_pid, cw, d, free[0]))
                self._next_pid += 1
                return

    def _update_pedestrians(self, dt: float) -> None:
        st = self.ctrl
        walking = st.pedestrian_signal == PedestrianSignal.WALK
        walk_left = max(0.0, self.controller.config.pedestrian_walk_time - st.phase_elapsed)
        for p in self.pedestrians:
            if p.state == PedestrianState.WAITING:
                p.wait_time += dt
                can_start = (walking and st.vehicle_signal == VehicleSignal.RED
                             and st.phase_elapsed >= 0.3 + 0.25 * p.slot
                             and walk_left >= 1.5
                             and not self._crosswalk_has_vehicle(p.crosswalk))
                if can_start:
                    p.state = PedestrianState.CROSSING
            elif p.state == PedestrianState.CROSSING:
                p.progress += PED_SPEED / G.CROSSING_LENGTH * dt
        self.pedestrians = [p for p in self.pedestrians if p.progress < 1.0]

    # ------------------------------------------------------------------ vehicles
    def _crossing_on(self, cw: int) -> bool:
        return any(p.state in (PedestrianState.CROSSING, PedestrianState.ENTERING, PedestrianState.CLEARING) and p.crosswalk == cw
                   for p in self.pedestrians)

    def _in_conflict(self, v: VehicleAgent) -> bool:
        line = G.STOP_LINE_S[v.approach]
        return v.front >= line - 1.0 and v.rear <= line + G.BOX_OUT_OFFSET

    def _update_vehicles(self, dt: float) -> None:
        vsig = self.ctrl.vehicle_signal
        lanes = {a: sorted([v for v in self.vehicles if v.approach == a],
                           key=lambda v: -v.s) for a in range(4)}

        # pass 1: who is committed (past the line, or cannot stop any more under a go signal)
        for v in self.vehicles:
            line = G.STOP_LINE_S[v.approach]
            if v.front >= line - 1.0:
                v.committed = True
            elif not v.committed and vsig != VehicleSignal.RED:
                if line - v.front < v.speed ** 2 / (2 * BRAKE * HARD_BRAKE_FACTOR) + 1.0 \
                        and v.speed > 8.0:
                    v.committed = True

        # intersection-box token (conflict resolution between the two roads)
        self._update_token(dt, lanes)

        conflict = {0: any(self._in_conflict(v) for v in self.vehicles if v.axis == 0),
                    1: any(self._in_conflict(v) for v in self.vehicles if v.axis == 1)}
        blocked_cw = {cw: self._crossing_on(cw) for cw in range(4)}

        for a in range(4):
            leader: Optional[VehicleAgent] = None
            for v in lanes[a]:
                line = G.STOP_LINE_S[a]
                limit: Optional[float] = None
                if leader is not None:
                    limit = leader.rear - MIN_GAP
                if not v.committed and self._must_hold(v, vsig, conflict, blocked_cw):
                    stop_lim = line - 2.0
                    limit = stop_lim if limit is None else min(limit, stop_lim)
                self._advance(v, limit, dt)
                leader = v

        self.vehicles = [v for v in self.vehicles if v.rear <= G.PATH_LEN[v.approach]]

    def _must_hold(self, v, vsig, conflict, blocked_cw) -> bool:
        if vsig == VehicleSignal.RED:
            return True
        line = G.STOP_LINE_S[v.approach]
        dist = line - v.front
        if blocked_cw[v.approach] or blocked_cw[G.opposite(v.approach)]:
            return True
        if dist > ZONE:
            return False
        if vsig == VehicleSignal.YELLOW:
            return True        # anything that could still stop was not committed in pass 1
        other = 1 - v.axis
        if v.axis != self.token_axis or self.yielding:
            return True
        return conflict[other]

    def _update_token(self, dt: float, lanes) -> None:
        self.token_time += dt
        mine = self.token_axis
        waiting = {0: False, 1: False}
        busy = {0: False, 1: False}
        for v in self.vehicles:
            line = G.STOP_LINE_S[v.approach]
            if v.front < line - 1.0 and line - v.front <= ZONE:
                waiting[v.axis] = True
            if self._in_conflict(v):
                busy[v.axis] = True
        other = 1 - mine
        if self.ctrl.vehicle_signal != VehicleSignal.GREEN:
            self.yielding = False
            return
        if waiting[other] and not waiting[mine] and not busy[mine]:
            self._swap_token()
        elif waiting[other] and self.token_time >= TOKEN_MAX:
            self.yielding = True
            if not busy[mine]:
                self._swap_token()
        else:
            self.yielding = False

    def _swap_token(self) -> None:
        self.token_axis = 1 - self.token_axis
        self.token_time = 0.0
        self.yielding = False

    def _advance(self, v: VehicleAgent, limit: Optional[float], dt: float) -> None:
        target = v.vmax
        if limit is not None:
            room = limit - v.front
            target = 0.0 if room <= 0 else min(target, math.sqrt(2 * BRAKE * room))
        old_speed = v.speed
        if target > v.speed:
            v.speed = min(target, v.speed + v.accel * dt)
        else:
            v.speed = max(target, v.speed - BRAKE * HARD_BRAKE_FACTOR * dt)
        new_s = v.s + v.speed * dt
        if limit is not None:
            new_s = max(v.s, min(new_s, limit - v.length / 2))
            v.speed = (new_s - v.s) / dt
        v.s = new_s
        if v.speed < 1.0:
            v.state = VehicleState.STOPPED
            v.speed = 0.0 if limit is not None else v.speed
        elif v.speed < old_speed - 0.05:
            v.state = VehicleState.BRAKING
        else:
            v.state = VehicleState.MOVING

    def _spawn(self, dt: float) -> None:
        sc = self.scenario
        for a in range(4):
            self._veh_credit[a] += sc.vehicle_rates[a] / 60.0 * dt
            if self._veh_credit[a] < 1.0:
                continue
            kind = VEHICLE_MIX[(self._mix_i + a) % len(VEHICLE_MIX)]
            length = VehicleAgent(0, a, kind, 0).length
            start_s = -length / 2
            clear = all(v.rear > start_s + length / 2 + MIN_GAP + 6 for v in self.vehicles
                        if v.approach == a)
            if clear:
                self.vehicles.append(VehicleAgent(self._next_vid, a, kind, start_s, 55.0))
                self._next_vid += 1
                self._mix_i += 1
                self._veh_credit[a] -= 1.0
        self._ped_credit += sc.pedestrian_rate / 60.0 * dt
        if self._ped_credit >= 1.0:
            self._ped_credit -= 1.0
            self._spawn_pedestrian()

    # ------------------------------------------------------------------ test/inspect helpers
    def place_vehicle(self, approach: int, front_distance_to_line: float,
                      kind: VehicleKind = VehicleKind.CAR, speed: float = 0.0) -> VehicleAgent:
        v = VehicleAgent(self._next_vid, approach, kind, 0.0, speed)
        v.s = G.STOP_LINE_S[approach] - front_distance_to_line - v.length / 2
        self._next_vid += 1
        self.vehicles.append(v)
        return v

    def place_pedestrian(self, crosswalk: int = 0, direction: int = 1, slot: int = 0) -> PedestrianAgent:
        p = PedestrianAgent(self._next_pid, crosswalk, direction, slot)
        self._next_pid += 1
        self.pedestrians.append(p)
        return p

    # ------------------------------------------------------------------ recording for the renderer
    def frame(self, reason_table: List[str], t0: float = 0.0) -> dict:
        st = self.ctrl
        if st.decision_reason not in reason_table:
            reason_table.append(st.decision_reason)
        vsig = ("GREEN", "YELLOW", "RED").index(st.vehicle_signal.value)
        psig = ("DONT_WALK", "WALK", "CLEARING").index(st.pedestrian_signal.value)
        dec = ("VEHICLE_PRIORITY", "PEDESTRIAN_PRIORITY", "FIXED_TIME_FALLBACK").index(st.decision.value)
        veh = []
        for v in self.vehicles:
            x, y = v.position
            veh.append([v.id, round(x, 1), round(y, 1), G.HEADING[v.approach],
                        KIND_ORDER.index(v.kind), 1 if v.state != VehicleState.MOVING else 0])
        ped = []
        for p in self.pedestrians:
            x, y = p.position
            ped.append([p.id, round(x, 1), round(y, 1),
                        1 if p.state in (PedestrianState.CROSSING, PedestrianState.ENTERING, PedestrianState.CLEARING) else 0, round(p.heading)])
        d = self.demand
        return {"t": round(self.time - t0, 2), "ph": PHASE_ORDER.index(st.current_phase),
                "vs": vsig, "ps": psig, "dc": dec, "rs": reason_table.index(st.decision_reason),
                "rm": round(st.time_remaining, 1), "fs": int(st.fail_safe_active),
                "nv": int(d.vehicle_demand), "np": int(d.pedestrians_waiting),
                "nc": int(d.pedestrians_in_crosswalk),
                "nq": int(d.vehicle_queue),
                "na": max(0, int(d.vehicle_demand) - int(d.vehicle_queue)),   # demand = approaching + queued
                "lw": round(float(d.longest_pedestrian_wait_s), 1), "v": veh, "p": ped}

    def record(self, seconds: float, fps: int = 10) -> Tuple[List[dict], List[str]]:
        """Advance the simulation and record frames for the renderer."""
        reasons: List[str] = []
        frames: List[dict] = []
        t0 = self.time                       # frame times are relative to the recording start
        steps_per_frame = max(1, int(round(1.0 / fps / DT)))
        for i in range(int(round(seconds / DT))):
            self.step(DT)
            if i % steps_per_frame == 0:
                frames.append(self.frame(reasons, t0))
        return frames, reasons


def build_frames(scenario: str = "balanced", seconds: float = 70.0, warmup: float = 20.0,
                 hold_phase: Optional[Phase] = None, seed: int = 7,
                 fps: int = 10) -> Tuple[List[dict], List[str], BloomCityEngine]:
    """Convenience: run a scenario (or inspect one frozen phase) and return recorded frames."""
    eng = BloomCityEngine(scenario=scenario, seed=seed)
    if hold_phase is None:
        eng.run(warmup)
    else:
        eng.run(warmup)                      # populate roads under normal control first
        eng.controller.force_phase(hold_phase, hold=True)
        eng.run(6.0)                         # let agents react to the forced phase
    frames, reasons = eng.record(seconds, fps)
    return frames, reasons, eng
