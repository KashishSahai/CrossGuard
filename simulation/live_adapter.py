"""Bloom City Live Vision State Adapter for CrossGuard.

Couples verified live perception/telemetry/FSM results (PipelineFrameResult)
directly to Bloom City's physical representative simulation.

The authoritative pipeline outputs (PipelineFrameResult) drive the signals,
pedestrian waiting/crossing behavior, and vehicle deceleration/movement
in Bloom City without duplicating detection, tracking, or FSM logic.
Pure Python; no UI dependencies.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from core.models import (
    PHASE_ORDER,
    ControllerState,
    Decision,
    DemandSnapshot,
    PedestrianSignal,
    Phase,
    VehicleSignal,
)
from core.pipeline import PipelineFrameResult
from core.spatial_analysis import Approach, SpatialZone, ZoneType
from simulation import geometry as G
from simulation.bloom_city_engine import (
    BRAKE,
    DT,
    HARD_BRAKE_FACTOR,
    MIN_GAP,
    PED_SPEED,
    ZONE,
    BloomCityEngine,
)
from simulation.bloom_city_renderer import render_bloom_city_html
from simulation.controller_bridge import (
    BloomCitySignalState,
    SimulationMode,
    map_controller_state,
)
from simulation.simulation_agents import (
    KIND_ORDER,
    KIND_SPECS,
    PedestrianAgent,
    PedestrianState,
    VehicleAgent,
    VehicleKind,
    VehicleState,
)

LIVE_MODE_LABEL = "LIVE CONTROLLER"
LIVE_SOURCE_NOTE = "Representative agents driven by each processed frame's PipelineFrameResult"

APPROACH_TO_INDEX: Dict[Approach, int] = {
    Approach.NORTH: 0,
    Approach.EAST: 1,
    Approach.SOUTH: 2,
    Approach.WEST: 3,
}

INDEX_TO_APPROACH: Dict[int, Approach] = {
    0: Approach.NORTH,
    1: Approach.EAST,
    2: Approach.SOUTH,
    3: Approach.WEST,
}


def _is_pipeline_frame_result(obj: Any) -> bool:
    """Robust type validation for PipelineFrameResult across Streamlit module reloads."""
    if isinstance(obj, PipelineFrameResult):
        return True
    return (
        getattr(obj, "__class__", None) is not None
        and obj.__class__.__name__ == "PipelineFrameResult"
        and hasattr(obj, "frame_index")
        and hasattr(obj, "controller_state")
        and hasattr(obj, "demand_snapshot")
    )


class BloomCityLiveAdapter(BloomCityEngine):
    """Video-driven Bloom City simulation adapter.

    Maintains progressive physical simulation state across video batches and frames,
    driving Bloom City visual agents and signals strictly from PipelineFrameResult.
    """

    def __init__(self, seed: int = 7, target_fps: float = 10.0) -> None:
        super().__init__(seed=seed)
        self.target_fps = max(1.0, float(target_fps))
        self.video_name: str = ""
        self.processed_frames_count: int = 0
        self.last_frame_index: int = -1
        self.recorded_frames: List[dict] = []
        self.reason_table: List[str] = []
        self.last_recorded_time: float = -1.0
        self.initial_timestamp: Optional[float] = None
        self.approaching: int = 0
        self.completed_pedestrians: List[PedestrianAgent] = []
        self._init_live_agents()

    def reset(self) -> None:
        """Reset the live simulation state back to t=0 with initial corridor traffic."""
        super().reset()
        self.processed_frames_count = 0
        self.last_frame_index = -1
        self.recorded_frames = []
        self.reason_table = []
        self.last_recorded_time = -1.0
        self.initial_timestamp = None
        self.approaching = 0
        self.completed_pedestrians = []
        self._init_live_agents()

    def _init_live_agents(self) -> None:
        """Populate initial representative corridor traffic on North-South road."""
        self.vehicles.clear()
        self.pedestrians.clear()
        self.completed_pedestrians.clear()
        self._next_vid = 1
        self._next_pid = 1
        # Place initial vehicles moving along main corridor (North approach 0, South approach 2)
        self.place_vehicle(approach=0, front_distance_to_line=110.0, kind=VehicleKind.CAR, speed=52.0)
        self.place_vehicle(approach=2, front_distance_to_line=130.0, kind=VehicleKind.CAR, speed=48.0)

    # -------------------------------------------------------------------------
    # Core Processing: Single Frame & Batch
    # -------------------------------------------------------------------------
    def process_frame_result(
        self,
        res: PipelineFrameResult,
        video_name: str = "",
        force_record: bool = True,
    ) -> dict:
        """Process one authoritative PipelineFrameResult and update simulation state.

        Args:
            res: Authoritative result from AdaptivePipeline.process_frame().
            video_name: Optional filename for display metadata.
            force_record: If True, always records this frame into recorded_frames.

        Returns:
            The recorded or latest frame dictionary matching the Bloom City schema.
        """
        if not _is_pipeline_frame_result(res):
            raise TypeError(f"Expected PipelineFrameResult, got {type(res)}")

        dt = max(0.001, float(res.dt))
        self.time = float(res.timestamp)
        self.ctrl = res.controller_state
        self.demand = res.demand_snapshot
        self.approaching = int(res.approaching_vehicles)
        self.last_frame_index = res.frame_index
        self.processed_frames_count += 1
        if video_name:
            self.video_name = video_name

        # Keep internal controller state in sync with authoritative pipeline result
        cs = res.controller_state
        self.controller._phase = cs.current_phase
        self.controller._elapsed = cs.phase_elapsed
        self.controller._total = cs.total_time
        self.controller._decision = cs.decision
        self.controller._reason = cs.decision_reason
        self.controller._fail_safe = cs.fail_safe_active
        self.controller._cycles = cs.cycles_completed
        self.controller._ped_served = cs.pedestrian_phases_served

        # Ensure signal state is available and safe
        sig = res.signal_state or map_controller_state(self.ctrl)
        assert sig.is_safe, f"Safety invariant violation: {sig.vehicle_signal} vs {sig.pedestrian_signal}"

        # 1. Synchronize representative pedestrians from video spatial detections & demand
        self._sync_pedestrians(res, dt)

        # 2. Synchronize representative vehicles from video vehicle demand & zones
        self._sync_vehicles(res, dt)

        # 3. Record frame for visualization player
        frame_dict = self._record_current_frame(force=force_record)
        return frame_dict

    def process_batch(
        self,
        results: Sequence[PipelineFrameResult],
        video_name: str = "",
    ) -> List[dict]:
        """Process an ordered batch of PipelineFrameResult items progressively."""
        if not results:
            return []
        out = []
        n = len(results)
        for i, res in enumerate(results):
            # Force record on the very first and very last frame of a batch
            force = (i == 0 or i == n - 1)
            frame_dict = self.process_frame_result(res, video_name=video_name, force_record=force)
            out.append(frame_dict)
        return out

    # -------------------------------------------------------------------------
    # Pedestrian Synchronization
    # -------------------------------------------------------------------------
    def _sync_pedestrians(self, res: PipelineFrameResult, dt: float) -> None:
        """Map spatial waiting/crossing zones to persistent representative pedestrians.

        Maintains persistent representative pedestrians across video frames:
        - Associates detections with agents via ByteTrack track_id when available
        - Handles track-ID fragmentation by re-associating existing agents on the approach
        - Preserves waiting agents across momentary detection drops using grace_period
        - Never destroys an agent that is ENTERING, CROSSING, or CLEARING until progress >= 1.0
        - Enforces state progression: WAITING -> ENTERING -> CROSSING -> CLEARING -> CLEARED
        - Records exact lifecycle timestamps on every representative pedestrian
        """
        detected_waiting_tracks: Dict[int, int] = {}   # track_id -> crosswalk index
        detected_crossing_tracks: Dict[int, int] = {}  # track_id -> crosswalk index
        det_zone_by_track: Dict[int, str] = {}
        waiting_by_approach: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        crossing_by_approach: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        # Analyze spatial detections
        for det in res.spatial_detections:
            cw = None
            if det.approach is not None:
                cw = APPROACH_TO_INDEX.get(det.approach)
            elif det.zone is not None:
                for app in Approach:
                    if det.zone.value.startswith(app.value + "_"):
                        cw = APPROACH_TO_INDEX.get(app)
                        break
            if cw is None:
                cw = 1  # default to East crosswalk

            if det.track_id is not None and det.zone is not None:
                det_zone_by_track[det.track_id] = det.zone.value

            if det.zone_type == ZoneType.WAITING:
                waiting_by_approach[cw] += 1
                if det.track_id is not None:
                    detected_waiting_tracks[det.track_id] = cw
            elif det.zone_type == ZoneType.CROSSING:
                crossing_by_approach[cw] += 1
                if det.track_id is not None:
                    detected_crossing_tracks[det.track_id] = cw

        # Fallback to demand snapshot if spatial zone mapping was not approach-attributed
        total_waiting_zones = sum(waiting_by_approach.values())
        if res.waiting_pedestrians > total_waiting_zones:
            diff = res.waiting_pedestrians - total_waiting_zones
            target_cw = 1 if waiting_by_approach[1] > 0 or total_waiting_zones == 0 else max(waiting_by_approach, key=waiting_by_approach.get)
            waiting_by_approach[target_cw] += diff

        total_crossing_zones = sum(crossing_by_approach.values())
        if res.crossing_pedestrians > total_crossing_zones:
            diff = res.crossing_pedestrians - total_crossing_zones
            target_cw = 1 if crossing_by_approach[1] > 0 or total_crossing_zones == 0 else max(crossing_by_approach, key=crossing_by_approach.get)
            crossing_by_approach[target_cw] += diff

        # 1. Update existing agents matching detected track IDs
        for p in self.pedestrians:
            if p.track_id is not None:
                if p.track_id in detected_waiting_tracks:
                    p.last_seen = self.time
                    p.crosswalk = detected_waiting_tracks[p.track_id]
                    if not p.source_zone and p.track_id in det_zone_by_track:
                        p.source_zone = det_zone_by_track[p.track_id]
                elif p.track_id in detected_crossing_tracks:
                    p.last_seen = self.time
                    p.crosswalk = detected_crossing_tracks[p.track_id]
                    if not p.source_zone and p.track_id in det_zone_by_track:
                        p.source_zone = det_zone_by_track[p.track_id]
                    if p.state == PedestrianState.WAITING:
                        if self.ctrl.pedestrian_signal == PedestrianSignal.WALK and self.ctrl.vehicle_signal == VehicleSignal.RED:
                            p.state = PedestrianState.CROSSING
                            if p.crossing_timestamp is None:
                                p.crossing_timestamp = round(self.time, 2)

        # 2. Re-associate or spawn agents for detected waiting tracks
        for tid, cw in detected_waiting_tracks.items():
            if any(p.track_id == tid for p in self.pedestrians):
                continue
            # Associate with an unassigned waiting agent on cw if available
            unassigned = next(
                (p for p in self.pedestrians
                 if p.crosswalk == cw and p.state == PedestrianState.WAITING and p.track_id is None),
                None
            )
            if unassigned is not None:
                unassigned.track_id = tid
                unassigned.last_seen = self.time
                if not unassigned.source_zone and tid in det_zone_by_track:
                    unassigned.source_zone = det_zone_by_track[tid]
                continue

            # Check if there's an existing waiting agent on cw whose track is no longer detected (ID switch)
            existing_cw_waiting = [p for p in self.pedestrians if p.crosswalk == cw and p.state == PedestrianState.WAITING]
            candidate = next(
                (p for p in existing_cw_waiting if p.track_id not in detected_waiting_tracks),
                None
            )
            if candidate is not None and len(existing_cw_waiting) >= waiting_by_approach[cw]:
                candidate.track_id = tid
                candidate.last_seen = self.time
                if not candidate.source_zone and tid in det_zone_by_track:
                    candidate.source_zone = det_zone_by_track[tid]
                continue

            # Spawn new waiting agent
            used_slots = {p.slot for p in self.pedestrians if p.crosswalk == cw and p.state == PedestrianState.WAITING}
            free_slots = [s for s in range(G.MAX_SLOTS) if s not in used_slots]
            slot = free_slots.pop(0) if free_slots else (len(used_slots) % G.MAX_SLOTS)
            sz = det_zone_by_track.get(tid, INDEX_TO_APPROACH[cw].value + "_WAITING")
            self.pedestrians.append(
                PedestrianAgent(
                    id=self._next_pid,
                    crosswalk=cw,
                    direction=1,
                    slot=slot,
                    state=PedestrianState.WAITING,
                    progress=0.0,
                    wait_time=0.0,
                    track_id=tid,
                    last_seen=self.time,
                    source_zone=sz,
                    waiting_timestamp=round(self.time, 2),
                )
            )
            self._next_pid += 1

        # 3. Re-associate or spawn agents for detected crossing tracks
        for tid, cw in detected_crossing_tracks.items():
            if any(p.track_id == tid for p in self.pedestrians):
                continue
            # Check if an existing crossing agent on cw is no longer tracked (track ID fragmentation)
            candidate = next(
                (p for p in self.pedestrians
                 if p.crosswalk == cw
                 and p.state in (PedestrianState.ENTERING, PedestrianState.CROSSING, PedestrianState.CLEARING)
                 and p.track_id not in detected_crossing_tracks),
                None
            )
            if candidate is not None:
                candidate.track_id = tid
                candidate.last_seen = self.time
                if not candidate.source_zone and tid in det_zone_by_track:
                    candidate.source_zone = det_zone_by_track[tid]
                continue

            # Only spawn if current crossing agents on cw < crossing_by_approach[cw]
            existing_cw_crossing = sum(
                1 for p in self.pedestrians
                if p.crosswalk == cw and p.state in (PedestrianState.ENTERING, PedestrianState.CROSSING, PedestrianState.CLEARING)
            )
            if existing_cw_crossing < crossing_by_approach[cw]:
                used_slots = {p.slot for p in self.pedestrians if p.crosswalk == cw}
                slot = len(used_slots) % G.MAX_SLOTS
                sz = det_zone_by_track.get(tid, INDEX_TO_APPROACH[cw].value + "_CROSSING")
                self.pedestrians.append(
                    PedestrianAgent(
                        id=self._next_pid,
                        crosswalk=cw,
                        direction=1,
                        slot=slot,
                        state=PedestrianState.CROSSING,
                        progress=0.15,
                        wait_time=0.0,
                        track_id=tid,
                        last_seen=self.time,
                        source_zone=sz,
                        entering_timestamp=round(self.time, 2),
                        crossing_timestamp=round(self.time, 2),
                    )
                )
                self._next_pid += 1

        # 4. Fill unassigned waiting & crossing slots per crosswalk to match demand
        for cw in range(4):
            # Fill unassigned waiting slots
            target_w = waiting_by_approach[cw]
            current_waiting = [
                p for p in self.pedestrians
                if p.crosswalk == cw and p.state == PedestrianState.WAITING
            ]
            if len(current_waiting) < target_w:
                used_slots = {p.slot for p in current_waiting}
                free_slots = [s for s in range(G.MAX_SLOTS) if s not in used_slots]
                for _ in range(target_w - len(current_waiting)):
                    slot = free_slots.pop(0) if free_slots else 0
                    self.pedestrians.append(
                        PedestrianAgent(
                            id=self._next_pid,
                            crosswalk=cw,
                            direction=1,
                            slot=slot,
                            state=PedestrianState.WAITING,
                            progress=0.0,
                            wait_time=0.0,
                            track_id=None,
                            last_seen=self.time,
                            source_zone=INDEX_TO_APPROACH[cw].value + "_WAITING",
                            waiting_timestamp=round(self.time, 2),
                        )
                    )
                    self._next_pid += 1

            # Fill unassigned crossing slots
            target_c = crossing_by_approach[cw]
            current_crossing = [
                p for p in self.pedestrians
                if p.crosswalk == cw and p.state in (PedestrianState.ENTERING, PedestrianState.CROSSING, PedestrianState.CLEARING)
            ]
            if len(current_crossing) < target_c:
                used_slots = {p.slot for p in self.pedestrians if p.crosswalk == cw}
                for _ in range(target_c - len(current_crossing)):
                    slot = len(used_slots) % G.MAX_SLOTS
                    self.pedestrians.append(
                        PedestrianAgent(
                            id=self._next_pid,
                            crosswalk=cw,
                            direction=1,
                            slot=slot,
                            state=PedestrianState.CROSSING,
                            progress=0.15,
                            wait_time=0.0,
                            track_id=None,
                            last_seen=self.time,
                            source_zone=INDEX_TO_APPROACH[cw].value + "_CROSSING",
                            entering_timestamp=round(self.time, 2),
                            crossing_timestamp=round(self.time, 2),
                        )
                    )
                    self._next_pid += 1

        # 5. Graceful pruning for WAITING agents (crossing agents always persist until cleared)
        walking = (self.ctrl.pedestrian_signal == PedestrianSignal.WALK)
        veh_red = (self.ctrl.vehicle_signal == VehicleSignal.RED)
        clearing = (self.ctrl.pedestrian_signal == PedestrianSignal.CLEARING)

        if not walking:
            surviving = []
            for cw in range(4):
                # Prune WAITING: prioritize actively detected tracks up to target_w
                cw_waiting = [p for p in self.pedestrians if p.crosswalk == cw and p.state == PedestrianState.WAITING]
                target_w = waiting_by_approach[cw]
                cw_waiting.sort(key=lambda p: (p.track_id in detected_waiting_tracks, p.track_id is not None, p.last_seen), reverse=True)
                for p in cw_waiting:
                    time_unseen = self.time - p.last_seen
                    kept_w = sum(1 for x in surviving if x.crosswalk == cw and x.state == PedestrianState.WAITING)
                    if kept_w < target_w:
                        surviving.append(p)
                    elif p.track_id is not None and (
                        (res.waiting_pedestrians == 0 and time_unseen <= p.grace_period and kept_w < 1)
                        or (time_unseen <= 0.12)
                    ):
                        # Preserve agent during full-frame dropout or momentary 1-2 frame flicker
                        surviving.append(p)

            # CROSSING, ENTERING, CLEARING agents ALWAYS persist until progress >= 1.0
            for p in self.pedestrians:
                if p.state != PedestrianState.WAITING:
                    surviving.append(p)

            self.pedestrians = surviving

        # 6. Advance pedestrian states and crossing progress
        for p in self.pedestrians:
            if p.state == PedestrianState.WAITING:
                p.wait_time += dt
                # Transition WAITING -> ENTERING -> CROSSING
                if walking and veh_red and not self._crosswalk_has_vehicle(p.crosswalk):
                    p.state = PedestrianState.CROSSING
                    if p.entering_timestamp is None:
                        p.entering_timestamp = round(self.time, 2)
                    if p.crossing_timestamp is None:
                        p.crossing_timestamp = round(self.time, 2)
            if p.state in (PedestrianState.ENTERING, PedestrianState.CROSSING):
                p.state = PedestrianState.CROSSING
                if p.crossing_timestamp is None:
                    p.crossing_timestamp = round(self.time, 2)
                p.progress += (PED_SPEED / G.CROSSING_LENGTH) * dt
                if clearing:
                    p.state = PedestrianState.CLEARING
                    if p.clearing_timestamp is None:
                        p.clearing_timestamp = round(self.time, 2)
            elif p.state == PedestrianState.CLEARING:
                p.progress += (PED_SPEED / G.CROSSING_LENGTH) * dt

            if p.progress >= 1.0:
                p.state = PedestrianState.CLEARED
                if p.cleared_timestamp is None:
                    p.cleared_timestamp = round(self.time, 2)
                self.completed_pedestrians.append(p)

        # 7. Only remove pedestrians who have completed their crossing (CLEARED)
        self.pedestrians = [p for p in self.pedestrians if p.state != PedestrianState.CLEARED]

    # -------------------------------------------------------------------------
    # Vehicle Synchronization
    # -------------------------------------------------------------------------
    def _sync_vehicles(self, res: PipelineFrameResult, dt: float) -> None:
        """Map vehicle demand to representative approach vehicles with realistic physics."""
        active_approaches: Set[int] = set()

        for det in res.spatial_detections:
            if det.approach and det.zone_type in (ZoneType.APPROACH, ZoneType.QUEUE):
                app_idx = APPROACH_TO_INDEX.get(det.approach)
                if app_idx is not None:
                    active_approaches.add(app_idx)

        # If approach zones were general, default corridor is North-South (0 and 2)
        if not active_approaches:
            if res.vehicle_demand > 0 or res.approaching_vehicles > 0 or res.queued_vehicles > 0:
                active_approaches = {0, 2}
            elif self.ctrl.vehicle_signal == VehicleSignal.GREEN:
                active_approaches = {0, 2}

        # Spawn representative vehicles on active approaches if lane is open
        for a in active_approaches:
            line = G.STOP_LINE_S[a]
            vehs_on_arm = [v for v in self.vehicles if v.approach == a and v.front < line + 80.0]
            target_count = 2 if self.ctrl.vehicle_signal != VehicleSignal.GREEN else 1
            if len(vehs_on_arm) < target_count:
                kind = VehicleKind.CAR
                length = KIND_SPECS[kind][0]
                start_s = -length / 2
                entrance_clear = all(
                    v.rear > start_s + length + MIN_GAP + 15 for v in self.vehicles if v.approach == a
                )
                if entrance_clear:
                    initial_speed = 52.0 if self.ctrl.vehicle_signal == VehicleSignal.GREEN else 35.0
                    self.vehicles.append(
                        VehicleAgent(
                            id=self._next_vid,
                            approach=a,
                            kind=kind,
                            s=start_s,
                            speed=initial_speed,
                        )
                    )
                    self._next_vid += 1

        # Advance vehicles using BloomCityEngine physics
        self._update_vehicles(dt)

    # -------------------------------------------------------------------------
    # Frame Recording
    # -------------------------------------------------------------------------
    # Frame Recording
    # -------------------------------------------------------------------------
    def _record_current_frame(self, force: bool = True) -> dict:
        """Record the current simulation state into recorded_frames."""
        if self.initial_timestamp is None:
            self.initial_timestamp = self.time

        t_rel = round(max(0.0, self.time - self.initial_timestamp), 3)
        interval = 1.0 / self.target_fps

        should_record = (
            force
            or not self.recorded_frames
            or (self.time - self.last_recorded_time >= interval - 0.01)
        )

        # Ensure t is strictly increasing across frames so JS never divides by 0 or freezes
        if self.recorded_frames and t_rel <= self.recorded_frames[-1]["t"]:
            t_rel = round(self.recorded_frames[-1]["t"] + 0.05, 3)

        st = self.ctrl
        if st.decision_reason not in self.reason_table:
            self.reason_table.append(st.decision_reason)

        vsig = ("GREEN", "YELLOW", "RED").index(st.vehicle_signal.value)
        psig = ("DONT_WALK", "WALK", "CLEARING").index(st.pedestrian_signal.value)
        dec = ("VEHICLE_PRIORITY", "PEDESTRIAN_PRIORITY", "FIXED_TIME_FALLBACK").index(
            st.decision.value if st.decision in Decision else Decision.FIXED_TIME_FALLBACK.value
        )

        veh = []
        for v in self.vehicles:
            x, y = v.position
            veh.append([
                v.id,
                round(x, 1),
                round(y, 1),
                G.HEADING[v.approach],
                KIND_ORDER.index(v.kind),
                1 if v.state != VehicleState.MOVING else 0,
            ])

        ped = []
        for p in self.pedestrians:
            x, y = p.position
            is_crossing = 1 if p.state in (PedestrianState.CROSSING, PedestrianState.ENTERING, PedestrianState.CLEARING) else 0
            ped.append([
                p.id,
                round(x, 1),
                round(y, 1),
                is_crossing,
                round(p.heading),
            ])

        d = self.demand
        waiting_count = sum(1 for p in self.pedestrians if p.state == PedestrianState.WAITING)
        crossing_count = sum(1 for p in self.pedestrians if p.state in (PedestrianState.CROSSING, PedestrianState.ENTERING, PedestrianState.CLEARING))
        frame_dict = {
            "t": t_rel,
            "ph": PHASE_ORDER.index(st.current_phase),
            "vs": vsig,
            "ps": psig,
            "dc": dec,
            "rs": self.reason_table.index(st.decision_reason),
            "rm": round(st.time_remaining, 1),
            "fs": int(st.fail_safe_active),
            "nv": int(d.vehicle_demand) if d else 0,
            "np": waiting_count,
            "nc": crossing_count,
            "nq": int(d.vehicle_queue) if d else 0,
            "na": int(self.approaching),
            "lw": round(float(d.longest_pedestrian_wait_s), 1) if d else 0.0,
            "vt": round(float(self.time), 3),
            "frame_index": self.last_frame_index,
            "v": veh,
            "p": ped,
        }

        if should_record:
            self.recorded_frames.append(frame_dict)
            self.last_recorded_time = self.time

        # Cap recorded frames buffer to prevent unbounded growth during long sessions
        if len(self.recorded_frames) > 500:
            self.recorded_frames = self.recorded_frames[-500:]

        return frame_dict

    # -------------------------------------------------------------------------
    # Episode Continuation (Video Observation vs Simulation Time Decoupling)
    # -------------------------------------------------------------------------
    def should_continue_episode(self) -> bool:
        """Check whether the simulation episode should continue after video observation ends.

        Returns True if:
        1. A pedestrian phase is active or in transition (yellow, all-red, walk, clearance).
        2. Any representative pedestrian is currently crossing or clearing.
        3. Pedestrians are waiting and vehicle green has elapsed (or is within 1.0s of expiring).
        """
        # 1. Controller is in active pedestrian or transition phase
        if self.ctrl.current_phase != Phase.VEHICLE_GREEN:
            return True

        # 2. Any pedestrian is in crosswalk
        if any(p.state in (PedestrianState.ENTERING, PedestrianState.CROSSING, PedestrianState.CLEARING) for p in self.pedestrians):
            return True

        # 3. Pedestrians are waiting and min vehicle green is elapsed or close to expiring
        waiting_count = sum(1 for p in self.pedestrians if p.state == PedestrianState.WAITING)
        if waiting_count > 0:
            left_green = self.controller.config.min_vehicle_green - self.controller._elapsed
            if left_green <= 1.0:
                return True

        return False

    def continue_simulation_episode(
        self,
        max_duration: float = 35.0,
        dt: Optional[float] = None,
    ) -> List[dict]:
        """Continue the active controller episode after video observation ends.

        Separates video observation time from simulation time: runs the authoritative
        Adaptive FSM and Bloom City physical simulation until the active pedestrian
        crossing episode completes (phase returns to VEHICLE_GREEN and crosswalk clears)
        or max_duration is reached.
        """
        if not self.should_continue_episode():
            return []

        step_dt = dt if dt is not None else (1.0 / self.target_fps)
        initial_time = self.time
        continuation_frames: List[dict] = []
        saw_pedestrian_phase = (self.ctrl.current_phase in (Phase.PEDESTRIAN_WALK, Phase.PEDESTRIAN_CLEARANCE))
        green_after_ped_time = 0.0

        while (self.time - initial_time) < max_duration:
            # 1. Measure demand from current simulation agents
            waiting_count = sum(1 for p in self.pedestrians if p.state == PedestrianState.WAITING)
            crossing_count = sum(1 for p in self.pedestrians if p.state in (PedestrianState.ENTERING, PedestrianState.CROSSING, PedestrianState.CLEARING))
            max_wait = max((p.wait_time for p in self.pedestrians if p.state == PedestrianState.WAITING), default=0.0)
            v_dem = self.demand.vehicle_demand if self.demand is not None else 3
            v_queue = self.demand.vehicle_queue if self.demand is not None else 0

            sim_demand = DemandSnapshot(
                pedestrians_waiting=waiting_count,
                pedestrians_in_crosswalk=crossing_count,
                longest_pedestrian_wait_s=max_wait,
                vehicle_demand=v_dem,
                vehicle_queue=v_queue,
                source="simulation_continuation",
            )

            # 2. Advance authoritative controller
            self.ctrl = self.controller.step(dt=step_dt, demand=sim_demand)
            self.demand = sim_demand
            self.time += step_dt

            if self.ctrl.current_phase in (Phase.PEDESTRIAN_WALK, Phase.PEDESTRIAN_CLEARANCE):
                saw_pedestrian_phase = True

            # 3. Advance pedestrian agents
            walking = (self.ctrl.pedestrian_signal == PedestrianSignal.WALK)
            veh_red = (self.ctrl.vehicle_signal == VehicleSignal.RED)
            clearing = (self.ctrl.pedestrian_signal == PedestrianSignal.CLEARING)

            for p in self.pedestrians:
                if p.state == PedestrianState.WAITING:
                    p.wait_time += step_dt
                    if walking and veh_red and not self._crosswalk_has_vehicle(p.crosswalk):
                        p.state = PedestrianState.CROSSING
                        if p.entering_timestamp is None:
                            p.entering_timestamp = round(self.time, 2)
                        if p.crossing_timestamp is None:
                            p.crossing_timestamp = round(self.time, 2)
                if p.state in (PedestrianState.ENTERING, PedestrianState.CROSSING):
                    p.state = PedestrianState.CROSSING
                    if p.crossing_timestamp is None:
                        p.crossing_timestamp = round(self.time, 2)
                    p.progress += (PED_SPEED / G.CROSSING_LENGTH) * step_dt
                    if clearing:
                        p.state = PedestrianState.CLEARING
                        if p.clearing_timestamp is None:
                            p.clearing_timestamp = round(self.time, 2)
                elif p.state == PedestrianState.CLEARING:
                    p.progress += (PED_SPEED / G.CROSSING_LENGTH) * step_dt

                if p.progress >= 1.0:
                    p.state = PedestrianState.CLEARED
                    if p.cleared_timestamp is None:
                        p.cleared_timestamp = round(self.time, 2)
                    self.completed_pedestrians.append(p)

            self.pedestrians = [p for p in self.pedestrians if p.state != PedestrianState.CLEARED]

            # 4. Advance vehicle agents with Bloom City physics
            self._update_vehicles(step_dt)

            # 5. Record frame
            frame_dict = self._record_current_frame(force=True)
            continuation_frames.append(frame_dict)

            # 6. Check termination condition
            if saw_pedestrian_phase and self.ctrl.current_phase == Phase.VEHICLE_GREEN:
                remaining_crossing = sum(1 for p in self.pedestrians if p.state in (PedestrianState.ENTERING, PedestrianState.CROSSING, PedestrianState.CLEARING))
                if remaining_crossing == 0:
                    green_after_ped_time += step_dt
                    if green_after_ped_time >= 2.0:
                        break

        return continuation_frames

    # -------------------------------------------------------------------------
    # Render & Export Helpers
    # -------------------------------------------------------------------------
    def get_render_frames(self) -> List[dict]:
        """Frames for the renderer: exactly the recorded frames, with time rebased to 0.

        The recording buffer is capped, so the first kept frame can have t > 0; the player needs a
        zero-based, strictly increasing clock. A single recorded frame is duplicated unchanged
        (same signals, same positions) so the player always has two points - no motion is invented.
        """
        if not self.recorded_frames:
            return []
        base = self.recorded_frames[0]["t"]
        frames = [dict(f, t=round(f["t"] - base, 3)) for f in self.recorded_frames]
        if len(frames) == 1:
            twin = dict(frames[0])
            twin["t"] = round(frames[0]["t"] + 0.1, 3)
            frames.append(twin)
        return frames

    def render_html(self, theme: str = "day", speed: float = 1.0) -> str:
        """Render self-contained HTML for Live AI Control mode."""
        frames = self.get_render_frames()
        if not frames:
            return ""

        clip_str = f"Clip: {self.video_name} · " if self.video_name else ""
        frame_str = f"Frame {self.last_frame_index}" if self.last_frame_index >= 0 else ""
        scenario_title = f"{clip_str}{frame_str}".strip(" · ")

        return render_bloom_city_html(
            frames=frames,
            reasons=list(self.reason_table),
            theme=theme,
            mode="LIVE AI CONTROL: Driven by video telemetry",
            scenario_title=scenario_title,
            source_note=LIVE_SOURCE_NOTE,
            speed=speed,
        )

    # -------------------------------------------------------------------------
    # Properties & Status
    # -------------------------------------------------------------------------
    @property
    def current_phase(self) -> Phase:
        return self.ctrl.current_phase

    @property
    def vehicle_signal(self) -> VehicleSignal:
        return self.ctrl.vehicle_signal

    @property
    def pedestrian_signal(self) -> PedestrianSignal:
        return self.ctrl.pedestrian_signal

    @property
    def is_safe(self) -> bool:
        """Safety invariant: vehicles cannot have green/yellow while pedestrians have walk."""
        return not (
            self.vehicle_signal in (VehicleSignal.GREEN, VehicleSignal.YELLOW)
            and self.pedestrian_signal == PedestrianSignal.WALK
        )

    @property
    def waiting_pedestrians_count(self) -> int:
        return sum(1 for p in self.pedestrians if p.state == PedestrianState.WAITING)

    @property
    def crossing_pedestrians_count(self) -> int:
        return sum(1 for p in self.pedestrians if p.state in (PedestrianState.ENTERING, PedestrianState.CROSSING, PedestrianState.CLEARING))

    @property
    def duration_s(self) -> float:
        if not self.recorded_frames:
            return 0.0
        return self.recorded_frames[-1]["t"]

    def get_pedestrian_lifecycle_summary(self) -> List[dict]:
        """Return lifecycle history for all representative pedestrians (active & completed)."""
        all_peds = list(self.completed_pedestrians) + list(self.pedestrians)
        seen_ids = set()
        out = []
        for p in all_peds:
            if p.id in seen_ids:
                continue
            seen_ids.add(p.id)
            out.append({
                "id": p.id,
                "track_id": p.track_id,
                "source_zone": p.source_zone or (f"{INDEX_TO_APPROACH[p.crosswalk].value}_CROSSWALK"),
                "crosswalk": p.crosswalk,
                "approach": INDEX_TO_APPROACH[p.crosswalk].value,
                "waiting_timestamp": p.waiting_timestamp,
                "entering_timestamp": p.entering_timestamp,
                "crossing_timestamp": p.crossing_timestamp,
                "clearing_timestamp": p.clearing_timestamp,
                "cleared_timestamp": p.cleared_timestamp,
                "state": p.state.value,
                "progress": round(p.progress, 3),
            })
        return out
