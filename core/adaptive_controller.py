"""
CrossGuard Adaptive Pedestrian-Priority Signal Controller
=========================================================
Finite State Machine (FSM) implementing dynamic pedestrian priority, group accumulation,
maximum waiting time bounds, and crosswalk occupancy safety interlocks.
Strictly adheres to research simulation safety standards with conservative fail-safe states.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from config.settings import SignalFSMConfig
from core.spatial_analysis import SpatialSnapshot


class SignalPhase(str, Enum):
    """Discrete operational phases of the intersection signal controller."""
    VEHICLE_GREEN = "VEHICLE_GREEN"
    VEHICLE_YELLOW = "VEHICLE_YELLOW"
    ALL_RED_CLEARANCE = "ALL_RED_CLEARANCE"
    PEDESTRIAN_WALK = "PEDESTRIAN_WALK"
    PEDESTRIAN_CLEARANCE = "PEDESTRIAN_CLEARANCE"
    FAIL_SAFE_STOP = "FAIL_SAFE_STOP"


@dataclass
class ControllerDecision:
    """Explains a state transition or phase continuation decision."""
    timestamp: float
    current_phase: SignalPhase
    previous_phase: SignalPhase
    phase_elapsed_seconds: float
    phase_remaining_seconds: float
    decision_case: str  # "CASE_A", "CASE_B", "CASE_C", "CASE_D", "CASE_E", "CASE_F", "FAIL_SAFE"
    reason: str
    pedestrians_waiting: int
    max_wait_time: float
    crosswalk_occupied: bool
    vehicles_in_approach: int
    clearance_extensions: int


class AdaptiveSignalController:
    """
    Pedestrian-priority adaptive traffic signal controller finite state machine.
    """

    def __init__(self, config: Optional[SignalFSMConfig] = None):
        self.config = config or SignalFSMConfig()

        # State Variables
        self.current_phase: SignalPhase = SignalPhase.VEHICLE_GREEN
        self.previous_phase: SignalPhase = SignalPhase.VEHICLE_GREEN
        self.phase_start_time: float = 0.0
        self.phase_elapsed: float = 0.0
        self.current_nominal_duration: float = self.config.min_vehicle_green

        # Pedestrian Accumulation & Priority Tracking
        self.accumulation_start_time: Optional[float] = None
        self.clearance_extensions_count: int = 0
        self.consecutive_clear_observations: int = 0

        # Safety & Fault State
        self.fault_active: bool = False
        self.fault_reason: str = ""

        # Audit History
        self.decision_history: List[ControllerDecision] = []
        self.last_decision: Optional[ControllerDecision] = None

    def trigger_fault(self, reason: str, timestamp: float = 0.0) -> None:
        """Forces the controller into conservative fail-safe state."""
        self.fault_active = True
        self.fault_reason = reason
        self._transition_to(SignalPhase.FAIL_SAFE_STOP, timestamp, "FAIL_SAFE", f"Fault activated: {reason}")
        decision = ControllerDecision(
            timestamp=timestamp,
            current_phase=SignalPhase.FAIL_SAFE_STOP,
            previous_phase=self.previous_phase,
            phase_elapsed_seconds=0.0,
            phase_remaining_seconds=0.0,
            decision_case="FAIL_SAFE",
            reason=f"CRITICAL SAFETY FAULT: {reason}. All conflicting movements stopped.",
            pedestrians_waiting=self.last_decision.pedestrians_waiting if self.last_decision else 0,
            max_wait_time=self.last_decision.max_wait_time if self.last_decision else 0.0,
            crosswalk_occupied=self.last_decision.crosswalk_occupied if self.last_decision else False,
            vehicles_in_approach=self.last_decision.vehicles_in_approach if self.last_decision else 0,
            clearance_extensions=self.clearance_extensions_count,
        )
        self.last_decision = decision
        self.decision_history.append(decision)

    def clear_fault(self, timestamp: float = 0.0) -> None:
        """Restores controller from fail-safe into initial vehicular green."""
        self.fault_active = False
        self.fault_reason = ""
        self._transition_to(SignalPhase.VEHICLE_GREEN, timestamp, "RESET", "Fault cleared; restarting fail-safe cycle.")
        decision = ControllerDecision(
            timestamp=timestamp,
            current_phase=SignalPhase.VEHICLE_GREEN,
            previous_phase=self.previous_phase,
            phase_elapsed_seconds=0.0,
            phase_remaining_seconds=self.current_nominal_duration,
            decision_case="RESET",
            reason="Fault cleared; restarting fail-safe cycle.",
            pedestrians_waiting=self.last_decision.pedestrians_waiting if self.last_decision else 0,
            max_wait_time=self.last_decision.max_wait_time if self.last_decision else 0.0,
            crosswalk_occupied=self.last_decision.crosswalk_occupied if self.last_decision else False,
            vehicles_in_approach=self.last_decision.vehicles_in_approach if self.last_decision else 0,
            clearance_extensions=0,
        )
        self.last_decision = decision
        self.decision_history.append(decision)

    def step(self, snapshot: SpatialSnapshot, timestamp: float) -> ControllerDecision:
        """
        Executes one evaluation step of the finite state machine.

        Args:
            snapshot: Validated spatial measurements from SpatialZoneEngine.
            timestamp: Video elapsed time in seconds.

        Returns:
            ControllerDecision detailing current phase, countdown, and explanation.
        """
        # If fault is active, hold fail-safe state
        if self.fault_active:
            decision = ControllerDecision(
                timestamp=timestamp,
                current_phase=SignalPhase.FAIL_SAFE_STOP,
                previous_phase=self.previous_phase,
                phase_elapsed_seconds=round(timestamp - self.phase_start_time, 1),
                phase_remaining_seconds=0.0,
                decision_case="FAIL_SAFE",
                reason=f"CRITICAL SAFETY FAULT: {self.fault_reason}. All conflicting movements stopped.",
                pedestrians_waiting=snapshot.waiting_pedestrians_count,
                max_wait_time=snapshot.max_pedestrian_waiting_time,
                crosswalk_occupied=snapshot.crosswalk_occupied,
                vehicles_in_approach=snapshot.approach_vehicles_count,
                clearance_extensions=self.clearance_extensions_count,
            )
            self.last_decision = decision
            return decision

        self.phase_elapsed = max(0.0, timestamp - self.phase_start_time)
        phase_remaining = max(0.0, self.current_nominal_duration - self.phase_elapsed)

        # -------------------------------------------------------------
        # STATE 1: VEHICLE_GREEN
        # -------------------------------------------------------------
        if self.current_phase == SignalPhase.VEHICLE_GREEN:
            min_green_met = self.phase_elapsed >= self.config.min_vehicle_green

            # Case A: No pedestrians waiting
            if snapshot.waiting_pedestrians_count == 0:
                self.accumulation_start_time = None
                reason = "Case A: No pedestrians waiting. Arterial vehicle green maintained."
                decision = self._create_decision(
                    timestamp, "CASE_A", reason, phase_remaining, snapshot
                )
                return decision

            # Pedestrians are present -> start accumulation timer if not already active
            if self.accumulation_start_time is None:
                self.accumulation_start_time = timestamp

            accumulation_duration = timestamp - self.accumulation_start_time

            # Case D: Maximum waiting time reached (starvation prevention)
            if snapshot.max_pedestrian_waiting_time >= self.config.max_waiting_time and min_green_met:
                reason = (
                    f"Case D: Maximum pedestrian waiting time reached ({snapshot.max_pedestrian_waiting_time:.1f}s >= "
                    f"{self.config.max_waiting_time:.1f}s). Priority granted to prevent indefinite delay."
                )
                self._transition_to(SignalPhase.VEHICLE_YELLOW, timestamp, "CASE_D", reason)
                return self._create_decision(timestamp, "CASE_D", reason, self.current_nominal_duration, snapshot)

            # Case C: Group threshold reached
            if snapshot.waiting_pedestrians_count >= self.config.pedestrian_group_threshold and min_green_met:
                reason = (
                    f"Case C: Pedestrian group threshold reached ({snapshot.waiting_pedestrians_count} >= "
                    f"{self.config.pedestrian_group_threshold}). Scheduling vehicle transition."
                )
                self._transition_to(SignalPhase.VEHICLE_YELLOW, timestamp, "CASE_C", reason)
                return self._create_decision(timestamp, "CASE_C", reason, self.current_nominal_duration, snapshot)

            # Case B: Small number of pedestrians arriving (< group threshold)
            if accumulation_duration < self.config.accumulation_interval:
                rem_acc = self.config.accumulation_interval - accumulation_duration
                reason = (
                    f"Case B: Small pedestrian group ({snapshot.waiting_pedestrians_count} peds); "
                    f"accumulation interval active ({accumulation_duration:.1f}s / {self.config.accumulation_interval:.1f}s, {rem_acc:.1f}s remaining)."
                )
                return self._create_decision(timestamp, "CASE_B", reason, phase_remaining, snapshot)
            else:
                # Accumulation window expired
                if min_green_met:
                    reason = (
                        f"Case B: Pedestrian accumulation interval ({self.config.accumulation_interval:.1f}s) expired. "
                        f"Serving small pedestrian group ({snapshot.waiting_pedestrians_count} peds)."
                    )
                    self._transition_to(SignalPhase.VEHICLE_YELLOW, timestamp, "CASE_B", reason)
                    return self._create_decision(timestamp, "CASE_B", reason, self.current_nominal_duration, snapshot)
                else:
                    reason = "Case B: Accumulation complete; holding vehicle green until minimum green safety boundary."
                    return self._create_decision(timestamp, "CASE_B", reason, phase_remaining, snapshot)

        # -------------------------------------------------------------
        # STATE 2: VEHICLE_YELLOW
        # -------------------------------------------------------------
        elif self.current_phase == SignalPhase.VEHICLE_YELLOW:
            if self.phase_elapsed >= self.config.vehicle_yellow_duration:
                reason = f"Vehicle yellow change interval ({self.config.vehicle_yellow_duration:.1f}s) complete. Transitioning to all-red clearance."
                self._transition_to(SignalPhase.ALL_RED_CLEARANCE, timestamp, "SAFETY_TRANSITION", reason)
                return self._create_decision(timestamp, "TRANSITION", reason, self.current_nominal_duration, snapshot)
            else:
                reason = f"Vehicle yellow clearance active ({phase_remaining:.1f}s remaining)."
                return self._create_decision(timestamp, "SAFETY_TRANSITION", reason, phase_remaining, snapshot)

        # -------------------------------------------------------------
        # STATE 3: ALL_RED_CLEARANCE
        # -------------------------------------------------------------
        elif self.current_phase == SignalPhase.ALL_RED_CLEARANCE:
            if self.phase_elapsed >= self.config.all_red_clearance_duration:
                reason = f"All-red clearance interval ({self.config.all_red_clearance_duration:.1f}s) complete. Protected pedestrian walk phase granted."
                self._transition_to(SignalPhase.PEDESTRIAN_WALK, timestamp, "SAFETY_TRANSITION", reason)
                return self._create_decision(timestamp, "TRANSITION", reason, self.current_nominal_duration, snapshot)
            else:
                reason = f"All-red clearance active ({phase_remaining:.1f}s remaining). All conflicting movements halted."
                return self._create_decision(timestamp, "SAFETY_TRANSITION", reason, phase_remaining, snapshot)

        # -------------------------------------------------------------
        # STATE 4: PEDESTRIAN_WALK
        # -------------------------------------------------------------
        elif self.current_phase == SignalPhase.PEDESTRIAN_WALK:
            if self.phase_elapsed >= self.config.pedestrian_walk_duration:
                reason = f"Pedestrian walk interval ({self.config.pedestrian_walk_duration:.1f}s) complete. Initiating flashing pedestrian clearance."
                self._transition_to(SignalPhase.PEDESTRIAN_CLEARANCE, timestamp, "SAFETY_TRANSITION", reason)
                return self._create_decision(timestamp, "TRANSITION", reason, self.current_nominal_duration, snapshot)
            else:
                reason = f"Pedestrian WALK active ({phase_remaining:.1f}s remaining). Pedestrians permitted to initiate crossing."
                return self._create_decision(timestamp, "PEDESTRIAN_WALK", reason, phase_remaining, snapshot)

        # -------------------------------------------------------------
        # STATE 5: PEDESTRIAN_CLEARANCE
        # -------------------------------------------------------------
        elif self.current_phase == SignalPhase.PEDESTRIAN_CLEARANCE:
            nominal_elapsed = self.phase_elapsed >= self.current_nominal_duration

            # Case E: Pedestrians remain on the crossing
            if snapshot.crosswalk_occupied:
                self.consecutive_clear_observations = 0
                if nominal_elapsed:
                    if self.clearance_extensions_count < self.config.max_clearance_extensions:
                        self.clearance_extensions_count += 1
                        self.current_nominal_duration += self.config.extension_seconds
                        reason = (
                            f"Case E: Pedestrians remain on crossing ({snapshot.crossing_pedestrians_count} active)! "
                            f"Extending clearance by {self.config.extension_seconds:.1f}s "
                            f"(Extension {self.clearance_extensions_count}/{self.config.max_clearance_extensions}). Conflicting vehicles held."
                        )
                        return self._create_decision(timestamp, "CASE_E", reason, self.config.extension_seconds, snapshot)
                    else:
                        reason = (
                            f"Case E: Crossing occupied but max clearance extensions reached ({self.config.max_clearance_extensions}). "
                            "Maintaining conservative hold for pedestrian safety."
                        )
                        return self._create_decision(timestamp, "CASE_E", reason, 0.0, snapshot)
                else:
                    reason = f"Pedestrian clearance active ({phase_remaining:.1f}s remaining). Crossing occupied ({snapshot.crossing_pedestrians_count} peds)."
                    return self._create_decision(timestamp, "PEDESTRIAN_CLEARANCE", reason, phase_remaining, snapshot)

            # Case F: Crossing becomes clear
            else:
                if nominal_elapsed:
                    self.consecutive_clear_observations += 1
                    if self.consecutive_clear_observations >= self.config.consecutive_clear_observations:
                        reason = (
                            f"Case F: Crossing verified clear after {self.consecutive_clear_observations} "
                            "consecutive observations. Releasing conflicting vehicular traffic to green."
                        )
                        self._transition_to(SignalPhase.VEHICLE_GREEN, timestamp, "CASE_F", reason)
                        return self._create_decision(timestamp, "CASE_F", reason, self.current_nominal_duration, snapshot)
                    else:
                        rem_obs = self.config.consecutive_clear_observations - self.consecutive_clear_observations
                        reason = f"Case F: Crossing observed clear. Awaiting {rem_obs} more consecutive clear confirmation frames."
                        return self._create_decision(timestamp, "CASE_F", reason, 0.0, snapshot)
                else:
                    reason = f"Pedestrian clearance active ({phase_remaining:.1f}s remaining). Crossing currently clear."
                    return self._create_decision(timestamp, "PEDESTRIAN_CLEARANCE", reason, phase_remaining, snapshot)

        # Fallback default
        reason = f"Operating in {self.current_phase.value}."
        return self._create_decision(timestamp, "DEFAULT", reason, phase_remaining, snapshot)

    def _transition_to(
        self,
        new_phase: SignalPhase,
        timestamp: float,
        case_label: str,
        reason: str,
    ) -> None:
        """Executes a validated phase transition and sets appropriate nominal timer."""
        self.previous_phase = self.current_phase
        self.current_phase = new_phase
        self.phase_start_time = timestamp
        self.phase_elapsed = 0.0

        if new_phase == SignalPhase.VEHICLE_GREEN:
            self.current_nominal_duration = self.config.min_vehicle_green
            self.accumulation_start_time = None
            self.clearance_extensions_count = 0
            self.consecutive_clear_observations = 0
        elif new_phase == SignalPhase.VEHICLE_YELLOW:
            self.current_nominal_duration = self.config.vehicle_yellow_duration
        elif new_phase == SignalPhase.ALL_RED_CLEARANCE:
            self.current_nominal_duration = self.config.all_red_clearance_duration
        elif new_phase == SignalPhase.PEDESTRIAN_WALK:
            self.current_nominal_duration = self.config.pedestrian_walk_duration
        elif new_phase == SignalPhase.PEDESTRIAN_CLEARANCE:
            self.current_nominal_duration = self.config.pedestrian_clearance_duration
            self.clearance_extensions_count = 0
            self.consecutive_clear_observations = 0
        elif new_phase == SignalPhase.FAIL_SAFE_STOP:
            self.current_nominal_duration = 0.0

    def _create_decision(
        self,
        timestamp: float,
        case_label: str,
        reason: str,
        remaining_seconds: float,
        snapshot: SpatialSnapshot,
    ) -> ControllerDecision:
        """Constructs an auditable decision record."""
        decision = ControllerDecision(
            timestamp=round(timestamp, 2),
            current_phase=self.current_phase,
            previous_phase=self.previous_phase,
            phase_elapsed_seconds=round(self.phase_elapsed, 1),
            phase_remaining_seconds=round(max(0.0, remaining_seconds), 1),
            decision_case=case_label,
            reason=reason,
            pedestrians_waiting=snapshot.waiting_pedestrians_count,
            max_wait_time=snapshot.max_pedestrian_waiting_time,
            crosswalk_occupied=snapshot.crosswalk_occupied,
            vehicles_in_approach=snapshot.approach_vehicles_count,
            clearance_extensions=self.clearance_extensions_count,
        )
        self.last_decision = decision
        self.decision_history.append(decision)
        if len(self.decision_history) > 500:
            self.decision_history.pop(0)
        return decision
