"""Shared data models for the CrossGuard controller. No Streamlit / UI imports."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Tuple


class Phase(str, Enum):
    VEHICLE_GREEN = "VEHICLE_GREEN"
    VEHICLE_YELLOW = "VEHICLE_YELLOW"
    ALL_RED_CLEARANCE = "ALL_RED_CLEARANCE"
    PEDESTRIAN_WALK = "PEDESTRIAN_WALK"
    PEDESTRIAN_CLEARANCE = "PEDESTRIAN_CLEARANCE"


PHASE_ORDER = (
    Phase.VEHICLE_GREEN,
    Phase.VEHICLE_YELLOW,
    Phase.ALL_RED_CLEARANCE,
    Phase.PEDESTRIAN_WALK,
    Phase.PEDESTRIAN_CLEARANCE,
)

PHASE_LABELS: Dict[Phase, str] = {
    Phase.VEHICLE_GREEN: "Vehicle Green",
    Phase.VEHICLE_YELLOW: "Yellow",
    Phase.ALL_RED_CLEARANCE: "All Red",
    Phase.PEDESTRIAN_WALK: "Pedestrian Walk",
    Phase.PEDESTRIAN_CLEARANCE: "Clearance",
}


class VehicleSignal(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

    @property
    def label(self) -> str:
        return {"GREEN": "GO", "YELLOW": "CAUTION", "RED": "STOP"}[self.value]


class PedestrianSignal(str, Enum):
    WALK = "WALK"
    DONT_WALK = "DONT_WALK"
    CLEARING = "CLEARING"

    @property
    def label(self) -> str:
        return {"WALK": "CROSS", "DONT_WALK": "WAIT", "CLEARING": "FINISH CROSSING"}[self.value]


class Decision(str, Enum):
    VEHICLE_PRIORITY = "VEHICLE_PRIORITY"
    PEDESTRIAN_PRIORITY = "PEDESTRIAN_PRIORITY"
    FIXED_TIME_FALLBACK = "FIXED_TIME_FALLBACK"


# The single source of truth for FSM state -> signal outputs.
PHASE_OUTPUTS: Dict[Phase, Tuple[VehicleSignal, PedestrianSignal]] = {
    Phase.VEHICLE_GREEN: (VehicleSignal.GREEN, PedestrianSignal.DONT_WALK),
    Phase.VEHICLE_YELLOW: (VehicleSignal.YELLOW, PedestrianSignal.DONT_WALK),
    Phase.ALL_RED_CLEARANCE: (VehicleSignal.RED, PedestrianSignal.DONT_WALK),
    Phase.PEDESTRIAN_WALK: (VehicleSignal.RED, PedestrianSignal.WALK),
    Phase.PEDESTRIAN_CLEARANCE: (VehicleSignal.RED, PedestrianSignal.CLEARING),
}


@dataclass
class ControllerConfig:
    """Research safety parameters (MUTCD-inspired, not a compliance claim). Seconds unless noted."""
    min_vehicle_green: float = 10.0
    max_vehicle_green: float = 40.0
    yellow_time: float = 3.5
    all_red_time: float = 3.0
    pedestrian_walk_time: float = 8.0
    pedestrian_clearance_time: float = 8.0
    max_clearance_extension: float = 6.0   # extra clearance while someone is still in the crosswalk
    # demand rules
    pedestrian_demand_threshold: int = 2   # waiting pedestrians that justify a walk phase
    single_pedestrian_wait: float = 15.0   # a lone pedestrian is served after this wait
    max_pedestrian_wait: float = 30.0      # forced service regardless of vehicle demand
    vehicle_release_threshold: int = 7     # vehicle demand at/above this is "heavy"
    # fail-safe
    failsafe_after: float = 3.0            # seconds of invalid/missing telemetry before fallback
    fixed_green_time: float = 25.0         # fixed-time fallback vehicle green

    def __post_init__(self) -> None:
        for name in ("min_vehicle_green", "max_vehicle_green", "yellow_time", "all_red_time",
                     "pedestrian_walk_time", "pedestrian_clearance_time", "failsafe_after",
                     "fixed_green_time"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a positive finite number, got {v!r}")
        if self.max_vehicle_green < self.min_vehicle_green:
            raise ValueError("max_vehicle_green must be >= min_vehicle_green")
        if self.fixed_green_time < self.min_vehicle_green:
            raise ValueError("fixed_green_time must be >= min_vehicle_green")
        if self.max_clearance_extension < 0:
            raise ValueError("max_clearance_extension must be >= 0")
        if self.pedestrian_demand_threshold < 1:
            raise ValueError("pedestrian_demand_threshold must be >= 1")
        if self.max_pedestrian_wait < self.single_pedestrian_wait:
            raise ValueError("max_pedestrian_wait must be >= single_pedestrian_wait")


@dataclass
class DemandSnapshot:
    """Spatially-relevant demand handed to the controller (never raw session totals)."""
    pedestrians_waiting: int = 0
    longest_pedestrian_wait_s: float = 0.0
    pedestrians_in_crosswalk: int = 0
    vehicle_demand: int = 0     # vehicles approaching or queued
    vehicle_queue: int = 0      # subset that is stopped
    source: str = "unspecified"  # "live" | "demo" | "simulation"

    def is_valid(self) -> bool:
        nums = (self.pedestrians_waiting, self.longest_pedestrian_wait_s,
                self.pedestrians_in_crosswalk, self.vehicle_demand, self.vehicle_queue)
        return all(isinstance(n, (int, float)) and math.isfinite(n) and n >= 0 for n in nums)


@dataclass(frozen=True)
class ControllerState:
    """Everything the simulation / UI needs from the controller."""
    current_phase: Phase
    vehicle_signal: VehicleSignal
    pedestrian_signal: PedestrianSignal
    decision: Decision
    decision_reason: str
    phase_elapsed: float
    phase_target: float        # planned duration (minimum, for VEHICLE_GREEN)
    time_remaining: float      # seconds left of the planned duration (>= 0)
    fail_safe_active: bool
    total_time: float
    cycles_completed: int
    pedestrian_phases_served: int

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("current_phase", "vehicle_signal", "pedestrian_signal", "decision"):
            d[k] = d[k].value if isinstance(d[k], Enum) else d[k]
        return d
