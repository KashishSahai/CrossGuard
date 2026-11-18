"""Representative simulation agents (not one-to-one YOLO tracks)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from . import geometry as G


class VehicleKind(str, Enum):
    CAR = "car"
    VAN = "van"
    BUS = "bus"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"


# length, width (px), top speed (px/s), acceleration (px/s^2)
KIND_SPECS = {
    VehicleKind.CAR: (46, 24, 92, 70),
    VehicleKind.VAN: (54, 26, 86, 60),
    VehicleKind.BUS: (88, 28, 74, 42),
    VehicleKind.TRUCK: (76, 28, 78, 46),
    VehicleKind.MOTORCYCLE: (26, 11, 100, 90),
}
KIND_ORDER = list(KIND_SPECS)


class VehicleState(str, Enum):
    MOVING = "MOVING"
    BRAKING = "BRAKING"
    STOPPED = "STOPPED"


class PedestrianState(str, Enum):
    WAITING = "WAITING"
    ENTERING = "ENTERING"
    CROSSING = "CROSSING"
    CLEARING = "CLEARING"
    CLEARED = "CLEARED"


@dataclass
class VehicleAgent:
    id: int
    approach: int            # 0..3 (N, E, S, W)
    kind: VehicleKind
    s: float                 # centre position along the lane path
    speed: float = 0.0
    state: VehicleState = VehicleState.MOVING
    committed: bool = False  # past / unable to stop before the stop line

    @property
    def length(self) -> float:
        return KIND_SPECS[self.kind][0]

    @property
    def width(self) -> float:
        return KIND_SPECS[self.kind][1]

    @property
    def vmax(self) -> float:
        return KIND_SPECS[self.kind][2]

    @property
    def accel(self) -> float:
        return KIND_SPECS[self.kind][3]

    @property
    def front(self) -> float:
        return self.s + self.length / 2

    @property
    def rear(self) -> float:
        return self.s - self.length / 2

    @property
    def position(self):
        return G.lane_point(self.approach, self.s)

    @property
    def axis(self) -> int:
        return G.axis_of(self.approach)

    def bbox(self):
        x, y = self.position
        if self.approach in (0, 2):
            return (x - self.width / 2, y - self.length / 2, x + self.width / 2, y + self.length / 2)
        return (x - self.length / 2, y - self.width / 2, x + self.length / 2, y + self.width / 2)


@dataclass
class PedestrianAgent:
    id: int
    crosswalk: int           # 0..3, same index as approach arm
    direction: int           # +1: end A -> end B, -1: end B -> end A
    slot: int
    state: PedestrianState = PedestrianState.WAITING
    progress: float = 0.0    # 0..1 along the crossing
    wait_time: float = 0.0
    track_id: Optional[int] = None
    last_seen: float = 0.0
    grace_period: float = 2.0  # seconds to persist without visual detection while WAITING
    source_zone: str = ""
    waiting_timestamp: Optional[float] = None
    entering_timestamp: Optional[float] = None
    crossing_timestamp: Optional[float] = None
    clearing_timestamp: Optional[float] = None
    cleared_timestamp: Optional[float] = None

    @property
    def position(self):
        a, b = G.CROSSWALK_ENDS[self.crosswalk]
        origin, dest = (a, b) if self.direction > 0 else (b, a)
        dx, dy = G.slot_offset(self.crosswalk, self.slot)
        x0, y0 = origin[0] + dx, origin[1] + dy
        x1, y1 = dest[0] + dx, dest[1] + dy
        return x0 + (x1 - x0) * self.progress, y0 + (y1 - y0) * self.progress

    @property
    def heading(self) -> float:
        a, b = G.CROSSWALK_ENDS[self.crosswalk]
        o, d = (a, b) if self.direction > 0 else (b, a)
        import math
        return math.degrees(math.atan2(d[1] - o[1], d[0] - o[0]))
