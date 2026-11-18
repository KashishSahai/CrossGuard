"""Spatial Analysis Layer for CrossGuard.

Maps tracked 2D object positions (from future YOLO/ByteTrack or simulation)
into semantically meaningful intersection demand zones using deterministic
point-in-polygon and geometric zone rules. Pure Python; no UI dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from simulation import geometry as G


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class Approach(str, Enum):
    NORTH = "NORTH"
    EAST = "EAST"
    SOUTH = "SOUTH"
    WEST = "WEST"


class ZoneType(str, Enum):
    WAITING = "WAITING"
    CROSSING = "CROSSING"
    APPROACH = "APPROACH"
    QUEUE = "QUEUE"


class SpatialZone(str, Enum):
    # Pedestrian zones
    NORTH_WAITING = "NORTH_WAITING"
    SOUTH_WAITING = "SOUTH_WAITING"
    EAST_WAITING = "EAST_WAITING"
    WEST_WAITING = "WEST_WAITING"

    NORTH_CROSSING = "NORTH_CROSSING"
    SOUTH_CROSSING = "SOUTH_CROSSING"
    EAST_CROSSING = "EAST_CROSSING"
    WEST_CROSSING = "WEST_CROSSING"

    # Vehicle zones
    NORTH_APPROACH = "NORTH_APPROACH"
    SOUTH_APPROACH = "SOUTH_APPROACH"
    EAST_APPROACH = "EAST_APPROACH"
    WEST_APPROACH = "WEST_APPROACH"

    NORTH_QUEUE = "NORTH_QUEUE"
    SOUTH_QUEUE = "SOUTH_QUEUE"
    EAST_QUEUE = "EAST_QUEUE"
    WEST_QUEUE = "WEST_QUEUE"

    NONE = "NONE"

    @property
    def approach(self) -> Optional[Approach]:
        for app in Approach:
            if self.value.startswith(app.value + "_"):
                return app
        return None

    @property
    def zone_type(self) -> Optional[ZoneType]:
        for zt in ZoneType:
            if self.value.endswith("_" + zt.value):
                return zt
        return None

    @property
    def is_pedestrian_zone(self) -> bool:
        return self.zone_type in (ZoneType.WAITING, ZoneType.CROSSING)

    @property
    def is_vehicle_zone(self) -> bool:
        return self.zone_type in (ZoneType.APPROACH, ZoneType.QUEUE)


# -----------------------------------------------------------------------------
# Object Class Classifications
# -----------------------------------------------------------------------------

PEDESTRIAN_CLASSES: Set[str] = {
    "person", "pedestrian", "walker"
}

VEHICLE_CLASSES: Set[str] = {
    "car", "van", "bus", "truck", "motorcycle", "vehicle", "automobile"
}


def normalize_class(object_class: str) -> str:
    if not isinstance(object_class, str):
        return ""
    return object_class.strip().lower()


def is_pedestrian(object_class: str) -> bool:
    return normalize_class(object_class) in PEDESTRIAN_CLASSES


def is_vehicle(object_class: str) -> bool:
    return normalize_class(object_class) in VEHICLE_CLASSES


# -----------------------------------------------------------------------------
# Geometric Regions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RectRegion:
    """Axis-aligned rectangle with configurable boundary inclusivity."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    inclusive_min_x: bool = True
    inclusive_max_x: bool = True
    inclusive_min_y: bool = True
    inclusive_max_y: bool = True

    def contains(self, x: float, y: float) -> bool:
        if self.inclusive_min_x:
            if x < self.x_min:
                return False
        else:
            if x <= self.x_min:
                return False

        if self.inclusive_max_x:
            if x > self.x_max:
                return False
        else:
            if x >= self.x_max:
                return False

        if self.inclusive_min_y:
            if y < self.y_min:
                return False
        else:
            if y <= self.y_min:
                return False

        if self.inclusive_max_y:
            if y > self.y_max:
                return False
        else:
            if y >= self.y_max:
                return False

        return True


def point_in_polygon(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    """Deterministic ray casting point-in-polygon algorithm."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) <= y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


@dataclass
class ZoneDefinition:
    zone: SpatialZone
    approach: Optional[Approach]
    zone_type: Optional[ZoneType]
    regions: List[RectRegion] = field(default_factory=list)
    polygons: List[Sequence[Tuple[float, float]]] = field(default_factory=list)
    description: str = ""

    def contains(self, x: float, y: float) -> bool:
        for r in self.regions:
            if r.contains(x, y):
                return True
        for poly in self.polygons:
            if point_in_polygon(x, y, poly):
                return True
        return False


# -----------------------------------------------------------------------------
# Demand Data Contracts
# -----------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """Structured container for tracked perception objects."""
    id: Union[int, str]
    x: float
    y: float
    object_class: str
    speed: float = 0.0
    track_id: Optional[Union[int, str]] = None


@dataclass(frozen=True)
class ApproachDemand:
    approach: Approach
    waiting_pedestrians: int = 0
    crossing_pedestrians: int = 0
    approaching_vehicles: int = 0
    queued_vehicles: int = 0


@dataclass(frozen=True)
class SpatialDemandSummary:
    """Deterministic demand summary ready for telemetry consumption."""
    waiting_pedestrians: int = 0
    crossing_pedestrians: int = 0
    approaching_vehicles: int = 0
    queued_vehicles: int = 0
    by_approach: Dict[Approach, ApproachDemand] = field(default_factory=dict)
    zone_counts: Dict[SpatialZone, int] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Spatial Analyzer
# -----------------------------------------------------------------------------

class SpatialAnalyzer:
    """Classifies 2D object positions into intersection demand zones.

    Built directly on top of `simulation.geometry` coordinates.
    Boundary resolution policy:
      - At stop line (e.g. y = 206 for North): vehicle is QUEUE.
      - At queue boundary (e.g. y = 106 for North): vehicle is QUEUE.
      - At crosswalk curb edge (e.g. x = 430 for North): pedestrian is CROSSING.
    """

    def __init__(self, queue_depth: float = 100.0, approach_depth: float = 260.0) -> None:
        self.queue_depth = float(queue_depth)
        self.approach_depth = float(approach_depth)
        self._zones: Dict[SpatialZone, ZoneDefinition] = {}
        self._build_zones()

    def _build_zones(self) -> None:
        """Construct the 16 intersection zones using simulation.geometry."""
        qd = self.queue_depth

        # Stop line positions along the road
        # North: approach arm 0, incoming southbound lane is x in [CX - ROAD_HALF, CX] = [430, 500]
        # South: approach arm 2, incoming northbound lane is x in [CX, CX + ROAD_HALF] = [500, 570]
        # East: approach arm 1, incoming westbound lane is y in [CY - ROAD_HALF, CY] = [280, 350]
        # West: approach arm 3, incoming eastbound lane is y in [CY, CY + ROAD_HALF] = [350, 420]
        cx, cy = G.CX, G.CY
        rh = G.ROAD_HALF
        stop_dist = G.STOP_DIST

        north_stop_y = cy - stop_dist     # 206.0
        south_stop_y = cy + stop_dist     # 494.0
        east_stop_x = cx + stop_dist      # 644.0
        west_stop_x = cx - stop_dist      # 356.0

        # -----------------------------------------------------------------
        # 1. PEDESTRIAN CROSSING ZONES (Zebra crossings)
        # -----------------------------------------------------------------
        # North crossing: x in [430, 570], y in [210, 250]
        z_n = G.ZEBRAS[0]  # (430, 214, 570, 246)
        self._zones[SpatialZone.NORTH_CROSSING] = ZoneDefinition(
            zone=SpatialZone.NORTH_CROSSING,
            approach=Approach.NORTH,
            zone_type=ZoneType.CROSSING,
            regions=[RectRegion(z_n[0], 210.0, z_n[2], 250.0,
                                inclusive_min_x=True, inclusive_max_x=True,
                                inclusive_min_y=True, inclusive_max_y=True)],
            description="North zebra crossing zone across the roadway"
        )

        # South crossing: x in [430, 570], y in [450, 490]
        z_s = G.ZEBRAS[2]  # (430, 454, 570, 486)
        self._zones[SpatialZone.SOUTH_CROSSING] = ZoneDefinition(
            zone=SpatialZone.SOUTH_CROSSING,
            approach=Approach.SOUTH,
            zone_type=ZoneType.CROSSING,
            regions=[RectRegion(z_s[0], 450.0, z_s[2], 490.0,
                                inclusive_min_x=True, inclusive_max_x=True,
                                inclusive_min_y=True, inclusive_max_y=True)],
            description="South zebra crossing zone across the roadway"
        )

        # East crossing: x in [600, 640], y in [280, 420]
        z_e = G.ZEBRAS[1]  # (604, 280, 636, 420)
        self._zones[SpatialZone.EAST_CROSSING] = ZoneDefinition(
            zone=SpatialZone.EAST_CROSSING,
            approach=Approach.EAST,
            zone_type=ZoneType.CROSSING,
            regions=[RectRegion(600.0, z_e[1], 640.0, z_e[3],
                                inclusive_min_x=True, inclusive_max_x=True,
                                inclusive_min_y=True, inclusive_max_y=True)],
            description="East zebra crossing zone across the roadway"
        )

        # West crossing: x in [360, 400], y in [280, 420]
        z_w = G.ZEBRAS[3]  # (364, 280, 396, 420)
        self._zones[SpatialZone.WEST_CROSSING] = ZoneDefinition(
            zone=SpatialZone.WEST_CROSSING,
            approach=Approach.WEST,
            zone_type=ZoneType.CROSSING,
            regions=[RectRegion(360.0, z_w[1], 400.0, z_w[3],
                                inclusive_min_x=True, inclusive_max_x=True,
                                inclusive_min_y=True, inclusive_max_y=True)],
            description="West zebra crossing zone across the roadway"
        )

        # -----------------------------------------------------------------
        # 2. PEDESTRIAN WAITING ZONES (Curb staging areas on either side)
        # -----------------------------------------------------------------
        # North waiting: curb ends at (416, 230) and (584, 230).
        # West curb: x in [390, 430), y in [205, 255]
        # East curb: x in (570, 610], y in [205, 255]
        self._zones[SpatialZone.NORTH_WAITING] = ZoneDefinition(
            zone=SpatialZone.NORTH_WAITING,
            approach=Approach.NORTH,
            zone_type=ZoneType.WAITING,
            regions=[
                RectRegion(390.0, 205.0, 430.0, 255.0, inclusive_max_x=False),
                RectRegion(570.0, 205.0, 610.0, 255.0, inclusive_min_x=False),
            ],
            description="North sidewalk curbs waiting to cross North crosswalk"
        )

        # South waiting: curb ends at (416, 470) and (584, 470).
        # West curb: x in [390, 430), y in [445, 495]
        # East curb: x in (570, 610], y in [445, 495]
        self._zones[SpatialZone.SOUTH_WAITING] = ZoneDefinition(
            zone=SpatialZone.SOUTH_WAITING,
            approach=Approach.SOUTH,
            zone_type=ZoneType.WAITING,
            regions=[
                RectRegion(390.0, 445.0, 430.0, 495.0, inclusive_max_x=False),
                RectRegion(570.0, 445.0, 610.0, 495.0, inclusive_min_x=False),
            ],
            description="South sidewalk curbs waiting to cross South crosswalk"
        )

        # East waiting: curb ends at (620, 266) and (620, 434).
        # North curb: x in [600, 640], y in [245, 280)
        # South curb: x in [600, 640], y in (420, 455]
        self._zones[SpatialZone.EAST_WAITING] = ZoneDefinition(
            zone=SpatialZone.EAST_WAITING,
            approach=Approach.EAST,
            zone_type=ZoneType.WAITING,
            regions=[
                RectRegion(600.0, 245.0, 640.0, 280.0, inclusive_max_y=False),
                RectRegion(600.0, 420.0, 640.0, 455.0, inclusive_min_y=False),
            ],
            description="East sidewalk curbs waiting to cross East crosswalk"
        )

        # West waiting: curb ends at (380, 266) and (380, 434).
        # North curb: x in [360, 400], y in [245, 280)
        # South curb: x in [360, 400], y in (420, 455]
        self._zones[SpatialZone.WEST_WAITING] = ZoneDefinition(
            zone=SpatialZone.WEST_WAITING,
            approach=Approach.WEST,
            zone_type=ZoneType.WAITING,
            regions=[
                RectRegion(360.0, 245.0, 400.0, 280.0, inclusive_max_y=False),
                RectRegion(360.0, 420.0, 400.0, 455.0, inclusive_min_y=False),
            ],
            description="West sidewalk curbs waiting to cross West crosswalk"
        )

        # -----------------------------------------------------------------
        # 3. VEHICLE QUEUE ZONES (Immediately upstream of stop lines)
        # -----------------------------------------------------------------
        # North queue: southbound incoming lane x in [430, 500], y in [206 - qd, 206]
        self._zones[SpatialZone.NORTH_QUEUE] = ZoneDefinition(
            zone=SpatialZone.NORTH_QUEUE,
            approach=Approach.NORTH,
            zone_type=ZoneType.QUEUE,
            regions=[RectRegion(430.0, north_stop_y - qd, 500.0, north_stop_y)],
            description="North southbound approach queue zone before stop line"
        )

        # South queue: northbound incoming lane x in [500, 570], y in [494, 494 + qd]
        self._zones[SpatialZone.SOUTH_QUEUE] = ZoneDefinition(
            zone=SpatialZone.SOUTH_QUEUE,
            approach=Approach.SOUTH,
            zone_type=ZoneType.QUEUE,
            regions=[RectRegion(500.0, south_stop_y, 570.0, south_stop_y + qd)],
            description="South northbound approach queue zone before stop line"
        )

        # East queue: westbound incoming lane y in [280, 350], x in [644, 644 + qd]
        self._zones[SpatialZone.EAST_QUEUE] = ZoneDefinition(
            zone=SpatialZone.EAST_QUEUE,
            approach=Approach.EAST,
            zone_type=ZoneType.QUEUE,
            regions=[RectRegion(east_stop_x, 280.0, east_stop_x + qd, 350.0)],
            description="East westbound approach queue zone before stop line"
        )

        # West queue: eastbound incoming lane y in [350, 420], x in [356 - qd, 356]
        self._zones[SpatialZone.WEST_QUEUE] = ZoneDefinition(
            zone=SpatialZone.WEST_QUEUE,
            approach=Approach.WEST,
            zone_type=ZoneType.QUEUE,
            regions=[RectRegion(west_stop_x - qd, 350.0, west_stop_x, 420.0)],
            description="West eastbound approach queue zone before stop line"
        )

        # -----------------------------------------------------------------
        # 4. VEHICLE APPROACH ZONES (Advance detection upstream of queue)
        # -----------------------------------------------------------------
        # North approach: y in [0, 206 - qd)
        self._zones[SpatialZone.NORTH_APPROACH] = ZoneDefinition(
            zone=SpatialZone.NORTH_APPROACH,
            approach=Approach.NORTH,
            zone_type=ZoneType.APPROACH,
            regions=[RectRegion(430.0, 0.0, 500.0, north_stop_y - qd, inclusive_max_y=False)],
            description="North southbound advance approach lane"
        )

        # South approach: y in (494 + qd, 700]
        self._zones[SpatialZone.SOUTH_APPROACH] = ZoneDefinition(
            zone=SpatialZone.SOUTH_APPROACH,
            approach=Approach.SOUTH,
            zone_type=ZoneType.APPROACH,
            regions=[RectRegion(500.0, south_stop_y + qd, 570.0, float(G.H), inclusive_min_y=False)],
            description="South northbound advance approach lane"
        )

        # East approach: x in (644 + qd, 1000]
        self._zones[SpatialZone.EAST_APPROACH] = ZoneDefinition(
            zone=SpatialZone.EAST_APPROACH,
            approach=Approach.EAST,
            zone_type=ZoneType.APPROACH,
            regions=[RectRegion(east_stop_x + qd, 280.0, float(G.W), 350.0, inclusive_min_x=False)],
            description="East westbound advance approach lane"
        )

        # West approach: x in [0, 356 - qd)
        self._zones[SpatialZone.WEST_APPROACH] = ZoneDefinition(
            zone=SpatialZone.WEST_APPROACH,
            approach=Approach.WEST,
            zone_type=ZoneType.APPROACH,
            regions=[RectRegion(0.0, 350.0, west_stop_x - qd, 420.0, inclusive_max_x=False)],
            description="West eastbound advance approach lane"
        )

    # ------------------------------------------------------------------ API
    def get_zone_definition(self, zone: SpatialZone) -> Optional[ZoneDefinition]:
        return self._zones.get(zone)

    def all_zones(self) -> Dict[SpatialZone, ZoneDefinition]:
        return dict(self._zones)

    def classify(
        self,
        point: Optional[Tuple[float, float]] = None,
        object_class: str = "",
        *,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> SpatialZone:
        """Deterministically classify a tracked object into a SpatialZone.

        Parameters:
            point: (x, y) coordinates of the object.
            object_class: string label (e.g. "person", "car", "bus", "truck").
            bbox: optional (x0, y0, x1, y1) bounding box used if point is not provided.
        """
        # Resolve target point
        if point is None:
            if bbox is not None:
                point = ((bbox[0] + bbox[2]) / 2.0, bbox[3])  # bottom-center ground contact
            else:
                return SpatialZone.NONE

        x, y = point
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))
                and math.isfinite(x) and math.isfinite(y)):
            return SpatialZone.NONE

        # Enforce strict object class rules
        if is_pedestrian(object_class):
            # Check crossing first (crossing takes precedence at curb boundary), then waiting
            for zone_key in (
                SpatialZone.NORTH_CROSSING, SpatialZone.SOUTH_CROSSING,
                SpatialZone.EAST_CROSSING, SpatialZone.WEST_CROSSING,
                SpatialZone.NORTH_WAITING, SpatialZone.SOUTH_WAITING,
                SpatialZone.EAST_WAITING, SpatialZone.WEST_WAITING,
            ):
                if self._zones[zone_key].contains(x, y):
                    return zone_key
            return SpatialZone.NONE

        if is_vehicle(object_class):
            # Check queue first (queue takes precedence at stop line/boundary), then approach
            for zone_key in (
                SpatialZone.NORTH_QUEUE, SpatialZone.SOUTH_QUEUE,
                SpatialZone.EAST_QUEUE, SpatialZone.WEST_QUEUE,
                SpatialZone.NORTH_APPROACH, SpatialZone.SOUTH_APPROACH,
                SpatialZone.EAST_APPROACH, SpatialZone.WEST_APPROACH,
            ):
                if self._zones[zone_key].contains(x, y):
                    return zone_key
            return SpatialZone.NONE

        # Unknown / non-relevant object class
        return SpatialZone.NONE

    def classify_object(self, obj: Any) -> SpatialZone:
        """Classifies a TrackedObject, dict, or tuple."""
        if isinstance(obj, TrackedObject):
            return self.classify((obj.x, obj.y), obj.object_class)
        if isinstance(obj, dict):
            pt = obj.get("point")
            if pt is None and "x" in obj and "y" in obj:
                pt = (obj["x"], obj["y"])
            cls = obj.get("object_class") or obj.get("class") or obj.get("label", "")
            bb = obj.get("bbox")
            return self.classify(pt, cls, bbox=bb)
        if isinstance(obj, (tuple, list)):
            if len(obj) == 2 and isinstance(obj[0], (tuple, list)):
                return self.classify(obj[0], obj[1])
            if len(obj) >= 3:
                return self.classify((obj[0], obj[1]), str(obj[2]))
        return SpatialZone.NONE

    def summarize(self, objects: Iterable[Any]) -> SpatialDemandSummary:
        """Produce a deterministic, structured demand summary across all approaches."""
        app_demands = {app: ApproachDemand(approach=app) for app in Approach}
        zone_counts: Dict[SpatialZone, int] = {z: 0 for z in SpatialZone}

        counts = {
            app: {"waiting": 0, "crossing": 0, "approach": 0, "queue": 0}
            for app in Approach
        }

        total_waiting = 0
        total_crossing = 0
        total_approach = 0
        total_queue = 0

        for obj in objects:
            zone = self.classify_object(obj)
            zone_counts[zone] += 1
            if zone == SpatialZone.NONE:
                continue

            app = zone.approach
            zt = zone.zone_type
            if app is None or zt is None:
                continue

            if zt == ZoneType.WAITING:
                counts[app]["waiting"] += 1
                total_waiting += 1
            elif zt == ZoneType.CROSSING:
                counts[app]["crossing"] += 1
                total_crossing += 1
            elif zt == ZoneType.APPROACH:
                counts[app]["approach"] += 1
                total_approach += 1
            elif zt == ZoneType.QUEUE:
                counts[app]["queue"] += 1
                total_queue += 1

        by_approach = {
            app: ApproachDemand(
                approach=app,
                waiting_pedestrians=counts[app]["waiting"],
                crossing_pedestrians=counts[app]["crossing"],
                approaching_vehicles=counts[app]["approach"],
                queued_vehicles=counts[app]["queue"]
            )
            for app in Approach
        }

        return SpatialDemandSummary(
            waiting_pedestrians=total_waiting,
            crossing_pedestrians=total_crossing,
            approaching_vehicles=total_approach,
            queued_vehicles=total_queue,
            by_approach=by_approach,
            zone_counts=zone_counts
        )
