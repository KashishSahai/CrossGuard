"""Bloom City geometry (pixels in a 1000x700 world). Shared by engine and renderer."""
from __future__ import annotations

W, H = 1000, 700
CX, CY = 500, 350
ROAD_HALF = 70            # half road width -> box is 140x140
SIDEWALK = 28
LANE_OFFSET = 35
STOP_DIST = 144           # stop line distance from centre
BOX_IN = 74               # stop line -> box entry
BOX_LEN = 140

# Approach index: 0 = North (southbound), 1 = East (westbound), 2 = South (northbound), 3 = West (eastbound)
APPROACH_NAMES = ("N", "E", "S", "W")
SPAWN_MARGIN = 60


def axis_of(a: int) -> int:
    return a % 2          # 0 = north-south road, 1 = east-west road


def opposite(a: int) -> int:
    return (a + 2) % 4


# start point, unit direction, heading degrees
_START = {
    0: ((CX - LANE_OFFSET, -SPAWN_MARGIN), (0, 1), 90),
    1: ((W + SPAWN_MARGIN, CY - LANE_OFFSET), (-1, 0), 180),
    2: ((CX + LANE_OFFSET, H + SPAWN_MARGIN), (0, -1), -90),
    3: ((-SPAWN_MARGIN, CY + LANE_OFFSET), (1, 0), 0),
}
HEADING = {a: _START[a][2] for a in _START}


def lane_point(a: int, s: float):
    (x0, y0), (dx, dy), _ = _START[a]
    return x0 + dx * s, y0 + dy * s


def _stop_line_s(a: int) -> float:
    (x0, y0), (dx, dy), _ = _START[a]
    target = (CX, CY)
    if dx:
        return (CX + (-dx) * STOP_DIST - x0) / dx
    return (CY + (-dy) * STOP_DIST - y0) / dy


STOP_LINE_S = {a: _stop_line_s(a) for a in range(4)}      # front-bumper limit along the path
BOX_OUT_OFFSET = BOX_IN + BOX_LEN
PATH_LEN = {0: H + 2 * SPAWN_MARGIN, 1: W + 2 * SPAWN_MARGIN,
            2: H + 2 * SPAWN_MARGIN, 3: W + 2 * SPAWN_MARGIN}

# Zebra rectangles (x0, y0, x1, y1); index == approach arm
ZEBRAS = {
    0: (CX - ROAD_HALF, CY - 136, CX + ROAD_HALF, CY - 104),
    1: (CX + 104, CY - ROAD_HALF, CX + 136, CY + ROAD_HALF),
    2: (CX - ROAD_HALF, CY + 104, CX + ROAD_HALF, CY + 136),
    3: (CX - 136, CY - ROAD_HALF, CX - 104, CY + ROAD_HALF),
}
# Pedestrian walking lines per crosswalk: (end A, end B)
_C = ROAD_HALF + SIDEWALK // 2          # 84: curb-side waiting line offset
CROSSWALK_ENDS = {
    0: ((CX - _C, CY - 120), (CX + _C, CY - 120)),
    1: ((CX + 120, CY - _C), (CX + 120, CY + _C)),
    2: ((CX - _C, CY + 120), (CX + _C, CY + 120)),
    3: ((CX - 120, CY - _C), (CX - 120, CY + _C)),
}
CROSSING_LENGTH = 2 * _C
SLOTS_NS = [(-6, -12), (6, -12), (-6, 0), (6, 0), (-6, 12), (6, 12)]   # crosswalks 0 and 2
SLOTS_EW = [(-12, -6), (-12, 6), (0, -6), (0, 6), (12, -6), (12, 6)]    # crosswalks 1 and 3
MAX_SLOTS = 6


def slot_offset(cw: int, slot: int):
    return (SLOTS_NS if cw % 2 == 0 else SLOTS_EW)[slot]


def rects_overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]
