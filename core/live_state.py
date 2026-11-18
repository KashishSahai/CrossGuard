"""Single, UI-independent view of the demand in one PipelineFrameResult.

Every live surface (Live Vision KPI card, Semantic Demand table, Bloom City header and the Bloom City
frame record) reads demand through this object, so none of them can reconstruct or drift from the
authoritative per-frame values. Pure Python; no Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from core.spatial_analysis import Approach


@dataclass(frozen=True)
class ApproachRow:
    approach: str
    waiting: int
    crossing: int
    approaching: int
    queued: int


@dataclass(frozen=True)
class LiveDemandView:
    frame_index: int
    timestamp: float
    waiting: int
    crossing: int
    approaching: int
    queued: int
    longest_wait_s: float
    rows: Tuple[ApproachRow, ...]

    @property
    def table_totals(self) -> Dict[str, int]:
        return {
            "waiting": sum(r.waiting for r in self.rows),
            "crossing": sum(r.crossing for r in self.rows),
            "approaching": sum(r.approaching for r in self.rows),
            "queued": sum(r.queued for r in self.rows),
        }

    def is_consistent(self) -> bool:
        """KPI values must equal the column sums of the approach table."""
        t = self.table_totals
        return (t["waiting"], t["crossing"], t["approaching"], t["queued"]) == (
            self.waiting, self.crossing, self.approaching, self.queued)


def live_demand_view(res: Any) -> LiveDemandView:
    """Build the view from the authoritative PipelineFrameResult of ONE frame."""
    demand, snap = res.demand_snapshot, res.telemetry_snapshot
    rows = []
    for app in Approach:
        ad = snap.by_approach.get(app)
        rows.append(ApproachRow(
            approach=app.value,
            waiting=ad.waiting_pedestrians if ad else 0,
            crossing=ad.crossing_pedestrians if ad else 0,
            approaching=ad.approaching_vehicles if ad else 0,
            queued=ad.queued_vehicles if ad else 0))
    return LiveDemandView(
        frame_index=res.frame_index,
        timestamp=res.timestamp,
        waiting=int(demand.pedestrians_waiting),
        crossing=int(demand.pedestrians_in_crosswalk),
        approaching=int(res.approaching_vehicles),
        queued=int(demand.vehicle_queue),
        longest_wait_s=float(demand.longest_pedestrian_wait_s),
        rows=tuple(rows))
