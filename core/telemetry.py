"""Telemetry Layer for CrossGuard.

Aggregates individual SpatialDetection objects into structured traffic demand
telemetry and produces controller-compatible DemandSnapshot objects.
Pure Python; independent of the Adaptive FSM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from core.models import DemandSnapshot
from core.perception_adapter import SpatialDetection
from core.spatial_analysis import Approach, SpatialZone, ZoneType


@dataclass(frozen=True)
class ApproachDemandSnapshot:
    """Traffic demand for a single intersection approach arm."""
    approach: Approach
    waiting_pedestrians: int = 0
    crossing_pedestrians: int = 0
    approaching_vehicles: int = 0
    queued_vehicles: int = 0

    @property
    def total_pedestrians(self) -> int:
        return self.waiting_pedestrians + self.crossing_pedestrians

    @property
    def vehicle_demand(self) -> int:
        """Total vehicles on this approach (approaching + queued)."""
        return self.approaching_vehicles + self.queued_vehicles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approach": self.approach.value,
            "waiting_pedestrians": self.waiting_pedestrians,
            "crossing_pedestrians": self.crossing_pedestrians,
            "approaching_vehicles": self.approaching_vehicles,
            "queued_vehicles": self.queued_vehicles,
            "total_pedestrians": self.total_pedestrians,
            "vehicle_demand": self.vehicle_demand,
        }


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Rich telemetry snapshot with approach breakdown, zone counts, and controller DemandSnapshot."""
    demand: DemandSnapshot
    waiting_pedestrians: int
    crossing_pedestrians: int
    approaching_vehicles: int
    queued_vehicles: int
    by_approach: Dict[Approach, ApproachDemandSnapshot]
    zone_counts: Dict[SpatialZone, int]
    tracked_ids: Set[int]
    frame_index: int = 0

    @property
    def total_vehicle_demand(self) -> int:
        return self.approaching_vehicles + self.queued_vehicles

    def waiting_on(self, approach: Approach) -> int:
        return self.by_approach[approach].waiting_pedestrians

    def crossing_on(self, approach: Approach) -> int:
        return self.by_approach[approach].crossing_pedestrians

    def queued_on(self, approach: Approach) -> int:
        return self.by_approach[approach].queued_vehicles

    def approaching_on(self, approach: Approach) -> int:
        return self.by_approach[approach].approaching_vehicles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "waiting_pedestrians": self.waiting_pedestrians,
            "crossing_pedestrians": self.crossing_pedestrians,
            "approaching_vehicles": self.approaching_vehicles,
            "queued_vehicles": self.queued_vehicles,
            "vehicle_demand": self.total_vehicle_demand,
            "by_approach": {app.value: d.to_dict() for app, d in self.by_approach.items()},
            "zone_counts": {z.value: cnt for z, cnt in self.zone_counts.items() if cnt > 0},
            "active_tracks_count": len(self.tracked_ids),
        }


class TelemetryAggregator:
    """Aggregates SpatialDetection collections into DemandSnapshot and TelemetrySnapshot.

    Deduplicates detections by track_id to avoid artificial demand inflation.
    Preserves strict distinctions between waiting vs crossing pedestrians and
    approaching vs queued vehicles.
    """

    def __init__(self, source_name: str = "vision") -> None:
        self.source_name = source_name

    def aggregate(
        self,
        detections: Sequence[SpatialDetection],
        frame_index: int = 0,
        longest_pedestrian_wait_s: float = 0.0,
    ) -> TelemetrySnapshot:
        """Aggregate a sequence of SpatialDetection objects into a TelemetrySnapshot."""
        # Initialize approach counters
        approach_counts: Dict[Approach, Dict[str, int]] = {
            app: {"waiting": 0, "crossing": 0, "approach": 0, "queue": 0}
            for app in Approach
        }
        zone_counts: Dict[SpatialZone, int] = {z: 0 for z in SpatialZone}

        total_waiting = 0
        total_crossing = 0
        total_approach = 0
        total_queue = 0

        seen_track_ids: Set[int] = set()

        for det in detections:
            # Track-aware deduplication within the snapshot
            if det.track_id in seen_track_ids:
                continue
            seen_track_ids.add(det.track_id)

            zone = det.zone
            zone_counts[zone] = zone_counts.get(zone, 0) + 1

            # Ignore detections outside relevant zones
            if zone == SpatialZone.NONE or det.approach is None or det.zone_type is None:
                continue

            app = det.approach
            zt = det.zone_type

            if zt == ZoneType.WAITING:
                approach_counts[app]["waiting"] += 1
                total_waiting += 1
            elif zt == ZoneType.CROSSING:
                approach_counts[app]["crossing"] += 1
                total_crossing += 1
            elif zt == ZoneType.APPROACH:
                approach_counts[app]["approach"] += 1
                total_approach += 1
            elif zt == ZoneType.QUEUE:
                approach_counts[app]["queue"] += 1
                total_queue += 1

        # Construct approach snapshots
        by_approach: Dict[Approach, ApproachDemandSnapshot] = {
            app: ApproachDemandSnapshot(
                approach=app,
                waiting_pedestrians=approach_counts[app]["waiting"],
                crossing_pedestrians=approach_counts[app]["crossing"],
                approaching_vehicles=approach_counts[app]["approach"],
                queued_vehicles=approach_counts[app]["queue"],
            )
            for app in Approach
        }

        # Build standard controller-compatible DemandSnapshot
        # vehicle_demand is total vehicles approaching or queued
        total_vehicle_demand = total_approach + total_queue
        demand = DemandSnapshot(
            pedestrians_waiting=total_waiting,
            longest_pedestrian_wait_s=max(0.0, float(longest_pedestrian_wait_s)),
            pedestrians_in_crosswalk=total_crossing,
            vehicle_demand=total_vehicle_demand,
            vehicle_queue=total_queue,
            source=self.source_name,
        )

        return TelemetrySnapshot(
            demand=demand,
            waiting_pedestrians=total_waiting,
            crossing_pedestrians=total_crossing,
            approaching_vehicles=total_approach,
            queued_vehicles=total_queue,
            by_approach=by_approach,
            zone_counts=zone_counts,
            tracked_ids=seen_track_ids,
            frame_index=frame_index,
        )

    def to_demand_snapshot(
        self,
        detections: Sequence[SpatialDetection],
        frame_index: int = 0,
        longest_pedestrian_wait_s: float = 0.0,
    ) -> DemandSnapshot:
        """Produce directly the core.models.DemandSnapshot for the adaptive controller."""
        return self.aggregate(
            detections,
            frame_index=frame_index,
            longest_pedestrian_wait_s=longest_pedestrian_wait_s,
        ).demand

    def __call__(
        self,
        detections: Sequence[SpatialDetection],
        frame_index: int = 0,
        longest_pedestrian_wait_s: float = 0.0,
    ) -> TelemetrySnapshot:
        """Convenience callable interface returning TelemetrySnapshot."""
        return self.aggregate(
            detections,
            frame_index=frame_index,
            longest_pedestrian_wait_s=longest_pedestrian_wait_s,
        )
