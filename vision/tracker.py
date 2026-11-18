"""Deterministic ByteTrack multi-object tracking implementation.

Provides persistent track IDs across frames using two-stage IoU association:
  Stage 1: High-confidence detections matched against existing active tracks.
  Stage 2: Low-confidence detections matched against remaining unmatched tracks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .models import RawDetection, TrackedDetection


def bbox_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Calculate Intersection-over-Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    start_frame: int
    last_frame: int
    lost_frames: int = 0
    hit_streak: int = 1

    def update(self, det: RawDetection, frame_index: int) -> None:
        self.bbox = det.bbox
        self.confidence = det.confidence
        self.last_frame = frame_index
        self.lost_frames = 0
        self.hit_streak += 1

    def mark_missed(self) -> None:
        self.lost_frames += 1
        self.hit_streak = 0

    def to_tracked_detection(self, frame_index: int) -> TrackedDetection:
        x1, y1, x2, y2 = self.bbox
        return TrackedDetection(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            center_x=(x1 + x2) / 2.0,
            center_y=(y1 + y2) / 2.0,
            frame_index=frame_index,
        )


def _match_iou(
    tracks: Sequence[Track],
    detections: Sequence[RawDetection],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy IoU association between tracks and detections."""
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    pairs: List[Tuple[float, int, int]] = []
    for t_idx, t in enumerate(tracks):
        for d_idx, d in enumerate(detections):
            # Same class matching
            if t.class_name == d.class_name:
                iou = bbox_iou(t.bbox, d.bbox)
                if iou >= iou_threshold:
                    pairs.append((iou, t_idx, d_idx))

    # Sort descending by IoU
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_tracks = set()
    matched_dets = set()
    matches: List[Tuple[int, int]] = []

    for _, t_idx, d_idx in pairs:
        if t_idx not in matched_tracks and d_idx not in matched_dets:
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            matches.append((t_idx, d_idx))

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets


class ByteTracker:
    """ByteTrack multi-object tracker."""

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_iou_threshold: float = 0.5,
        max_lost_frames: int = 30,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_iou_threshold = match_iou_threshold
        self.max_lost_frames = max_lost_frames

        self._next_id: int = 1
        self._tracks: List[Track] = []

    def reset(self) -> None:
        """Reset all tracking state and counter."""
        self._next_id = 1
        self._tracks.clear()

    @property
    def active_tracks_count(self) -> int:
        return len(self._tracks)

    def update(self, detections: List[RawDetection], frame_index: int) -> List[TrackedDetection]:
        """Update tracker with frame detections and return active tracked objects."""
        # 1. Partition detections into high and low confidence
        high_dets: List[RawDetection] = []
        low_dets: List[RawDetection] = []

        for d in detections:
            if d.confidence >= self.high_threshold:
                high_dets.append(d)
            elif d.confidence >= self.low_threshold:
                low_dets.append(d)

        # 2. Stage 1: Associate high-confidence detections with existing tracks
        matches_1, unmatched_tracks_1, unmatched_high_dets = _match_iou(
            self._tracks, high_dets, self.match_iou_threshold
        )

        matched_track_indices = set()
        for t_idx, d_idx in matches_1:
            self._tracks[t_idx].update(high_dets[d_idx], frame_index)
            matched_track_indices.add(t_idx)

        # 3. Stage 2: Associate remaining tracks with low-confidence detections
        remaining_tracks = [self._tracks[i] for i in unmatched_tracks_1]
        matches_2, unmatched_tracks_2_rel, _ = _match_iou(
            remaining_tracks, low_dets, self.match_iou_threshold * 0.8
        )

        for rel_t_idx, d_idx in matches_2:
            real_t_idx = unmatched_tracks_1[rel_t_idx]
            self._tracks[real_t_idx].update(low_dets[d_idx], frame_index)
            matched_track_indices.add(real_t_idx)

        # 4. Handle unmatched tracks (increment lost frames)
        for i, track in enumerate(self._tracks):
            if i not in matched_track_indices:
                track.mark_missed()

        # 5. Initiate new tracks for unmatched high-confidence detections
        new_track_indices = set()
        for d_idx in unmatched_high_dets:
            det = high_dets[d_idx]
            new_track = Track(
                track_id=self._next_id,
                class_id=det.class_id,
                class_name=det.class_name,
                bbox=det.bbox,
                confidence=det.confidence,
                start_frame=frame_index,
                last_frame=frame_index,
            )
            self._next_id += 1
            self._tracks.append(new_track)
            new_track_indices.add(len(self._tracks) - 1)

        # 6. Prune lost tracks
        self._tracks = [t for t in self._tracks if t.lost_frames <= self.max_lost_frames]

        # 7. Collect output: only tracks that were detected/matched in the current frame
        active_in_frame: List[TrackedDetection] = []
        for i, t in enumerate(self._tracks):
            if i in matched_track_indices or i in new_track_indices:
                active_in_frame.append(t.to_tracked_detection(frame_index))

        return active_in_frame
