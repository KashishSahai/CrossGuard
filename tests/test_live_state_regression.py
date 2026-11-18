"""Regression tests for the Bloom City live-state bugs (frozen "0.0s left", blank reason, stale counters).

Root causes covered:
  1. the live renderer extrapolated `time_remaining` from the wall clock;
  2. frames indexed the adapter's reason table but the app passed a 1-entry list (blank reason);
  3. the Bloom City panel labelled vehicle_demand (approaching + queued) as "approaching".
Everything here runs frames through the REAL pipeline objects with a deterministic mock detector.
"""
from __future__ import annotations

import csv
import os
import re

import pytest

from core.adaptive_controller import AdaptiveController
from core.live_state import live_demand_view
from core.models import (PHASE_ORDER, PHASE_OUTPUTS, DemandSnapshot, PedestrianSignal, Phase,
                         VehicleSignal)
from core.pipeline import AdaptivePipeline, PipelineFrameResult
from core.telemetry import TelemetryAggregator
from simulation.controller_bridge import map_controller_state
from simulation.live_adapter import BloomCityLiveAdapter
from tests.live_asserts import assert_html_latest_frame_is, payload_of
from tests.test_adaptive_pipeline import MockBackend
from vision.models import RawDetection, VisionPipelineConfig
from vision.vision_pipeline import VisionPipeline

APP_SRC = open("app.py", encoding="utf-8").read()


# ------------------------------------------------------------------ helpers
def _det(cls: str, point, w=12.0, h=40.0, conf=0.9) -> RawDetection:
    """RawDetection whose ground-contact (bottom-centre) point is `point`."""
    cid = {"person": 0, "car": 2}[cls]
    x, y = point
    return RawDetection(x1=x - w / 2, y1=y - h, x2=x + w / 2, y2=y, confidence=conf,
                        class_id=cid, class_name=cls)


# Interior points of each zone in canonical intersection space (identity camera in tests).
USER_FRAME = [
    _det("person", (620, 276)), _det("person", (610, 272)),      # EAST_WAITING x2
    _det("person", (620, 348)),                                   # EAST_CROSSING
    _det("person", (380, 348)),                                   # WEST_CROSSING
    _det("car", (692, 312), w=40, h=20),                          # EAST_QUEUE
    _det("car", (532, 544), w=40, h=20), _det("car", (532, 520), w=40, h=20),   # SOUTH_QUEUE x2
]


def _pipeline():
    backend = MockBackend(device="cpu")
    vp = VisionPipeline(config=VisionPipelineConfig(device="cpu"), backend=backend)
    return AdaptivePipeline(vision_pipeline=vp, default_fps=10.0), backend


def _process(pipe, backend, dets, idx, dt):
    backend.set_detections(dets)
    return pipe.process_frame("dummy_frame", frame_index=idx, dt=dt)


# ------------------------------------------------------------------ single-frame consistency
def test_one_frame_demand_is_identical_on_every_surface():
    pipe, backend = _pipeline()
    res = _process(pipe, backend, USER_FRAME, 50, 0.1)

    d = res.demand_snapshot
    assert (d.pedestrians_waiting, d.pedestrians_in_crosswalk, d.vehicle_queue) == (2, 2, 3)
    assert res.approaching_vehicles == 0

    view = live_demand_view(res)                       # Live Vision KPI card + table + Bloom header
    assert (view.waiting, view.crossing, view.approaching, view.queued) == (2, 2, 0, 3)
    assert view.longest_wait_s == d.longest_pedestrian_wait_s
    assert view.is_consistent(), view.table_totals
    rows = {r.approach: r for r in view.rows}
    assert (rows["EAST"].waiting, rows["EAST"].crossing, rows["EAST"].queued) == (2, 1, 1)
    assert rows["SOUTH"].queued == 2 and rows["WEST"].crossing == 1 and rows["NORTH"].queued == 0

    adapter = BloomCityLiveAdapter()
    adapter.process_frame_result(res, video_name="clip.mp4")
    f = adapter.recorded_frames[-1]                    # what Bloom City will draw
    assert (f["np"], f["nc"], f["na"], f["nq"]) == (view.waiting, view.crossing, view.approaching, view.queued)
    assert f["lw"] == round(view.longest_wait_s, 1)
    assert f["frame_index"] == res.frame_index == 50
    assert f["vt"] == pytest.approx(res.timestamp)
    # the old bug: "vehicles approaching" must not show approaching+queued
    assert f["na"] != d.vehicle_demand or d.vehicle_demand == res.approaching_vehicles


def test_kpi_table_and_bloom_header_do_not_rebuild_demand_themselves():
    assert APP_SRC.count("live_demand_view(active_res)") >= 2      # Live Vision (KPI + table share it) and Bloom header
    assert "dv.rows" in APP_SRC
    assert "active_res.telemetry_snapshot" not in APP_SRC
    assert "active_res.approaching_vehicles" not in APP_SRC
    assert "active_res.queued_vehicles" not in APP_SRC


# ------------------------------------------------------------------ real FSM -> Bloom City, frame by frame
def test_real_fsm_progression_reaches_bloom_city_unchanged_every_frame():
    pipe, backend = _pipeline()
    adapter = BloomCityLiveAdapter()
    waiting_only = [_det("person", (620, 276)), _det("person", (610, 272))]   # 2 waiting, no vehicles
    seen, prev_total = [], -1.0
    for i in range(0, 400):
        res = _process(pipe, backend, waiting_only, i, 0.5)
        adapter.process_frame_result(res)
        c, sig, f = res.controller_state, res.signal_state, adapter.recorded_frames[-1]

        # one authoritative state for this frame, everywhere
        assert sig.phase == c.current_phase and res.current_phase == c.current_phase
        assert res.telemetry_snapshot.frame_index == res.frame_index == f["frame_index"] == i
        assert PHASE_ORDER[f["ph"]] == c.current_phase
        assert (("GREEN", "YELLOW", "RED")[f["vs"]], ("DONT_WALK", "WALK", "CLEARING")[f["ps"]]) == (
            PHASE_OUTPUTS[c.current_phase][0].value, PHASE_OUTPUTS[c.current_phase][1].value)
        assert adapter.reason_table[f["rs"]] == c.decision_reason != ""
        assert f["rm"] == round(c.time_remaining, 1)
        assert not (c.vehicle_signal == VehicleSignal.GREEN and c.pedestrian_signal == PedestrianSignal.WALK)
        assert c.total_time > prev_total                        # controller clock accumulates, never resets
        prev_total = c.total_time
        if not seen or seen[-1] != c.current_phase:
            seen.append(c.current_phase)
        if c.current_phase == Phase.PEDESTRIAN_WALK:
            break
    assert seen == [Phase.VEHICLE_GREEN, Phase.VEHICLE_YELLOW, Phase.ALL_RED_CLEARANCE, Phase.PEDESTRIAN_WALK]
    assert pipe.current_time == pytest.approx((i + 1) * 0.5)


REAL_LOG = "tools/step21_frame_log.csv"


@pytest.mark.skipif(not os.path.exists(REAL_LOG), reason="recorded 50 s telemetry log not present") \
    if hasattr(pytest, "mark") and hasattr(pytest.mark, "skipif") else (lambda f: f)
def test_recorded_real_video_demand_reaches_pedestrian_walk_and_bloom_city_follows():
    """Replays the real per-frame demand of tools/intersection_50s.mp4 through the real FSM + adapter.

    The timestamps are an expectation of this validation (about 43 s / 46.8 s / 49.8 s), not controller
    rules: nothing here forces or schedules a phase.
    """
    if not os.path.exists(REAL_LOG):
        pytest.skip("recorded telemetry log not present")
    rows = list(csv.DictReader(open(REAL_LOG, encoding="utf-8")))
    ctrl, adapter, agg = AdaptiveController(), BloomCityLiveAdapter(), TelemetryAggregator()
    dt, trans, last_res = 1 / 23.976, {}, None
    for r in rows:
        d = DemandSnapshot(pedestrians_waiting=int(r["ped_waiting"]),
                           longest_pedestrian_wait_s=float(r["longest_ped_wait"]),
                           pedestrians_in_crosswalk=int(r["ped_crossing"]),
                           vehicle_demand=int(r["veh_demand"]), vehicle_queue=int(r["veh_queue"]),
                           source="replay")
        st = ctrl.step(dt, d)
        res = PipelineFrameResult(frame_index=int(r["frame_index"]), timestamp=st.total_time, dt=dt,
                                  tracked_detections=[], spatial_detections=[],
                                  telemetry_snapshot=agg.aggregate([], frame_index=int(r["frame_index"])),
                                  demand_snapshot=d, controller_state=st, signal_state=map_controller_state(st))
        adapter.process_frame_result(res, force_record=True)
        trans.setdefault(st.current_phase, st.total_time)
        last_res = res
    assert [p for p in trans] == [Phase.VEHICLE_GREEN, Phase.VEHICLE_YELLOW,
                                  Phase.ALL_RED_CLEARANCE, Phase.PEDESTRIAN_WALK]
    assert 42.5 <= trans[Phase.VEHICLE_YELLOW] <= 44.0
    assert 46.0 <= trans[Phase.ALL_RED_CLEARANCE] <= 47.5
    assert 49.0 <= trans[Phase.PEDESTRIAN_WALK] <= 50.5
    assert last_res.current_phase == Phase.PEDESTRIAN_WALK
    assert last_res.vehicle_signal == VehicleSignal.RED and last_res.pedestrian_signal == PedestrianSignal.WALK

    from app import sync_bloom_city_with_signal
    _, _, html = sync_bloom_city_with_signal(last_res.signal_state, demand=last_res.demand_snapshot,
                                             live_adapter=adapter)
    assert_html_latest_frame_is(html, Phase.PEDESTRIAN_WALK)
    assert payload_of(html)["reasons"][payload_of(html)["frames"][-1]["rs"]] == last_res.controller_state.decision_reason


# ------------------------------------------------------------------ live mode never uses scenario data
def _live_adapter_with_frames(n=12):
    pipe, backend = _pipeline()
    adapter = BloomCityLiveAdapter()
    res = None
    for i in range(n):
        res = _process(pipe, backend, USER_FRAME, i, 0.1)
        adapter.process_frame_result(res, video_name="clip.mp4")
    return adapter, res


def test_live_html_ignores_scenario_arguments_and_resolves_every_reason():
    from app import sync_bloom_city_with_signal
    adapter, res = _live_adapter_with_frames()
    out = [sync_bloom_city_with_signal(res.signal_state, scenario=sc, seed=sd, demand=res.demand_snapshot,
                                       live_adapter=adapter) for sc, sd in (("quiet", 1), ("heavy_traffic", 99))]
    assert out[0][0] == out[1][0] and out[0][2] == out[1][2]
    data = payload_of(out[0][2])
    assert data["mode"] == "LIVE CONTROLLER"
    assert all(f["frame_index"] >= 0 for f in data["frames"])              # scenario frames carry no frame_index
    assert all(0 <= f["rs"] < len(data["reasons"]) for f in data["frames"])  # no blank / undefined reason
    assert data["reasons"][data["frames"][-1]["rs"]] == res.controller_state.decision_reason
    assert "hold_phase" not in out[0][2] and "Balanced" not in out[0][2]


def test_stale_adapter_cannot_override_authoritative_signal_state():
    from app import sync_bloom_city_with_signal
    adapter, _ = _live_adapter_with_frames()                               # adapter is in VEHICLE_GREEN
    ctrl = AdaptiveController()
    ctrl.force_phase(Phase.PEDESTRIAN_WALK, hold=True)
    authoritative = map_controller_state(ctrl.state)
    _, _, html = sync_bloom_city_with_signal(authoritative, live_adapter=adapter)
    assert_html_latest_frame_is(html, Phase.PEDESTRIAN_WALK)               # signal state wins


def test_live_renderer_has_no_wall_clock_extrapolation():
    adapter, _ = _live_adapter_with_frames()
    html = adapter.render_html()
    for forbidden in ("live_tail", "liveRm", "tau>T", "52.0*dt"):
        assert forbidden not in html


def test_render_frames_are_zero_based_and_single_frame_is_not_animated():
    adapter = BloomCityLiveAdapter()
    pipe, backend = _pipeline()
    res = _process(pipe, backend, USER_FRAME, 0, 0.1)
    adapter.process_frame_result(res)
    f0, f1 = adapter.get_render_frames()
    assert f0["v"] == f1["v"] and f0["p"] == f1["p"] and f0["t"] == 0 and f1["t"] > 0   # nothing invented

    adapter2, _ = _live_adapter_with_frames(520)                           # exceeds the 500-frame buffer
    frames = adapter2.get_render_frames()
    assert len(frames) == 500 and frames[0]["t"] == 0
    assert all(b["t"] > a["t"] for a, b in zip(frames, frames[1:]))


# ------------------------------------------------------------------ UI structure
def _live_branch_source() -> str:
    a = APP_SRC.index('    if mode_choice == "Live Controller Mode":')
    b = APP_SRC.index("        # DEMO MODE")
    return APP_SRC[a:b]


def test_scenario_controls_exist_only_in_scenario_demo():
    live, demo = _live_branch_source(), APP_SRC[APP_SRC.index("        # DEMO MODE"):]
    for forbidden in ("st.selectbox", "SCENARIOS", "hold_phase", "simulate_bloom_city", "Traffic Scenario",
                      '"Controller Mode"', "Time of Day"):
        assert forbidden not in live, forbidden
    for needed in ("Traffic Scenario", "Time of Day", '"Controller Mode"'):
        assert needed in demo, needed


def test_live_branch_shows_the_live_controller_fields():
    live = _live_branch_source()
    for label in ("Synchronized to Perception", "Live Phase", "Vehicle Signal", "Pedestrian Signal",
                  "Decision &amp; Reason", "Waiting Pedestrians", "In Crosswalk", "Approaching Vehicles",
                  "Queued Vehicles", "Longest Pedestrian Wait"):
        assert label in live, label


def test_live_adapter_is_created_in_one_place_and_both_session_keys_stay_together():
    body = APP_SRC[APP_SRC.index("def _new_live_adapter"):APP_SRC.index("def sync_bloom_city_with_signal")]
    assert 'st.session_state["bloom_city_live_adapter"] = adapter' in body
    assert 'st.session_state["live_bloom_city_engine"] = adapter' in body
    assert APP_SRC.count("BloomCityLiveAdapter()") == 1                    # only inside the helper
    assert "AdaptiveController(" not in APP_SRC                            # no second FSM in the UI
