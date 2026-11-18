"""Real-browser check of LIVE Bloom City: physical SVG signal heads + panel for every FSM phase.

Skipped if Playwright/Chromium are unavailable.  Setup: pip install playwright && python -m playwright install chromium
"""
import csv
import os
import tempfile

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

from core.adaptive_controller import AdaptiveController
from core.live_state import live_demand_view
from core.models import DemandSnapshot, PedestrianSignal, Phase, VehicleSignal
from core.pipeline import PipelineFrameResult
from core.telemetry import TelemetryAggregator
from simulation.controller_bridge import map_controller_state
from simulation.live_adapter import BloomCityLiveAdapter
from tests.test_live_state_regression import USER_FRAME, _pipeline, _process

LENS_ON = {"r": "#ff5d6c", "y": "#ffd23f", "g": "#38d889"}
VEH_LIT = {Phase.VEHICLE_GREEN: "g", Phase.VEHICLE_YELLOW: "y", Phase.ALL_RED_CLEARANCE: "r",
           Phase.PEDESTRIAN_WALK: "r", Phase.PEDESTRIAN_CLEARANCE: "r"}
PED_FILL = {PedestrianSignal.DONT_WALK: "#ff5d6c", PedestrianSignal.WALK: "#38d889", PedestrianSignal.CLEARING: "#ffb020"}
PED_GLYPH = {PedestrianSignal.DONT_WALK: "\u270b", PedestrianSignal.WALK: "\U0001F6B6", PedestrianSignal.CLEARING: "\u23f3"}
V_TEXT = {"g": "GREEN", "y": "YELLOW", "r": "RED"}
P_TEXT = {PedestrianSignal.DONT_WALK: "DON'T WALK", PedestrianSignal.WALK: "WALK", PedestrianSignal.CLEARING: "CLEARING"}
PH_CHIP = [Phase.VEHICLE_GREEN, Phase.VEHICLE_YELLOW, Phase.ALL_RED_CLEARANCE,
           Phase.PEDESTRIAN_WALK, Phase.PEDESTRIAN_CLEARANCE]


def _open(p, html):
    path = os.path.join(tempfile.mkdtemp(), "live.html")
    open(path, "w", encoding="utf-8").write(html)
    page = p.new_page(viewport={"width": 1250, "height": 760})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("file://" + path)
    page.wait_for_timeout(400)
    return page, errors


def _assert_dom_shows(page, res, view=None):
    ph, v, p = res.current_phase, res.vehicle_signal, res.pedestrian_signal
    lit = VEH_LIT[ph]
    for head in range(4):                                             # physical vehicle heads
        for k in "ryg":
            fill = page.get_attribute(f"#vl{head}{k}", "fill")
            assert (fill == LENS_ON[k]) == (k == lit), (ph, head, k, fill)
        assert page.get_attribute(f"#pl{head}", "fill") == PED_FILL[p]   # physical pedestrian heads
        assert page.text_content(f"#pt{head}") == PED_GLYPH[p]
    assert page.inner_text("#vsb") == V_TEXT[lit]
    assert page.inner_text("#psb") == P_TEXT[p]
    assert page.locator(".chip.on").count() == 1
    assert page.eval_on_selector(".chip.on", "e => e.textContent") == \
        ["Vehicle Green", "Yellow", "All Red", "Pedestrian Walk", "Clearance"][PH_CHIP.index(ph)]
    assert page.inner_text("#why") == res.controller_state.decision_reason != ""
    assert not (v == VehicleSignal.GREEN and p == PedestrianSignal.WALK)
    if view is not None:
        assert [page.inner_text(i) for i in ("#np", "#nc", "#na", "#nq")] == \
            [str(view.waiting), str(view.crossing), str(view.approaching), str(view.queued)]


def _phase_result(phase):
    pipe, backend = _pipeline()
    pipe.controller.force_phase(phase, hold=True)                 # the REAL controller, frozen in `phase`
    adapter, res = BloomCityLiveAdapter(), None
    for i in range(8):
        res = _process(pipe, backend, USER_FRAME, i, 0.1)
        adapter.process_frame_result(res, video_name="clip.mp4")
    return adapter, res


def test_every_phase_renders_correct_physical_signals_and_panel_and_does_not_drift_with_wall_clock():
    from app import sync_bloom_city_with_signal
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"Chromium unavailable: {exc}")
        for phase in Phase:
            adapter, res = _phase_result(phase)
            _, _, html = sync_bloom_city_with_signal(res.signal_state, demand=res.demand_snapshot, live_adapter=adapter)
            page, errors = _open(browser, html)
            view = live_demand_view(res)
            _assert_dom_shows(page, res, view)
            before = (page.inner_text("#tm"), page.inner_text("#why"))
            page.wait_for_timeout(2500)                                # wall clock must not change anything
            assert (page.inner_text("#tm"), page.inner_text("#why")) == before, phase
            _assert_dom_shows(page, res, view)
            assert not errors, errors
            page.close()
        browser.close()


def test_replaying_history_then_returning_to_latest_frame_keeps_signals_authoritative():
    from app import sync_bloom_city_with_signal
    pipe, backend = _pipeline()
    adapter, res = BloomCityLiveAdapter(), None
    for i in range(40):                                              # natural progression, real FSM
        res = _process(pipe, backend, [USER_FRAME[0], USER_FRAME[1]], i, 0.5)
        adapter.process_frame_result(res)
    assert res.current_phase != Phase.VEHICLE_GREEN
    _, _, html = sync_bloom_city_with_signal(res.signal_state, demand=res.demand_snapshot, live_adapter=adapter)
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"Chromium unavailable: {exc}")
        page, errors = _open(browser, html)
        _assert_dom_shows(page, res)                                  # opens on the latest frame
        page.evaluate("window.__bloom.draw(0)")                       # jump to the oldest frame
        assert page.get_attribute("#vl0g", "fill") == LENS_ON["g"]    # history: green
        page.evaluate("window.__bloom.draw(window.__bloom.duration)")
        _assert_dom_shows(page, res)                                  # end of history == latest authoritative state
        assert not errors, errors
        browser.close()


REAL_LOG = "tools/step21_frame_log.csv"


def test_real_video_demand_reaches_pedestrian_walk_in_the_rendered_scene():
    if not os.path.exists(REAL_LOG):
        pytest.skip("recorded telemetry log not present")
    from app import sync_bloom_city_with_signal
    rows = list(csv.DictReader(open(REAL_LOG, encoding="utf-8")))
    ctrl, adapter, agg, dt, res = AdaptiveController(), BloomCityLiveAdapter(), TelemetryAggregator(), 1 / 23.976, None
    for r in rows:
        d = DemandSnapshot(pedestrians_waiting=int(r["ped_waiting"]), longest_pedestrian_wait_s=float(r["longest_ped_wait"]),
                           pedestrians_in_crosswalk=int(r["ped_crossing"]), vehicle_demand=int(r["veh_demand"]),
                           vehicle_queue=int(r["veh_queue"]), source="replay")
        st = ctrl.step(dt, d)
        res = PipelineFrameResult(frame_index=int(r["frame_index"]), timestamp=st.total_time, dt=dt, tracked_detections=[],
                                  spatial_detections=[], telemetry_snapshot=agg.aggregate([], frame_index=int(r["frame_index"])),
                                  demand_snapshot=d, controller_state=st, signal_state=map_controller_state(st))
        adapter.process_frame_result(res, force_record=True)
    assert res.current_phase == Phase.PEDESTRIAN_WALK
    _, _, html = sync_bloom_city_with_signal(res.signal_state, demand=res.demand_snapshot, live_adapter=adapter)
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"Chromium unavailable: {exc}")
        page, errors = _open(browser, html)
        _assert_dom_shows(page, res)
        assert page.inner_text("#vsb") == "RED" and page.inner_text("#psb") == "WALK"
        page.screenshot(path=os.environ.get("CG_SHOT", os.path.join(tempfile.gettempdir(), "live_walk.png")))
        assert not errors, errors
        browser.close()
