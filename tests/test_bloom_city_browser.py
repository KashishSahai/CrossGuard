"""Real-browser check of rendered signal lenses and agents (skipped if Playwright/Chromium is absent).

Setup:  pip install playwright && python -m playwright install chromium
"""
import pytest

sync_api = pytest.importorskip("playwright.sync_api")

from core.models import Phase
from simulation.bloom_city_engine import build_frames
from simulation.bloom_city_renderer import render_bloom_city_html

LIT = {"vehicle": {Phase.VEHICLE_GREEN: "g", Phase.VEHICLE_YELLOW: "y", Phase.ALL_RED_CLEARANCE: "r",
                   Phase.PEDESTRIAN_WALK: "r", Phase.PEDESTRIAN_CLEARANCE: "r"}}
COLOR = {"g": "#38d889", "y": "#ffd23f", "r": "#ff5d6c"}


def test_rendered_lenses_agents_and_panel_in_a_real_browser(tmp_path):
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:                                  # browser binary not installed
            pytest.skip(f"Chromium unavailable: {exc}")
        for phase, lit in LIT["vehicle"].items():
            frames, reasons, _ = build_frames("ped_surge", 8, 20, hold_phase=phase)
            f = tmp_path / f"{phase.value}.html"
            f.write_text(render_bloom_city_html(frames, reasons), encoding="utf-8")
            page = browser.new_page(viewport={"width": 1180, "height": 720})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f.as_uri())
            page.evaluate("window.__bloom.draw(4)")
            for i in range(4):
                for k in "ryg":
                    fill = page.get_attribute(f"#vl{i}{k}", "fill")
                    assert (fill == COLOR[k]) == (k == lit), (phase, i, k, fill)
            box = page.eval_on_selector("#scene", "e => e.getBoundingClientRect().toJSON()")
            assert box["width"] > 400 and box["height"] > 300            # not a blank/collapsed canvas
            assert page.locator("#vehicles g").count() >= 1
            assert page.inner_text("#vsb") == {"g": "GREEN", "y": "YELLOW", "r": "RED"}[lit]
            assert page.locator(".chip.on").count() == 1
            assert not errors, errors
            page.close()
        browser.close()
