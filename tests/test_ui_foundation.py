"""Unit and smoke tests for the CrossGuard UI Foundation (Step 23)."""
from __future__ import annotations

import os
import py_compile
import pytest

from core.models import PHASE_LABELS, PedestrianSignal, Phase, VehicleSignal
from simulation.bloom_city_engine import SCENARIOS, build_frames
from simulation.bloom_city_renderer import render_bloom_city_html


def test_app_compilation():
    """Verify that app.py compiles with zero syntax errors."""
    app_path = os.path.abspath("app.py")
    assert os.path.exists(app_path), "app.py does not exist"
    # py_compile will raise py_compile.PyCompileError if syntax is invalid
    py_compile.compile(app_path, doraise=True)


def test_ui_branding_and_titles():
    """Verify product naming standards and absence of forbidden strings."""
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()

    assert "CROSSGUARD" in content
    assert "Adaptive Pedestrian-Priority Traffic Management Simulator" in content

    # Check forbidden strings
    forbidden = ["KIIT", "Minor Project", "University Minor", "Antigravity"]
    for word in forbidden:
        assert word.lower() not in content.lower(), f"Forbidden string '{word}' found in app.py"


def test_bloom_city_ui_integration():
    """Verify Bloom City simulation frames and HTML rendering for the UI."""
    assert "balanced" in SCENARIOS
    frames, reasons, _ = build_frames("balanced", seconds=2.0, seed=7)
    assert len(frames) > 0
    assert len(reasons) > 0

    html = render_bloom_city_html(
        frames,
        reasons,
        theme="day",
        mode="SCENARIO DEMO",
        scenario_title=SCENARIOS["balanced"].title,
    )
    assert isinstance(html, str)
    assert len(html) > 1000
    assert "<svg" in html or "<div" in html
