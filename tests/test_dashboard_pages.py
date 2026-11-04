"""
Unit tests for Dashboard Modules, Routing & Reusable Components
"""

from core.config import NAV_PAGES, SUBSYSTEM_REGISTRY, VERSION
from dashboard.analytics import render_analytics_page
from dashboard.bloom_city import render_bloom_city_page
from dashboard.components import (
    get_command_centre_plotly_layout,
    render_hud_header,
    render_metric_card,
    render_status_badge,
    render_traffic_signal_head,
)
from dashboard.gpu_performance import render_gpu_performance_page
from dashboard.overview import render_overview_page
from dashboard.signal_control import render_signal_control_page
from dashboard.vision_monitor import render_vision_monitor_page


def test_nav_pages_configuration():
    """Ensures all six pages are registered in NAV_PAGES."""
    expected_pages = [
        "overview",
        "vision_monitor",
        "bloom_city",
        "signal_control",
        "analytics",
        "gpu_performance",
    ]
    for page_id in expected_pages:
        assert page_id in NAV_PAGES
        assert "title" in NAV_PAGES[page_id]
        assert "description" in NAV_PAGES[page_id]


def test_subsystem_registry():
    """Verifies all architectural subsystems are properly documented."""
    assert len(SUBSYSTEM_REGISTRY) >= 6
    subsystem_ids = [sub.id for sub in SUBSYSTEM_REGISTRY]
    assert "video_ingest" in subsystem_ids
    assert "yolo_detector" in subsystem_ids
    assert "bytetrack_tracker" in subsystem_ids
    assert "zone_analyzer" in subsystem_ids
    assert "adaptive_controller" in subsystem_ids
    assert "bloom_city_bridge" in subsystem_ids


def test_plotly_layout_cyber_styling():
    """Ensures Plotly dark cyber styling returns valid dictionary."""
    layout = get_command_centre_plotly_layout(title="TEST_CHART", height=300)
    assert isinstance(layout, dict)
    assert layout["height"] == 300
    assert "paper_bgcolor" in layout
    assert "plot_bgcolor" in layout


def test_dashboard_page_callables():
    """Verifies that all page renderers are callable functions."""
    assert callable(render_overview_page)
    assert callable(render_vision_monitor_page)
    assert callable(render_bloom_city_page)
    assert callable(render_signal_control_page)
    assert callable(render_analytics_page)
    assert callable(render_gpu_performance_page)


def test_all_six_pages_with_apptest():
    """Renders the full Streamlit app and navigates across all six pages using AppTest."""
    import os
    from streamlit.testing.v1 import AppTest

    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    nav_radio = at.sidebar.radio[0]
    for opt in nav_radio.options:
        nav_radio.set_value(opt)
        at.run()
        assert len(at.exception) == 0, f"Exception rendered on page option: {opt}"
        # Assert no raw table or block tags leaked into markdown text elements
        for md in at.markdown:
            assert "<tr>" not in md.value, f"Raw <tr> string leaked in markdown on page {opt}: {md.value[:100]}"
            assert "<td>" not in md.value, f"Raw <td> string leaked in markdown on page {opt}: {md.value[:100]}"


