"""Shared assertions for live Bloom City HTML: check the frame data, not incidental text."""
import json
import re

from core.models import PHASE_ORDER, PHASE_OUTPUTS

_VS = ("GREEN", "YELLOW", "RED")
_PS = ("DONT_WALK", "WALK", "CLEARING")


def payload_of(html: str) -> dict:
    m = re.search(r'<script id="bcdata" type="application/json">(.*?)</script>', html, re.DOTALL)
    assert m is not None, "bcdata payload missing from Bloom City HTML"
    return json.loads(m.group(1).replace("<\\/", "</"))


def assert_html_latest_frame_is(html: str, phase) -> None:
    """The newest frame in the rendered payload must carry exactly this phase's signal outputs."""
    data = payload_of(html)
    assert "LIVE" in data["mode"], data["mode"]
    last = data["frames"][-1]
    v_sig, p_sig = PHASE_OUTPUTS[phase]
    assert PHASE_ORDER[last["ph"]] == phase
    assert _VS[last["vs"]] == v_sig.value
    assert _PS[last["ps"]] == p_sig.value
