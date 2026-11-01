"""
Unit tests for System Hardware Telemetry & Probe
"""

from core.hardware_info import get_hardware_telemetry


def test_hardware_telemetry_fields():
    """Verifies that all required system telemetry metrics are collected."""
    hw = get_hardware_telemetry()
    
    assert isinstance(hw, dict)
    assert "os_name" in hw
    assert "cpu_cores_physical" in hw
    assert "cpu_cores_logical" in hw
    assert "ram_total_gb" in hw
    assert "torch_version" in hw
    assert "cuda_available" in hw
    assert "acceleration_mode" in hw
    assert "benchmark_status" in hw
    
    # Check logical ranges
    assert hw["cpu_cores_logical"] >= 1
    assert hw["ram_total_gb"] > 0
    assert isinstance(hw["cuda_available"], bool)
