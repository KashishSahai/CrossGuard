"""
Unit tests for GPU & Hardware Benchmark Runner
"""

from benchmarking.benchmark_runner import run_inference_benchmark


def test_benchmark_runner_execution():
    """Runs a quick lightweight benchmark on host (CPU or CUDA) and verifies statistics."""
    res = run_inference_benchmark(
        model_name="yolov8n.pt",
        warmup_runs=2,
        test_runs=4,
        input_resolution=(320, 320),
    )

    assert res.test_runs == 4
    assert len(res.latencies_ms) == 4
    assert res.mean_latency_ms > 0
    assert res.median_latency_ms > 0
    assert res.fps > 0
    assert res.model_load_time_seconds > 0
    assert res.device in ("cpu", "cuda")
