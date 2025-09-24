from __future__ import annotations

import runpy


def test_bench_script_run_prints_expected_keys(capsys) -> None:
    ns = runpy.run_path("scripts/bench_kasmina.py")
    assert "run" in ns and callable(ns["run"])  # exported function exists
    # Call run() with tiny iteration count to keep test fast
    ns["run"](iterations=2, latency_ms=0.1, repeat=5, feature_shape="8,8")
    out = capsys.readouterr().out
    # Check a few expected keys in output
    assert "Iterations:" in out
    assert "Mean handle time:" in out
    assert "GPU cache hit rate:" in out
    assert "Blend mode:" in out
    assert "Blend latency ms:" in out

