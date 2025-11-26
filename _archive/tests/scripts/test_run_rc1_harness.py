from __future__ import annotations

import json
from pathlib import Path

import torch

from esper.tools import rc1_harness


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_run_steady_train(tmp_path: Path) -> None:
    result = rc1_harness.run_steady_train(
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        enable_compile=False,
        enable_graphs=False,
    )
    path = rc1_harness._write_results(result, tmp_path, "steady_train")
    data = _read(path)
    assert data["scenario"] == "steady_train"
    assert data["metrics"]["epochs"] == 1.0
    summary = (tmp_path / "summary.csv").read_text().strip().splitlines()
    assert summary and "steady_train" in summary[-1]


def test_run_tamiyo_timeout(tmp_path: Path) -> None:
    result = rc1_harness.run_tamiyo_timeout(
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        timeout_every=1,
        enable_compile=False,
        enable_graphs=False,
    )
    assert result.metrics["timeout_tamiyo"] >= 1.0
    rc1_harness._write_results(result, tmp_path, "tamiyo_timeout")


def test_run_kasmina_prefetch_burst(tmp_path: Path) -> None:
    result = rc1_harness.run_kasmina_prefetch_burst(
        requests=4,
        concurrency=2,
        ready_latency_ms=5.0,
        device=torch.device("cpu"),
    )
    assert result.metrics["requests"] == 4.0
    assert result.metrics["latency_max_ms"] >= 0.0
    rc1_harness._write_results(result, tmp_path, "kasmina_prefetch")


def test_run_rollback_deadline(tmp_path: Path) -> None:
    result = rc1_harness.run_rollback_deadline(
        device=torch.device("cpu"),
        batch_size=2,
        enable_compile=False,
        enable_graphs=False,
        deadline_ms=1.0,
    )
    assert result.metrics["rollback_deadline_exceeded"] >= 1.0
    rc1_harness._write_results(result, tmp_path, "rollback_deadline")
