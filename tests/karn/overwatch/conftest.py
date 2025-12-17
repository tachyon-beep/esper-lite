"""Pytest fixtures for Overwatch tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from esper.karn.overwatch import (
    TuiSnapshot,
    EnvSummary,
    SlotChipState,
    TamiyoState,
    ConnectionStatus,
    DeviceVitals,
    FeedEvent,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def healthy_snapshot() -> TuiSnapshot:
    """A snapshot with all envs healthy."""
    return TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-18T12:00:00Z",
        connection=ConnectionStatus(True, 1000.0, 0.5),
        tamiyo=TamiyoState(
            action_counts={"GERMINATE": 10, "BLEND": 20, "CULL": 5, "WAIT": 65},
            recent_actions=["W", "W", "B", "G", "W"],
            kl_divergence=0.019,
            entropy=1.24,
            clip_fraction=0.048,
            explained_variance=0.42,
            confidence_mean=0.73,
        ),
        run_id="healthy-run-001",
        task_name="cifar10",
        episode=47,
        batch=1203,
        best_metric=82.1,
        runtime_s=8040.0,
        devices=[
            DeviceVitals(0, "GPU 0", 94.0, 11.2, 12.0, 72),
            DeviceVitals(1, "GPU 1", 91.0, 10.8, 12.0, 68),
        ],
        flight_board=[
            EnvSummary(
                env_id=0,
                device_id=0,
                status="OK",
                throughput_fps=98.5,
                slots={"r0c1": SlotChipState("r0c1", "FOSSILIZED", "conv_light", 1.0)},
                anomaly_score=0.05,
            ),
            EnvSummary(
                env_id=1,
                device_id=0,
                status="OK",
                throughput_fps=101.2,
                slots={"r0c1": SlotChipState("r0c1", "TRAINING", "mlp_narrow", 0.3)},
                anomaly_score=0.08,
            ),
            EnvSummary(
                env_id=2,
                device_id=1,
                status="OK",
                throughput_fps=99.0,
                slots={"r0c1": SlotChipState("r0c1", "BLENDING", "conv_light", 0.7, gate_last="G2", gate_passed=True)},
                anomaly_score=0.1,
            ),
            EnvSummary(
                env_id=3,
                device_id=1,
                status="OK",
                throughput_fps=97.8,
                slots={"r0c1": SlotChipState("r0c1", "GERMINATED", "conv_light", 0.1)},
                anomaly_score=0.03,
            ),
        ],
        envs_ok=4,
        envs_warn=0,
        envs_crit=0,
    )


@pytest.fixture
def anomaly_snapshot() -> TuiSnapshot:
    """A snapshot with anomalies detected."""
    return TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-18T12:05:00Z",
        connection=ConnectionStatus(True, 1000.0, 0.5),
        tamiyo=TamiyoState(
            kl_divergence=0.08,  # High KL
            entropy=0.15,  # Low entropy (collapsed)
            entropy_collapsed=True,
            explained_variance=0.25,  # Low EV
            ev_warning=True,
        ),
        run_id="anomaly-run-001",
        task_name="cifar10",
        episode=100,
        flight_board=[
            EnvSummary(
                env_id=0,
                device_id=0,
                status="OK",
                anomaly_score=0.1,
            ),
            EnvSummary(
                env_id=1,
                device_id=0,
                status="WARN",
                throughput_fps=45.0,  # Low throughput
                anomaly_score=0.65,
                anomaly_reasons=["Throughput 55% below baseline"],
            ),
            EnvSummary(
                env_id=2,
                device_id=1,
                status="CRIT",
                reward_last=-2.5,
                anomaly_score=0.85,
                anomaly_reasons=[
                    "Unusual negative reward (-2.5)",
                    "High gradient ratio (15.2x)",
                    "Memory pressure (97%)",
                ],
                slots={"r0c1": SlotChipState("r0c1", "CULLED", "bad_blueprint", 0.0)},
            ),
        ],
        envs_ok=1,
        envs_warn=1,
        envs_crit=1,
    )


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to fixtures directory."""
    return FIXTURES_DIR
