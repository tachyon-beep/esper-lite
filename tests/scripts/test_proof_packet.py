"""Tests for proof packet generation."""

from __future__ import annotations

import json
import importlib.util
from datetime import datetime
from pathlib import Path

import pytest


_PROOF_PACKET_PATH = Path(__file__).parents[2] / "scripts" / "proof_packet.py"
_SPEC = importlib.util.spec_from_file_location("proof_packet", _PROOF_PACKET_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_PROOF_PACKET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROOF_PACKET)
build_proof_packet = _PROOF_PACKET.build_proof_packet


def _training_started_event(
    event_id: str,
    group_id: str,
    proof_baseline_mode: str,
) -> dict:
    return {
        "event_id": event_id,
        "event_type": "TRAINING_STARTED",
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id,
        "data": {
            "episode_id": "proof",
            "task": "cifar_impaired",
            "reward_mode": "simplified",
            "n_envs": 2,
            "n_episodes": 2,
            "max_epochs": 25,
            "max_batches": 1,
            "proof_baseline_mode": proof_baseline_mode,
            "proof_baseline_pair_id": "blueprint-health-proof",
        },
        "severity": "info",
    }


def _ppo_update_payload() -> dict:
    heads = (
        "slot",
        "blueprint",
        "style",
        "tempo",
        "alpha_target",
        "alpha_speed",
        "alpha_curve",
        "op",
    )
    payload = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.0,
        "grad_norm": 0.3,
        "kl_divergence": 0.01,
        "clip_fraction": 0.0,
    }
    for head in heads:
        payload[f"head_{head}_learnable_fraction"] = 1.0
        payload[f"head_{head}_gradient_state"] = "finite"
    payload["head_value_grad_norm"] = 0.4
    payload["head_value_gradient_state"] = "finite"
    return payload


def _ppo_update_event(event_id: str, group_id: str, epoch: int = 1) -> dict:
    return {
        "event_id": event_id,
        "event_type": "PPO_UPDATE_COMPLETED",
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "group_id": group_id,
        "data": _ppo_update_payload(),
        "severity": "info",
    }


def _episode_outcome_event(
    event_id: str,
    group_id: str,
    *,
    param_ratio: float = 1.0,
    final_accuracy: float = 80.0,
    epoch: int = 1,
) -> dict:
    return {
        "event_id": event_id,
        "event_type": "EPISODE_OUTCOME",
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "group_id": group_id,
        "data": {
            "env_id": 0,
            "episode_idx": 0,
            "final_accuracy": final_accuracy,
            "param_ratio": param_ratio,
            "num_fossilized": 1,
            "num_contributing_fossilized": 1,
            "episode_reward": 5.0,
            "stability_score": 1.0,
            "reward_mode": "simplified",
            "episode_length": 25,
            "outcome_type": "success",
            "germinate_count": 1,
            "prune_count": 0,
            "fossilize_count": 1,
        },
        "severity": "info",
    }


def test_proof_packet_blocks_verdict_when_confounders_present(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = [
        {
            "event_id": "start-1",
            "event_type": "TRAINING_STARTED",
            "timestamp": datetime.now().isoformat(),
            "group_id": "A",
            "data": {
                "episode_id": "proof",
                "task": "cifar_impaired",
                "reward_mode": "simplified",
                "n_envs": 2,
                "n_episodes": 2,
                "max_epochs": 25,
            },
            "severity": "info",
        },
        {
            "event_id": "ppo-1",
            "event_type": "PPO_UPDATE_COMPLETED",
            "timestamp": datetime.now().isoformat(),
            "epoch": 1,
            "group_id": "A",
            "data": _ppo_update_payload(),
            "severity": "info",
        },
        {
            "event_id": "bad-1",
            "event_type": "NUMERICAL_INSTABILITY_DETECTED",
            "timestamp": datetime.now().isoformat(),
            "epoch": 1,
            "group_id": "A",
            "data": {
                "anomaly_type": "numerical_instability",
                "env_id": 0,
                "episode": 1,
                "batch": 1,
                "inner_epoch": 7,
                "total_episodes": 2,
                "detail": "nonfinite policy loss",
            },
            "severity": "error",
        },
        {
            "event_id": "seed-1",
            "event_type": "SEED_GERMINATED",
            "timestamp": datetime.now().isoformat(),
            "seed_id": "seed-1",
            "slot_id": "r0c0",
            "epoch": 1,
            "group_id": "A",
            "data": {
                "env_id": 0,
                "episode_idx": 0,
                "blueprint_id": "conv_small",
                "params": 8,
            },
            "severity": "info",
        },
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(str(tmp_path))

    assert "Verdict: `BLOCKED`" in packet
    assert "## Reproduction Commands" in packet
    assert "--task cifar_impaired" in packet
    assert "--rounds" in packet
    assert "## Lifecycle Efficiency" in packet
    assert "germinated=1" in packet
    assert "NUMERICAL_INSTABILITY_DETECTED" in packet
    assert "nonfinite policy loss" in packet


def test_proof_packet_blocks_missing_learnability_telemetry(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    event = {
        "event_id": "ppo-1",
        "event_type": "PPO_UPDATE_COMPLETED",
        "timestamp": datetime.now().isoformat(),
        "epoch": 1,
        "group_id": "A",
        "data": {
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 1.0,
            "grad_norm": 0.3,
            "kl_divergence": 0.01,
            "clip_fraction": 0.0,
        },
        "severity": "info",
    }
    (run_dir / "events.jsonl").write_text(json.dumps(event) + "\n")

    packet = build_proof_packet(str(tmp_path))

    assert "Verdict: `BLOCKED`" in packet
    assert "missing learnability telemetry" in packet


def test_proof_packet_blocks_missing_blueprint_health_baselines(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = [
        _training_started_event("start-1", "off", "off_switch"),
        _training_started_event("start-2", "initial", "static_initial"),
        _training_started_event("start-3", "schedule", "fixed_schedule"),
        _training_started_event("start-4", "lockstep-a", "lockstep_reward_ab"),
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED`" in packet
    assert "## Blueprint-Health Baselines" in packet
    assert "missing required proof baseline `static_final`" in packet


def test_proof_packet_accepts_complete_blueprint_health_baseline_set(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = [
        _training_started_event("start-1", "off", "off_switch"),
        _training_started_event("start-2", "initial", "static_initial"),
        _training_started_event("start-3", "final", "static_final"),
        _training_started_event("start-4", "schedule", "fixed_schedule"),
        _training_started_event("start-5", "lockstep-a", "lockstep_reward_ab"),
        _training_started_event("start-6", "lockstep-b", "lockstep_reward_ab"),
        # Learnability + outcomes present so only the baseline gate is exercised.
        _ppo_update_event("ppo-1", "off"),
        _episode_outcome_event("eo-1", "off"),
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `REVIEW`" in packet
    assert "All required blueprint-health proof baselines are present." in packet


# ---------------------------------------------------------------------------
# KARN-PROOF-001: fail closed on missing runs / outcomes / empty telemetry
# ---------------------------------------------------------------------------


def test_proof_packet_blocks_empty_telemetry(tmp_path):
    """No events.jsonl files at all must BLOCK with explicit text."""
    packet = build_proof_packet(str(tmp_path))

    assert "Verdict: `BLOCKED`" in packet
    assert "BLOCKING empty telemetry" in packet


def test_proof_packet_blocks_missing_training_started(tmp_path):
    """Outcomes present but no TRAINING_STARTED must BLOCK (no cohort identity)."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = [
        _ppo_update_event("ppo-1", "A"),
        _episode_outcome_event("eo-1", "A"),
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path))

    assert "Verdict: `BLOCKED`" in packet
    assert "BLOCKING no `TRAINING_STARTED` events found" in packet


def test_proof_packet_blocks_missing_episode_outcome(tmp_path):
    """Runs + learnability present but no EPISODE_OUTCOME must BLOCK (no ROI)."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = [
        _training_started_event("start-1", "A", "static_initial"),
        _ppo_update_event("ppo-1", "A"),
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path))

    assert "Verdict: `BLOCKED`" in packet
    assert "BLOCKING no `EPISODE_OUTCOME` events found" in packet


# ---------------------------------------------------------------------------
# KARN-PROOF-002: rollback / reward-hacking / degradation are proof confounders
# ---------------------------------------------------------------------------


def _baseline_run_events() -> list[dict]:
    """A run that would otherwise pass: training started, learnability, outcomes."""
    return [
        {
            "event_id": "start-1",
            "event_type": "TRAINING_STARTED",
            "timestamp": datetime.now().isoformat(),
            "group_id": "A",
            "data": {
                "episode_id": "proof",
                "task": "cifar_impaired",
                "reward_mode": "simplified",
                "n_envs": 2,
                "n_episodes": 2,
                "max_epochs": 25,
                "max_batches": 1,
            },
            "severity": "info",
        },
        _ppo_update_event("ppo-1", "A"),
        _episode_outcome_event("eo-1", "A"),
    ]


@pytest.mark.parametrize(
    "event_type, data, marker",
    [
        (
            "GOVERNOR_ROLLBACK",
            {
                "env_id": 0,
                "device": "cpu",
                "reason": "governor_nan",
                "episode_idx": 1,
            },
            "GOVERNOR_ROLLBACK",
        ),
        (
            "REWARD_HACKING_SUSPECTED",
            {
                "pattern": "attribution_ratio",
                "slot_id": "r0c0",
                "seed_id": "seed-1",
                "seed_contribution": 0.9,
                "total_improvement": 0.1,
                "ratio": 9.0,
                "threshold": 2.0,
            },
            "REWARD_HACKING_SUSPECTED",
        ),
        (
            "PERFORMANCE_DEGRADATION",
            {
                "env_id": 0,
                "current_acc": 40.0,
                "rolling_avg_acc": 80.0,
                "drop_percent": 50.0,
                "threshold_percent": 20.0,
                "episode_idx": 1,
            },
            "PERFORMANCE_DEGRADATION",
        ),
    ],
)
def test_proof_packet_blocks_on_integrity_confounders(tmp_path, event_type, data, marker):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _baseline_run_events()
    events.append(
        {
            "event_id": "confounder-1",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "epoch": 1,
            "group_id": "A",
            "data": data,
            "severity": "error",
        }
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path))

    assert "Verdict: `BLOCKED`" in packet
    assert f"BLOCKING `{marker}`" in packet


# ---------------------------------------------------------------------------
# KARN-PROOF-003: fail-closed ingestion integrity
# ---------------------------------------------------------------------------


def test_proof_packet_blocks_malformed_jsonl(tmp_path):
    """A corrupt JSONL line is detected, reported, and BLOCKS the verdict."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    good_lines = [json.dumps(event) for event in _baseline_run_events()]
    # Deliberately corrupt line: truncated JSON that DuckDB would silently skip.
    corrupt = '{"event_id": "bad", "event_type": "EPISODE_OUTCOME", "data": {'
    (run_dir / "events.jsonl").write_text("\n".join(good_lines + [corrupt]) + "\n")

    packet = build_proof_packet(str(tmp_path))

    assert "Verdict: `BLOCKED`" in packet
    assert "BLOCKING malformed telemetry" in packet
    assert "1 JSONL line(s) failed to parse" in packet
    assert "events.jsonl" in packet


# ---------------------------------------------------------------------------
# KARN-PROOF-004: param_ratio semantic end-to-end through the packet ROI
# ---------------------------------------------------------------------------


def test_proof_packet_roi_uses_total_over_host_param_ratio(tmp_path):
    """A no-growth (1.0) and a 20%-growth (1.2) episode produce ROI = acc/ratio.

    Under the canonical semantic param_ratio = total/host, accuracy ROI is
    final_accuracy / param_ratio. For final_accuracy=90:
      - no growth (1.0): ROI = 90.0
      - 20% growth (1.2): ROI = 75.0
    """
    no_growth = tmp_path / "no_growth"
    no_growth.mkdir()
    no_growth_events = [
        {
            "event_id": "start-1",
            "event_type": "TRAINING_STARTED",
            "timestamp": datetime.now().isoformat(),
            "group_id": "A",
            "data": {
                "episode_id": "proof",
                "task": "cifar_impaired",
                "reward_mode": "simplified",
                "n_envs": 1,
                "n_episodes": 1,
                "max_epochs": 25,
                "max_batches": 1,
            },
            "severity": "info",
        },
        _ppo_update_event("ppo-1", "A"),
        _episode_outcome_event("eo-1", "A", param_ratio=1.0, final_accuracy=90.0),
    ]
    (no_growth / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in no_growth_events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path))

    # mean_param_ratio reports the 1.0 (no-growth) value verbatim, and ROI = 90/1.0.
    assert "mean_param_ratio=1.0" in packet
    assert "mean_accuracy_roi=90.0" in packet
