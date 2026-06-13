"""Tests for proof packet generation."""

from __future__ import annotations

import json
import importlib.util
from datetime import datetime
from pathlib import Path


_PROOF_PACKET_PATH = Path(__file__).parents[2] / "scripts" / "proof_packet.py"
_SPEC = importlib.util.spec_from_file_location("proof_packet", _PROOF_PACKET_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_PROOF_PACKET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROOF_PACKET)
build_proof_packet = _PROOF_PACKET.build_proof_packet


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
    return payload


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
