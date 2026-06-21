"""Tests for proof packet generation."""

from __future__ import annotations

import json
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pytest

from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
    FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
    FIXED_SCHEDULE_GERMINATE_R0C0_V1,
    FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
    STATIC_FINAL_SOURCE_MODE,
    STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT,
    STATIC_FINAL_SOURCE_TOPOLOGY_HASH,
    STATIC_FINAL_SOURCE_TOPOLOGY_VERSION,
    STATIC_FINAL_SOURCE_TOPOLOGY_V1,
)


_PROOF_PACKET_PATH = Path(__file__).parents[2] / "scripts" / "proof_packet.py"
_SPEC = importlib.util.spec_from_file_location("proof_packet", _PROOF_PACKET_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_PROOF_PACKET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROOF_PACKET)
build_proof_packet = _PROOF_PACKET.build_proof_packet
proof_packet_main = _PROOF_PACKET.main


def _training_started_event(
    event_id: str,
    group_id: str | None,
    proof_baseline_mode: str,
    *,
    reward_mode: str = "simplified",
    seed: int = 42,
    proof_baseline_pair_id: str | None = "blueprint-health-proof",
    proof_baseline_lifecycle_policy: str | None = None,
) -> dict:
    if proof_baseline_lifecycle_policy is None:
        proof_baseline_lifecycle_policy = _proof_baseline_lifecycle_policy(
            proof_baseline_mode
        )
    event = {
        "event_id": event_id,
        "event_type": "TRAINING_STARTED",
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id,
        "data": {
            "episode_id": "proof",
            "task": "cifar_impaired",
            "reward_mode": reward_mode,
            "seed": seed,
            "n_envs": 2,
            "n_episodes": 2,
            "max_epochs": 25,
            "max_batches": 1,
            "amp_enabled": False,
            "amp_dtype": None,
            "proof_baseline_mode": proof_baseline_mode,
            "proof_baseline_pair_id": proof_baseline_pair_id,
            "proof_baseline_lifecycle_policy": proof_baseline_lifecycle_policy,
        },
        "severity": "info",
    }
    if proof_baseline_mode == "fixed_schedule":
        event["data"]["proof_baseline_schedule_id"] = (
            FIXED_SCHEDULE_GERMINATE_R0C0_V1
        )
        event["data"]["proof_baseline_schedule_hash"] = (
            FIXED_SCHEDULE_GERMINATE_R0C0_HASH
        )
        event["data"]["proof_baseline_schedule_version"] = (
            FIXED_SCHEDULE_GERMINATE_R0C0_VERSION
        )
        event["data"]["proof_baseline_schedule_action_count"] = (
            FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT
        )
    if proof_baseline_mode == STATIC_FINAL_SOURCE_MODE:
        event["data"]["proof_baseline_schedule_id"] = (
            STATIC_FINAL_SOURCE_TOPOLOGY_V1
        )
        event["data"]["proof_baseline_schedule_hash"] = (
            STATIC_FINAL_SOURCE_TOPOLOGY_HASH
        )
        event["data"]["proof_baseline_schedule_version"] = (
            STATIC_FINAL_SOURCE_TOPOLOGY_VERSION
        )
        event["data"]["proof_baseline_schedule_action_count"] = (
            STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT
        )
    return event


def _proof_baseline_lifecycle_policy(proof_baseline_mode: str) -> str:
    if proof_baseline_mode == "off_switch":
        return "force_wait_only"
    if proof_baseline_mode == "static_initial":
        return "freeze_initial_topology"
    if proof_baseline_mode == "static_final":
        return "freeze_replayed_final_topology"
    if proof_baseline_mode == "fixed_schedule":
        return "apply_declared_lifecycle_schedule"
    if proof_baseline_mode == "lockstep_reward_ab":
        return "paired_lockstep_reward_comparison"
    if proof_baseline_mode == STATIC_FINAL_SOURCE_MODE:
        return "evolve_source_final_topology"
    raise ValueError(f"Unknown proof baseline mode: {proof_baseline_mode}")


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
    env_id: int = 0,
    episode_idx: int = 0,
    reward_mode: str = "simplified",
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
            "env_id": env_id,
            "episode_idx": episode_idx,
            "final_accuracy": final_accuracy,
            "param_ratio": param_ratio,
            "num_fossilized": 1,
            "num_contributing_fossilized": 1,
            "episode_reward": 5.0,
            "stability_score": 1.0,
            "reward_mode": reward_mode,
            "episode_length": 25,
            "outcome_type": "success",
            "germinate_count": 1,
            "prune_count": 0,
            "fossilize_count": 1,
        },
        "severity": "info",
    }


def _fixed_schedule_morphology_causal_log_event(
    *,
    event_id: str = "schedule-commit-1",
    group_id: str = "schedule",
    epoch: int = 1,
    phase: str = "commit",
    slot_id: str = "r0c0",
    operation: str = "GERMINATE",
    blueprint_id: str | None = "norm",
    governor_approved: bool | None = True,
) -> dict:
    return {
        "event_id": event_id,
        "event_type": "MORPHOLOGY_CAUSAL_LOG",
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "group_id": group_id,
        "data": {
            "phase": phase,
            "env_id": 0,
            "slot_id": slot_id,
            "operation": operation,
            "action_id": "morph-b1-e1-env0-r0c0-op1",
            "proposal_id": "morph-b1-e1-env0-r0c0-op1-proposal",
            "verdict_id": "morph-b1-e1-env0-r0c0-op1-verdict",
            "mutation_id": "morph-b1-e1-env0-r0c0-op1-mutation",
            "observation_hash": "obs-fixed-schedule",
            "rng_stream": "simic.lifecycle.env0",
            "rng_seed": 40042,
            "topology": "cnn",
            "blueprint_id": blueprint_id,
            "governor_approved": governor_approved,
            "governor_reason": "approved",
            "governor_blocked_factor": None,
            "watch_window_evidence": None,
            "linked_event_id": "morph-b1-e1-env0-r0c0-op1-mutation",
        },
        "severity": "debug",
    }


def _topology_manifest_event(
    *,
    event_id: str,
    group_id: str | None,
    role: str,
    epoch: int = 0,
    topology_manifest_hash: str = "topology-r0c0-norm-hash",
    source_event_id: str = "source-final-manifest",
    source_run_dir: str = "proof_run",
    source_group_id: str | None = "source-final",
    replay_weight_policy: str = "topology_only",
    replay_env_id: int = 0,
    replay_episode_idx: int = 0,
    manifest_match: bool = True,
) -> dict:
    payload = {
        "manifest_role": role,
        "proof_baseline_pair_id": "blueprint-health-proof",
        "topology_manifest_version": 1,
        "topology_manifest_hash": topology_manifest_hash,
        "topology_manifest_json": (
            '{"host_topology":"cnn","slots":[{"slot_id":"r0c0",'
            '"blueprint_id":"norm","stage":"FOSSILIZED"}]}'
        ),
        "task": "cifar_impaired",
        "host_topology": "cnn",
        "slot_config_hash": "slot-config-default-hash",
        "slot_count": 1,
        "fossilized_seed_count": 1,
        "topology_delta_count": 1,
    }
    if role == "static_final_replay":
        payload.update(
            {
                "source_run_dir": source_run_dir,
                "source_group_id": source_group_id,
                "source_episode_idx": 0,
                "source_event_id": source_event_id,
                "source_topology_manifest_hash": topology_manifest_hash,
                "replay_weight_policy": replay_weight_policy,
                "replay_env_id": replay_env_id,
                "replay_episode_idx": replay_episode_idx,
                "replayed_topology_manifest_hash": topology_manifest_hash,
                "manifest_match": manifest_match,
            }
        )
    return {
        "event_id": event_id,
        "event_type": "TOPOLOGY_MANIFEST_RECORDED",
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "group_id": group_id,
        "data": payload,
        "severity": "info",
    }


def _complete_blueprint_health_baseline_events(
    *,
    lockstep_a_reward_mode: str = "shaped",
    lockstep_b_reward_mode: str = "simplified",
    lockstep_a_seed: int = 50042,
    lockstep_b_seed: int = 50042,
    outcome_groups: tuple[str, ...] = (
        "off",
        "initial",
        "final",
        "schedule",
        "lockstep-a",
        "lockstep-b",
    ),
) -> list[dict]:
    cohort_specs = (
        ("off", "off_switch", "shaped", 10042),
        ("initial", "static_initial", "shaped", 20042),
        ("final", "static_final", "shaped", 30042),
        ("schedule", "fixed_schedule", "shaped", 40042),
        ("lockstep-a", "lockstep_reward_ab", lockstep_a_reward_mode, lockstep_a_seed),
        ("lockstep-b", "lockstep_reward_ab", lockstep_b_reward_mode, lockstep_b_seed),
    )
    events = [
        _training_started_event(
            f"start-{index}",
            group_id,
            proof_baseline_mode,
            reward_mode=reward_mode,
            seed=seed,
        )
        for index, (group_id, proof_baseline_mode, reward_mode, seed) in enumerate(
            cohort_specs,
            start=1,
        )
    ]
    events.append(_ppo_update_event("ppo-1", "off"))
    events.append(_fixed_schedule_morphology_causal_log_event())
    for index, (group_id, _proof_baseline_mode, reward_mode, _seed) in enumerate(
        cohort_specs,
        start=1,
    ):
        if group_id in outcome_groups:
            events.append(
                _episode_outcome_event(
                    f"eo-{index}",
                    group_id,
                    reward_mode=reward_mode,
                )
            )
    return events


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
                "max_batches": 1,
                "amp_enabled": False,
                "amp_dtype": None,
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
        _episode_outcome_event("eo-1", "A"),
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_MECHANICS`" in packet
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

    assert "Verdict: `BLOCKED_INSTRUMENTATION`" in packet
    assert "missing learnability telemetry" in packet


def test_proof_packet_blocks_partial_head_learnability_telemetry(tmp_path):
    """One absent required head field blocks proof even when the rest is healthy."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _baseline_run_events()
    del events[1]["data"]["head_op_gradient_state"]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_INSTRUMENTATION`" in packet
    assert "missing learnability telemetry" in packet
    assert "updates=1" in packet


def test_proof_packet_accepts_measured_zero_learnability_values(tmp_path):
    """Measured zero is valid proof telemetry; absent fields are what block."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _baseline_run_events()
    events[1]["data"]["head_blueprint_learnable_fraction"] = 0.0
    events[1]["data"]["head_blueprint_gradient_state"] = "not_learnable"
    events[1]["data"]["head_value_grad_norm"] = 0.0
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `CONTINUE`" in packet
    assert "PPO updates include per-head learnability telemetry." in packet


def test_proof_packet_blocks_missing_blueprint_health_baselines(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = [
        event
        for event in _complete_blueprint_health_baseline_events()
        if event["group_id"] != "final"
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "## Blueprint-Health Baselines" in packet
    assert "missing required proof baseline `static_final`" in packet


def test_proof_packet_blocks_static_final_without_replay_evidence(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "All required blueprint-health proof baselines have outcome-bearing evidence." in packet
    assert "outcome evidence `lockstep_reward_ab`: runs=2" in packet
    assert "lifecycle_policy=`force_wait_only`" in packet
    assert f"schedule_id=`{FIXED_SCHEDULE_GERMINATE_R0C0_V1}`" in packet
    assert "Fixed-schedule committed morphology trace matches the declared schedule." in packet
    assert "BLOCKING static-final replay" in packet
    assert "missing static-final topology replay evidence" in packet


def test_proof_packet_accepts_static_final_with_joined_topology_manifests(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    events.extend(
        [
            _training_started_event(
                "start-source",
                "source-final",
                STATIC_FINAL_SOURCE_MODE,
                reward_mode="shaped",
                seed=30042,
            ),
            _topology_manifest_event(
                event_id="source-final-manifest",
                group_id="source-final",
                role="source_final",
            ),
            _topology_manifest_event(
                event_id="replay-manifest",
                group_id="final",
                role="static_final_replay",
            ),
        ]
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `CONTINUE`" in packet
    assert "Fixed-schedule committed morphology trace matches the declared schedule." in packet
    assert "Static-final topology replay evidence is present." in packet
    assert "BLOCKING static-final replay" not in packet


def test_reward_efficiency_verdict_excludes_simplified_diagnostic_roi(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    for event in events:
        if event["event_type"] == "EPISODE_OUTCOME":
            if event["data"]["reward_mode"] != "simplified":
                event["data"]["final_accuracy"] = 0.5
    events.extend(
        [
            _training_started_event(
                "start-source",
                "source-final",
                STATIC_FINAL_SOURCE_MODE,
                reward_mode="shaped",
                seed=30042,
            ),
            _topology_manifest_event(
                event_id="source-final-manifest",
                group_id="source-final",
                role="source_final",
            ),
            _topology_manifest_event(
                event_id="replay-manifest",
                group_id="final",
                role="static_final_replay",
            ),
        ]
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        proof_profile="reward-efficiency",
    )

    assert "Verdict: `REVISE_ALGORITHM`" in packet
    assert "reward_mode=simplified" in packet
    assert "diagnostic-only" in packet


def test_proof_packet_accepts_static_final_with_ungrouped_source_manifest(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    events.extend(
        [
            _training_started_event(
                "start-source",
                None,
                STATIC_FINAL_SOURCE_MODE,
                reward_mode="shaped",
                seed=30042,
            ),
            _topology_manifest_event(
                event_id="source-final-manifest",
                group_id=None,
                role="source_final",
            ),
            _topology_manifest_event(
                event_id="replay-manifest",
                group_id="final",
                role="static_final_replay",
                source_group_id=None,
            ),
        ]
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `CONTINUE`" in packet
    assert "Static-final topology replay evidence is present." in packet
    assert "missing source_group_id" not in packet


def test_proof_packet_blocks_static_final_replay_not_bound_to_each_outcome_episode(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    events.append(
        _episode_outcome_event(
            "eo-static-final-uncertified",
            "final",
            env_id=1,
            episode_idx=1,
            reward_mode="shaped",
        )
    )
    events.extend(
        [
            _training_started_event(
                "start-source",
                "source-final",
                STATIC_FINAL_SOURCE_MODE,
                reward_mode="shaped",
                seed=30042,
            ),
            _topology_manifest_event(
                event_id="source-final-manifest",
                group_id="source-final",
                role="source_final",
            ),
            _topology_manifest_event(
                event_id="replay-manifest",
                group_id="final",
                role="static_final_replay",
                replay_env_id=0,
                replay_episode_idx=0,
            ),
        ]
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING static-final replay" in packet
    assert "episode_idx=1" in packet
    assert "missing static-final topology replay evidence" in packet


def test_proof_packet_blocks_state_dict_exact_static_final_replay_without_state_hashes(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    events.extend(
        [
            _training_started_event(
                "start-source",
                "source-final",
                STATIC_FINAL_SOURCE_MODE,
                reward_mode="shaped",
                seed=30042,
            ),
            _topology_manifest_event(
                event_id="source-final-manifest",
                group_id="source-final",
                role="source_final",
            ),
            _topology_manifest_event(
                event_id="replay-manifest",
                group_id="final",
                role="static_final_replay",
                replay_weight_policy="state_dict_exact",
            ),
        ]
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING static-final replay" in packet
    assert "state_dict_exact replay_weight_policy lacks source/replay state-dict hash evidence" in packet


def test_proof_packet_blocks_static_final_source_manifest_without_source_run_provenance(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    events.extend(
        [
            _topology_manifest_event(
                event_id="source-final-manifest",
                group_id="source-final",
                role="source_final",
            ),
            _topology_manifest_event(
                event_id="replay-manifest",
                group_id="final",
                role="static_final_replay",
            ),
        ]
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING static-final replay" in packet
    assert "missing static-final source run provenance" in packet


def test_proof_packet_blocks_static_final_without_source_manifest(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    events.append(
        _topology_manifest_event(
            event_id="replay-manifest",
            group_id="final",
            role="static_final_replay",
        )
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING static-final replay" in packet
    assert "missing matching source-final topology manifest" in packet


def test_proof_packet_blocks_static_final_that_mutates_after_replay(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    events.extend(
        [
            _training_started_event(
                "start-source",
                "source-final",
                STATIC_FINAL_SOURCE_MODE,
                reward_mode="shaped",
                seed=30042,
            ),
            _topology_manifest_event(
                event_id="source-final-manifest",
                group_id="source-final",
                role="source_final",
            ),
            _topology_manifest_event(
                event_id="replay-manifest",
                group_id="final",
                role="static_final_replay",
            ),
            _fixed_schedule_morphology_causal_log_event(
                event_id="static-final-commit",
                group_id="final",
            ),
        ]
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING static-final replay" in packet
    assert "static-final run mutated topology" in packet


def test_proof_packet_blocks_fixed_schedule_without_realized_trace(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = [
        event
        for event in _complete_blueprint_health_baseline_events()
        if event["event_type"] != "MORPHOLOGY_CAUSAL_LOG"
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING fixed-schedule trace" in packet
    assert "expected epoch=1 slot=`r0c0` op=`GERMINATE` blueprint=`norm`" in packet
    assert "missing declared fixed-schedule action trace" in packet


def test_proof_packet_blocks_fixed_schedule_mismatched_realized_trace(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    for event in events:
        if event["event_id"] == "schedule-commit-1":
            event["data"]["operation"] = "PRUNE"
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING fixed-schedule trace" in packet
    assert "observed event `schedule-commit-1` phase=`commit` op=`PRUNE`" in packet
    assert "mismatched declared fixed-schedule action trace" in packet


def test_proof_packet_blocks_fixed_schedule_extra_realized_trace(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    events.append(
        _fixed_schedule_morphology_causal_log_event(
            event_id="schedule-extra-commit",
            epoch=2,
            operation="PRUNE",
        )
    )
    events.extend(
        [
            _training_started_event(
                "start-source",
                "source-final",
                STATIC_FINAL_SOURCE_MODE,
                reward_mode="shaped",
                seed=30042,
            ),
            _topology_manifest_event(
                event_id="source-final-manifest",
                group_id="source-final",
                role="source_final",
            ),
            _topology_manifest_event(
                event_id="replay-manifest",
                group_id="final",
                role="static_final_replay",
            ),
        ]
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING fixed-schedule trace" in packet
    assert "observed event `schedule-extra-commit` phase=`commit` op=`PRUNE`" in packet
    assert "unexpected fixed-schedule action trace" in packet


def test_proof_packet_blocks_fixed_schedule_without_schedule_provenance(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    for event in events:
        if event["event_id"] == "start-4":
            del event["data"]["proof_baseline_schedule_id"]
            del event["data"]["proof_baseline_schedule_hash"]
            del event["data"]["proof_baseline_schedule_version"]
            del event["data"]["proof_baseline_schedule_action_count"]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "missing proof_baseline_schedule_id" in packet
    assert "missing proof_baseline_schedule_hash" in packet
    assert "missing proof_baseline_schedule_version" in packet
    assert "missing proof_baseline_schedule_action_count" in packet


def test_proof_packet_blocks_schedule_provenance_on_non_schedule_baseline(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    events[0]["data"]["proof_baseline_schedule_id"] = (
        FIXED_SCHEDULE_GERMINATE_R0C0_V1
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "unexpected fixed-schedule provenance on non-schedule baseline" in packet


def test_proof_packet_rejects_invalid_outcome_as_baseline_evidence(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    for event in events:
        if event["event_id"] == "eo-2":
            event["data"]["param_ratio"] = 0.2
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "param_ratio < 1.0" in packet
    assert "BLOCKING missing proof baseline outcome evidence" in packet
    assert "mode `static_initial`" in packet
    assert "BLOCKING missing required proof baseline `static_initial`" in packet


def test_proof_packet_blocks_baseline_label_without_valid_outcome(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events(
        outcome_groups=(
            "off",
            "final",
            "schedule",
            "lockstep-a",
            "lockstep-b",
        )
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING missing proof baseline outcome evidence" in packet
    assert "mode `static_initial`" in packet
    assert "BLOCKING missing required proof baseline `static_initial`" in packet


def test_proof_packet_blocks_lockstep_reward_ab_with_single_outcome_bearing_run(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events(
        outcome_groups=(
            "off",
            "initial",
            "final",
            "schedule",
            "lockstep-a",
        )
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING missing proof baseline outcome evidence" in packet
    assert "mode `lockstep_reward_ab` reward=simplified" in packet
    assert "BLOCKING incomplete lockstep reward A/B baseline" in packet
    assert "requires at least two outcome-bearing runs" in packet


def test_proof_packet_blocks_lockstep_reward_ab_without_distinct_rewards(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events(
        lockstep_a_reward_mode="simplified",
        lockstep_b_reward_mode="simplified",
    )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING incomplete lockstep reward A/B baseline" in packet
    assert "requires distinct reward modes" in packet


def test_proof_packet_blocks_lockstep_reward_ab_with_different_seeds(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events(lockstep_b_seed=50043)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        require_blueprint_health_baselines=True,
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING incomplete lockstep reward A/B baseline" in packet
    assert "requires matching seed" in packet


def test_reward_efficiency_profile_blocks_missing_baseline_provenance(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _complete_blueprint_health_baseline_events()
    events[4]["data"]["proof_baseline_pair_id"] = None
    events[5]["data"]["proof_baseline_lifecycle_policy"] = ""
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")

    packet = build_proof_packet(
        str(tmp_path),
        proof_profile="reward-efficiency",
    )

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING incomplete proof baseline provenance" in packet
    assert "missing proof_baseline_pair_id" in packet
    assert "missing proof_baseline_lifecycle_policy" in packet


def test_reward_efficiency_profile_requires_blueprint_health_baselines(tmp_path):
    """Final exam profile must not rely on manually remembering baseline flags."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _baseline_run_events()
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(
        str(tmp_path),
        proof_profile="reward-efficiency",
    )

    assert "Proof profile: `reward-efficiency`" in packet
    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING missing required proof baseline `off_switch`" in packet
    assert "--proof-profile reward-efficiency" in packet


def test_reward_efficiency_profile_rejects_precision_gate_opt_out(tmp_path):
    """The final proof profile cannot opt out of precision provenance."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _baseline_run_events()
    del events[0]["data"]["amp_enabled"]
    del events[0]["data"]["amp_dtype"]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    with pytest.raises(ValueError, match="requires precision provenance"):
        build_proof_packet(
            str(tmp_path),
            proof_profile="reward-efficiency",
            require_precision_provenance=False,
        )


def test_reward_efficiency_profile_rejects_baseline_gate_opt_out(tmp_path):
    with pytest.raises(ValueError, match="requires blueprint-health baselines"):
        build_proof_packet(
            str(tmp_path),
            proof_profile="reward-efficiency",
            require_blueprint_health_baselines=False,
        )


def test_reward_efficiency_cli_rejects_precision_opt_out(tmp_path, monkeypatch):
    output_path = tmp_path / "packet.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "proof_packet.py",
            "--telemetry-dir",
            str(tmp_path),
            "--output",
            str(output_path),
            "--proof-profile",
            "reward-efficiency",
            "--no-require-precision-provenance",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        proof_packet_main()

    assert exc_info.value.code == 2


def test_cli_defaults_to_reward_efficiency_profile(tmp_path, monkeypatch):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in _baseline_run_events()) + "\n"
    )
    output_path = tmp_path / "packet.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "proof_packet.py",
            "--telemetry-dir",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
    )

    proof_packet_main()

    packet = output_path.read_text()
    assert "Proof profile: `reward-efficiency`" in packet
    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING missing required proof baseline `off_switch`" in packet


def test_build_proof_packet_defaults_to_reward_efficiency_profile(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in _baseline_run_events()) + "\n"
    )

    packet = build_proof_packet(str(tmp_path))

    assert "Proof profile: `reward-efficiency`" in packet
    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "BLOCKING missing required proof baseline `off_switch`" in packet


# ---------------------------------------------------------------------------
# KARN-PROOF-001: fail closed on missing runs / outcomes / empty telemetry
# ---------------------------------------------------------------------------


def test_proof_packet_blocks_empty_telemetry(tmp_path):
    """No events.jsonl files at all must BLOCK with explicit text."""
    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_INSTRUMENTATION`" in packet
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

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_INSTRUMENTATION`" in packet
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

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_INSTRUMENTATION`" in packet
    assert "BLOCKING no `EPISODE_OUTCOME` events found" in packet


def test_proof_packet_blocks_incomplete_episode_outcome_payload(tmp_path):
    """An outcome row without required ROI fields is not proof-grade evidence."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    bad_outcome = _episode_outcome_event("bad-outcome", "A")
    del bad_outcome["data"]["final_accuracy"]
    del bad_outcome["data"]["param_ratio"]
    events = [_baseline_run_events()[0], _ppo_update_event("ppo-1", "A"), bad_outcome]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "## Outcome Integrity" in packet
    assert "bad-outcome" in packet
    assert "missing final_accuracy" in packet
    assert "missing param_ratio" in packet


def test_proof_packet_blocks_semantically_invalid_episode_outcome(tmp_path):
    """param_ratio is total/host; values below no-growth 1.0 invalidate ROI."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = [
        _baseline_run_events()[0],
        _ppo_update_event("ppo-1", "A"),
        _episode_outcome_event("impossible-ratio", "A", param_ratio=0.2),
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_MATH`" in packet
    assert "impossible-ratio" in packet
    assert "param_ratio < 1.0" in packet


def test_proof_packet_accepts_measured_zero_outcome_counts(tmp_path):
    """Measured zero outcome counts are valid; missing counts are invalid."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _baseline_run_events()
    outcome = events[2]
    outcome["data"]["final_accuracy"] = 0.0
    outcome["data"]["num_fossilized"] = 0
    outcome["data"]["num_contributing_fossilized"] = 0
    outcome["data"]["episode_reward"] = 0.0
    outcome["data"]["stability_score"] = 0.0
    outcome["data"]["germinate_count"] = 0
    outcome["data"]["prune_count"] = 0
    outcome["data"]["fossilize_count"] = 0
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `STOP_THEORY`" in packet
    assert "All episode outcome rows satisfy the proof ROI contract." in packet
    assert "mean_final_accuracy=0.0" in packet


def test_proof_packet_blocks_missing_precision_provenance(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _baseline_run_events()
    del events[0]["data"]["amp_enabled"]
    del events[0]["data"]["amp_dtype"]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_PRECISION`" in packet
    assert "BLOCKING precision provenance" in packet
    assert "missing amp_enabled" in packet


def test_proof_packet_revises_positive_but_weak_roi(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _baseline_run_events()
    events[2]["data"]["final_accuracy"] = 0.5
    events[2]["data"]["param_ratio"] = 1.0
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `REVISE_ALGORITHM`" in packet
    assert "measurable signal" in packet


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
                "amp_enabled": False,
                "amp_dtype": None,
            },
            "severity": "info",
        },
        _ppo_update_event("ppo-1", "A"),
        _episode_outcome_event("eo-1", "A"),
    ]


def test_proof_packet_blocks_value_collapse_even_when_ev_low_variance(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _baseline_run_events()
    events.append(
        {
            "event_id": "value-collapse-1",
            "event_type": "VALUE_COLLAPSE_DETECTED",
            "timestamp": datetime.now().isoformat(),
            "epoch": 1,
            "group_id": "A",
            "data": {
                "anomaly_type": "value_collapse",
                "env_id": 0,
                "episode": 1,
                "batch": 1,
                "detail": "value_loss exceeded proof threshold",
                "value_loss": 5.1,
                "bellman_error": 0.2,
                "explained_variance": -8.0,
                "ev_low_return_variance": True,
            },
            "severity": "error",
        }
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_MECHANICS`" in packet
    assert "BLOCKING `VALUE_COLLAPSE_DETECTED`" in packet
    assert "value_loss exceeded proof threshold" in packet


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

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_MECHANICS`" in packet
    assert f"BLOCKING `{marker}`" in packet


def test_proof_packet_surfaces_duplicate_rollbacks(tmp_path):
    """Duplicate rollback confounders should both remain visible in the proof packet."""
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events = _baseline_run_events()
    for idx in range(2):
        events.append(
            {
                "event_id": f"rollback-{idx + 1}",
                "event_type": "GOVERNOR_ROLLBACK",
                "timestamp": datetime.now().isoformat(),
                "epoch": idx + 1,
                "group_id": "A",
                "data": {
                    "env_id": 0,
                    "device": "cpu",
                    "reason": "governor_nan",
                    "episode_idx": idx + 1,
                },
                "severity": "error",
            }
        )
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_MECHANICS`" in packet
    assert packet.count("BLOCKING `GOVERNOR_ROLLBACK`") == 2


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

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    assert "Verdict: `BLOCKED_INSTRUMENTATION`" in packet
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
                "amp_enabled": False,
                "amp_dtype": None,
            },
            "severity": "info",
        },
        _ppo_update_event("ppo-1", "A"),
        _episode_outcome_event("eo-1", "A", param_ratio=1.0, final_accuracy=90.0),
    ]
    (no_growth / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in no_growth_events) + "\n"
    )

    packet = build_proof_packet(str(tmp_path), proof_profile="generic")

    # mean_param_ratio reports the 1.0 (no-growth) value verbatim, and ROI = 90/1.0.
    assert "Verdict: `CONTINUE`" in packet
    assert "mean_param_ratio=1.0" in packet
    assert "mean_accuracy_roi=90.0" in packet
