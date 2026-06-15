"""Tests for static-final topology manifest replay primitives."""

from __future__ import annotations

from typing import cast

import pytest

from esper.kasmina.host import MorphogeneticModel
from esper.kasmina.slot import SeedSlot
from esper.leyline import AlphaAlgorithm, SeedStage
from esper.leyline.telemetry import TelemetryEvent, TelemetryEventType
from esper.simic.training.static_final_replay import (
    STATE_DICT_EXACT_REPLAY_WEIGHT_POLICY,
    TOPOLOGY_ONLY_REPLAY_WEIGHT_POLICY,
    capture_source_final_manifest,
    capture_static_final_replay_manifest,
    materialize_static_final_topology,
)
from esper.simic.training.vectorized_trainer import VectorizedPPOTrainer
from esper.tolaria.environment import create_model


class _TaskSpec:
    name = "cifar_baseline"


class _Governor:
    def __init__(self) -> None:
        self.snapshot_count = 0

    def snapshot(self) -> None:
        self.snapshot_count += 1


class _EnvState:
    def __init__(self, model: MorphogeneticModel, governor: _Governor) -> None:
        self.model = model
        self.governor = governor


class _Emitter:
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []
        self.episode_idx: int | None = 0

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


def _model_with_fossilized_seed(
    alpha_algorithm: AlphaAlgorithm = AlphaAlgorithm.ADD,
) -> MorphogeneticModel:
    model = create_model(task="cifar_baseline", device="cpu", slots=["r0c1"])
    assert isinstance(model, MorphogeneticModel)
    model.germinate_seed(
        "conv_small",
        "seed-final",
        slot="r0c1",
        alpha_algorithm=alpha_algorithm,
    )
    slot = cast(SeedSlot, model.seed_slots["r0c1"])
    assert slot.state is not None
    slot.state.stage = SeedStage.FOSSILIZED
    slot.state.alpha = 1.0
    slot.state.alpha_controller.alpha = 1.0
    slot.set_alpha(1.0)
    return model


def test_capture_source_final_manifest_requires_fossilized_seed() -> None:
    empty_model = create_model(task="cifar_baseline", device="cpu", slots=["r0c1"])

    with pytest.raises(ValueError, match="at least one fossilized seed"):
        capture_source_final_manifest(
            model=empty_model,
            task="cifar_baseline",
            proof_baseline_pair_id="blueprint-health-proof",
        )

    training_model = create_model(task="cifar_baseline", device="cpu", slots=["r0c1"])
    training_model.germinate_seed("conv_small", "seed-training", slot="r0c1")

    with pytest.raises(ValueError, match="fossilized seeds only"):
        capture_source_final_manifest(
            model=training_model,
            task="cifar_baseline",
            proof_baseline_pair_id="blueprint-health-proof",
        )


def test_materialize_static_final_topology_round_trips_manifest_hash() -> None:
    source_model = _model_with_fossilized_seed(alpha_algorithm=AlphaAlgorithm.MULTIPLY)
    source_manifest = capture_source_final_manifest(
        model=source_model,
        task="cifar_baseline",
        proof_baseline_pair_id="blueprint-health-proof",
    )

    replay_model = create_model(task="cifar_baseline", device="cpu", slots=["r0c1"])
    materialize_static_final_topology(
        model=replay_model,
        source_manifest_json=source_manifest.topology_manifest_json,
    )
    replay_slot = cast(SeedSlot, replay_model.seed_slots["r0c1"])
    assert replay_slot.state is not None
    assert replay_slot.state.stage == SeedStage.FOSSILIZED
    assert replay_slot.state.blueprint_id == "conv_small"
    assert replay_slot.state.alpha_algorithm == AlphaAlgorithm.MULTIPLY
    assert replay_slot.alpha == 1.0

    replay_manifest = capture_static_final_replay_manifest(
        model=replay_model,
        task="cifar_baseline",
        proof_baseline_pair_id="blueprint-health-proof",
        source_manifest=source_manifest,
        source_run_dir="source_run",
        source_group_id="dynamic",
        source_episode_idx=0,
        source_event_id="source-final-manifest",
        replay_weight_policy=TOPOLOGY_ONLY_REPLAY_WEIGHT_POLICY,
        replay_env_id=0,
        replay_episode_idx=0,
    )

    assert replay_manifest.manifest_role == "static_final_replay"
    assert replay_manifest.source_topology_manifest_hash == source_manifest.topology_manifest_hash
    assert replay_manifest.replayed_topology_manifest_hash == source_manifest.topology_manifest_hash
    assert replay_manifest.manifest_match is True
    assert replay_manifest.replay_env_id == 0
    assert replay_manifest.replay_episode_idx == 0
    assert replay_manifest.fossilized_seed_count == 1
    assert replay_manifest.topology_delta_count == 1


def test_capture_static_final_replay_manifest_rejects_state_dict_exact_without_state_evidence() -> None:
    source_model = _model_with_fossilized_seed()
    source_manifest = capture_source_final_manifest(
        model=source_model,
        task="cifar_baseline",
        proof_baseline_pair_id="blueprint-health-proof",
    )
    replay_model = create_model(task="cifar_baseline", device="cpu", slots=["r0c1"])
    materialize_static_final_topology(
        model=replay_model,
        source_manifest_json=source_manifest.topology_manifest_json,
    )

    with pytest.raises(ValueError, match="state-dict hash evidence"):
        capture_static_final_replay_manifest(
            model=replay_model,
            task="cifar_baseline",
            proof_baseline_pair_id="blueprint-health-proof",
            source_manifest=source_manifest,
            source_run_dir="source_run",
            source_group_id="dynamic",
            source_episode_idx=0,
            source_event_id="source-final-manifest",
            replay_weight_policy=STATE_DICT_EXACT_REPLAY_WEIGHT_POLICY,
            replay_env_id=0,
            replay_episode_idx=0,
        )


def test_materialize_static_final_topology_refuses_occupied_target_slot() -> None:
    source_model = _model_with_fossilized_seed()
    source_manifest = capture_source_final_manifest(
        model=source_model,
        task="cifar_baseline",
        proof_baseline_pair_id="blueprint-health-proof",
    )
    occupied_model = create_model(task="cifar_baseline", device="cpu", slots=["r0c1"])
    occupied_model.germinate_seed("conv_small", "already-active", slot="r0c1")

    with pytest.raises(RuntimeError, match="requires a dormant target slot"):
        materialize_static_final_topology(
            model=occupied_model,
            source_manifest_json=source_manifest.topology_manifest_json,
        )


def test_capture_static_final_replay_manifest_rejects_unknown_weight_policy() -> None:
    source_model = _model_with_fossilized_seed()
    source_manifest = capture_source_final_manifest(
        model=source_model,
        task="cifar_baseline",
        proof_baseline_pair_id="blueprint-health-proof",
    )
    replay_model = create_model(task="cifar_baseline", device="cpu", slots=["r0c1"])
    materialize_static_final_topology(
        model=replay_model,
        source_manifest_json=source_manifest.topology_manifest_json,
    )

    with pytest.raises(ValueError, match="Unknown static-final replay_weight_policy"):
        capture_static_final_replay_manifest(
            model=replay_model,
            task="cifar_baseline",
            proof_baseline_pair_id="blueprint-health-proof",
            source_manifest=source_manifest,
            source_run_dir="source_run",
            source_group_id="dynamic",
            source_episode_idx=0,
            source_event_id="source-final-manifest",
            replay_weight_policy="weights_maybe",
            replay_env_id=0,
            replay_episode_idx=0,
        )

    replay_manifest = capture_static_final_replay_manifest(
        model=replay_model,
        task="cifar_baseline",
        proof_baseline_pair_id="blueprint-health-proof",
        source_manifest=source_manifest,
        source_run_dir="source_run",
        source_group_id="dynamic",
        source_episode_idx=0,
        source_event_id="source-final-manifest",
        replay_weight_policy=TOPOLOGY_ONLY_REPLAY_WEIGHT_POLICY,
        replay_env_id=0,
        replay_episode_idx=0,
    )
    assert replay_manifest.replay_weight_policy == TOPOLOGY_ONLY_REPLAY_WEIGHT_POLICY


def test_vectorized_trainer_materializes_static_final_replay_evidence() -> None:
    source_model = _model_with_fossilized_seed()
    source_manifest = capture_source_final_manifest(
        model=source_model,
        task="cifar_baseline",
        proof_baseline_pair_id="blueprint-health-proof",
    )
    replay_model = create_model(task="cifar_baseline", device="cpu", slots=["r0c1"])
    governor = _Governor()
    env_state = _EnvState(model=replay_model, governor=governor)
    emitter = _Emitter()
    trainer = object.__new__(VectorizedPPOTrainer)
    trainer.task_spec = _TaskSpec()
    trainer.static_final_source_manifest = source_manifest
    trainer.static_final_source_run_dir = "source_run"
    trainer.static_final_source_group_id = "dynamic"
    trainer.static_final_source_episode_idx = 0
    trainer.static_final_source_event_id = "source-final-manifest"

    VectorizedPPOTrainer._materialize_static_final_replay(
        trainer,
        env_states=[env_state],
        emitters=[emitter],
    )

    replay_slot = cast(SeedSlot, replay_model.seed_slots["r0c1"])
    assert replay_slot.state is not None
    assert replay_slot.state.stage == SeedStage.FOSSILIZED
    assert governor.snapshot_count == 1
    assert len(emitter.events) == 1
    replay_event = emitter.events[0]
    assert replay_event.event_type == TelemetryEventType.TOPOLOGY_MANIFEST_RECORDED
    replay_payload = replay_event.data
    assert replay_payload.manifest_role == "static_final_replay"
    assert replay_payload.replay_weight_policy == TOPOLOGY_ONLY_REPLAY_WEIGHT_POLICY
    assert replay_payload.manifest_match is True
    assert replay_payload.replay_env_id == 0
    assert replay_payload.replay_episode_idx == 0
    assert replay_payload.source_topology_manifest_hash == (
        source_manifest.topology_manifest_hash
    )


def test_vectorized_trainer_emits_source_final_manifest_evidence() -> None:
    source_model = _model_with_fossilized_seed()
    governor = _Governor()
    env_state = _EnvState(model=source_model, governor=governor)
    emitter = _Emitter()
    emitter.group_id = "static_final_source"
    emitter.episode_idx = 7
    trainer = object.__new__(VectorizedPPOTrainer)
    trainer.task_spec = _TaskSpec()
    trainer.proof_baseline_pair_id = "blueprint-health-proof"
    trainer.telemetry_run_dir = "source_run"

    records = VectorizedPPOTrainer._emit_source_final_manifests(
        trainer,
        env_states=[env_state],
        emitters=[emitter],
        epoch=25,
    )

    assert len(emitter.events) == 1
    source_event = emitter.events[0]
    assert source_event.event_type == TelemetryEventType.TOPOLOGY_MANIFEST_RECORDED
    source_payload = source_event.data
    assert source_payload.manifest_role == "source_final"
    assert source_payload.proof_baseline_pair_id == "blueprint-health-proof"
    assert source_payload.fossilized_seed_count == 1
    assert source_payload.topology_delta_count == 1
    assert records == [
        {
            "event_id": source_event.event_id,
            "run_dir": "source_run",
            "group_id": "static_final_source",
            "episode_idx": 7,
            "payload": {
                "manifest_role": "source_final",
                "proof_baseline_pair_id": "blueprint-health-proof",
                "topology_manifest_version": source_payload.topology_manifest_version,
                "topology_manifest_hash": source_payload.topology_manifest_hash,
                "topology_manifest_json": source_payload.topology_manifest_json,
                "task": source_payload.task,
                "host_topology": source_payload.host_topology,
                "slot_config_hash": source_payload.slot_config_hash,
                "slot_count": source_payload.slot_count,
                "fossilized_seed_count": source_payload.fossilized_seed_count,
                "topology_delta_count": source_payload.topology_delta_count,
                "source_run_dir": None,
                "source_group_id": None,
                "source_episode_idx": None,
                "source_event_id": None,
                "source_topology_manifest_hash": None,
                "replay_weight_policy": None,
                "replay_env_id": None,
                "replay_episode_idx": None,
                "replayed_topology_manifest_hash": None,
                "manifest_match": None,
            },
        }
    ]
