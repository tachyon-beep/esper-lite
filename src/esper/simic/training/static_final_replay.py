"""Static-final topology manifest capture and replay primitives."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from esper.kasmina.host import MorphogeneticModel
from esper.kasmina.slot import SeedSlot
from esper.leyline import AlphaAlgorithm, SeedStage
from esper.leyline.telemetry import TopologyManifestPayload


TOPOLOGY_MANIFEST_VERSION = 1
TOPOLOGY_ONLY_REPLAY_WEIGHT_POLICY = "topology_only"
STATE_DICT_EXACT_REPLAY_WEIGHT_POLICY = "state_dict_exact"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _hash_payload(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def _slot_config_hash(slot_ids: tuple[str, ...]) -> str:
    return _hash_payload(_canonical_json({"slot_ids": slot_ids}))


def _manifest_body(
    *,
    model: MorphogeneticModel,
    task: str,
) -> dict[str, Any]:
    host_topology = model.host.topology
    slot_ids = tuple(model.seed_slots.keys())
    slots: list[dict[str, Any]] = []
    for slot_id in slot_ids:
        slot = model.seed_slots[slot_id]
        assert isinstance(slot, SeedSlot)
        state = slot.state
        if state is None:
            continue
        if state.stage != SeedStage.FOSSILIZED:
            raise ValueError(
                "Static-final topology manifests require fossilized seeds only; "
                f"slot_id={slot_id} has stage={state.stage.name}."
            )
        slots.append(
            {
                "slot_id": slot_id,
                "seed_id": state.seed_id,
                "blueprint_id": state.blueprint_id,
                "stage": state.stage.name,
                "alpha": state.alpha,
                "alpha_algorithm": int(state.alpha_algorithm),
                "blend_tempo_epochs": state.blend_tempo_epochs,
                "seed_params": slot.active_seed_params,
            }
        )

    fossilized_seed_count = len(slots)
    if fossilized_seed_count == 0:
        raise ValueError("Static-final topology manifest requires at least one fossilized seed.")

    return {
        "topology_manifest_version": TOPOLOGY_MANIFEST_VERSION,
        "task": task,
        "host_topology": host_topology,
        "slot_config_hash": _slot_config_hash(slot_ids),
        "slot_count": len(slot_ids),
        "fossilized_seed_count": fossilized_seed_count,
        "topology_delta_count": fossilized_seed_count,
        "slots": slots,
    }


def capture_source_final_manifest(
    *,
    model: MorphogeneticModel,
    task: str,
    proof_baseline_pair_id: str,
) -> TopologyManifestPayload:
    """Capture a source-final topology manifest from a fully evolved model."""
    body = _manifest_body(model=model, task=task)
    manifest_json = _canonical_json(body)
    manifest_hash = _hash_payload(manifest_json)
    return TopologyManifestPayload(
        manifest_role="source_final",
        proof_baseline_pair_id=proof_baseline_pair_id,
        topology_manifest_version=TOPOLOGY_MANIFEST_VERSION,
        topology_manifest_hash=manifest_hash,
        topology_manifest_json=manifest_json,
        task=task,
        host_topology=body["host_topology"],
        slot_config_hash=body["slot_config_hash"],
        slot_count=body["slot_count"],
        fossilized_seed_count=body["fossilized_seed_count"],
        topology_delta_count=body["topology_delta_count"],
    )


def materialize_static_final_topology(
    *,
    model: MorphogeneticModel,
    source_manifest_json: str,
) -> None:
    """Materialize the source-final topology into a fresh model."""
    manifest = json.loads(source_manifest_json)
    slots = manifest["slots"]
    if not isinstance(slots, list):
        raise ValueError("Static-final topology manifest field `slots` must be a list.")

    for slot_spec in slots:
        if not isinstance(slot_spec, dict):
            raise ValueError("Static-final topology manifest slot entries must be objects.")
        slot_id = slot_spec["slot_id"]
        slot = model.seed_slots[slot_id]
        assert isinstance(slot, SeedSlot)
        if slot.state is not None or slot.seed is not None:
            raise RuntimeError(
                f"Static-final replay requires a dormant target slot; slot_id={slot_id} is occupied."
            )
        stage = SeedStage[slot_spec["stage"]]
        if stage != SeedStage.FOSSILIZED:
            raise ValueError(
                "Static-final replay only materializes fossilized source slots; "
                f"slot_id={slot_id} has stage={stage.name}."
            )
        model.germinate_seed(
            slot_spec["blueprint_id"],
            slot_spec["seed_id"],
            slot=slot_id,
            blend_tempo_epochs=slot_spec["blend_tempo_epochs"],
            alpha_target=slot_spec["alpha"],
            alpha_algorithm=AlphaAlgorithm(int(slot_spec["alpha_algorithm"])),
        )
        if slot.state is None:
            raise RuntimeError(f"Static-final replay failed to materialize slot_id={slot_id}.")
        slot.state.stage = SeedStage.FOSSILIZED
        slot.state.alpha = slot_spec["alpha"]
        slot.state.alpha_controller.alpha = slot_spec["alpha"]
        slot.set_alpha(slot_spec["alpha"])


def capture_static_final_replay_manifest(
    *,
    model: MorphogeneticModel,
    task: str,
    proof_baseline_pair_id: str,
    source_manifest: TopologyManifestPayload,
    source_run_dir: str,
    source_group_id: str,
    source_episode_idx: int,
    source_event_id: str,
    replay_weight_policy: str,
    replay_env_id: int,
    replay_episode_idx: int,
) -> TopologyManifestPayload:
    """Capture replay evidence after materializing a source-final topology."""
    if replay_weight_policy == STATE_DICT_EXACT_REPLAY_WEIGHT_POLICY:
        raise ValueError(
            "Static-final replay_weight_policy `state_dict_exact` requires source "
            "and replay state-dict hash evidence."
        )
    if replay_weight_policy != TOPOLOGY_ONLY_REPLAY_WEIGHT_POLICY:
        raise ValueError(f"Unknown static-final replay_weight_policy: {replay_weight_policy}")

    body = _manifest_body(model=model, task=task)
    manifest_json = _canonical_json(body)
    manifest_hash = _hash_payload(manifest_json)
    return TopologyManifestPayload(
        manifest_role="static_final_replay",
        proof_baseline_pair_id=proof_baseline_pair_id,
        topology_manifest_version=TOPOLOGY_MANIFEST_VERSION,
        topology_manifest_hash=manifest_hash,
        topology_manifest_json=manifest_json,
        task=task,
        host_topology=body["host_topology"],
        slot_config_hash=body["slot_config_hash"],
        slot_count=body["slot_count"],
        fossilized_seed_count=body["fossilized_seed_count"],
        topology_delta_count=body["topology_delta_count"],
        source_run_dir=source_run_dir,
        source_group_id=source_group_id,
        source_episode_idx=source_episode_idx,
        source_event_id=source_event_id,
        source_topology_manifest_hash=source_manifest.topology_manifest_hash,
        replay_weight_policy=replay_weight_policy,
        replay_env_id=replay_env_id,
        replay_episode_idx=replay_episode_idx,
        replayed_topology_manifest_hash=manifest_hash,
        manifest_match=manifest_hash == source_manifest.topology_manifest_hash,
    )
