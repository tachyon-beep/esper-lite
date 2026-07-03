"""PIN-E placebo schedule lifecycle — the go/no-go gate before any GPU run.

Proves, on CPU with mock data, the full mechanism chain the noise-floor
harness depends on (esper-lite-94869250f1):

1. The declared 3-slot placebo schedule force-germinates the PLACEBO blueprint
   (enum member + mask extra_blueprints union) in all three slots.
2. With seed_lr=0 the seeds still pass permissive G2 (gradient measured,
   health >= 0.7) and G3 on the exact scheduled epochs, and their weights
   never move (the frozen-delta property tau depends on).
3. No placebo ever fossilizes (fossilized slots are ablation-invisible at
   shapley_synergy_scale=0, which would silence the measurement).
4. The fused val pass measures the placebo LOO once alpha > 0, and builds the
   all_off + pair configs once >= 2/3 seeds are active — the 2^3 factorial the
   offline phi assembly needs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from esper.kasmina.slot import SeedState
from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_ACTION_COUNT,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_VERSION,
    ProofBaselineMode,
)
from esper.simic.training.config import TrainingConfig
from esper.simic.training.vectorized import train_ppo_vectorized
from esper.simic.training.vectorized_trainer import VectorizedPPOTrainer

_SLOTS = ("r0c0", "r0c1", "r0c2")


@pytest.mark.integration
def test_placebo_schedule_reaches_holding_and_never_fossilizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import esper.runtime as runtime

    original_get_task_spec = runtime.get_task_spec

    def get_mock_task_spec(name: str) -> Any:
        spec = original_get_task_spec(name)
        return replace(
            spec,
            dataloader_defaults={**spec.dataloader_defaults, "mock": True},
        )

    monkeypatch.setattr(runtime, "get_task_spec", get_mock_task_spec)

    # Gradient-health spy (template: test_gradient_measurement_telemetry_independent).
    health_records: list[tuple[str, float]] = []
    original_sync = SeedState.sync_telemetry

    def spying_sync(self: SeedState, *args: Any, **kwargs: Any) -> None:
        health = kwargs.get("gradient_health")
        if health is not None:
            health_records.append((self.stage.name, float(health)))
        original_sync(self, *args, **kwargs)

    monkeypatch.setattr(SeedState, "sync_telemetry", spying_sync)

    # Per-epoch probe: wrap the fused val pass (runs once per epoch, BEFORE the
    # scheduled action executes, so probe(E) reflects actions through E-1).
    probes: dict[int, dict[str, Any]] = {}
    weight_snapshots: dict[int, torch.Tensor] = {}
    original_fused = VectorizedPPOTrainer._run_fused_val_pass

    def probing_fused(self: VectorizedPPOTrainer, **kwargs: Any):
        result = original_fused(self, **kwargs)
        epoch = kwargs["epoch"]
        env_state = kwargs["env_states"][0]
        slot_view: dict[str, Any] = {}
        for slot_id in _SLOTS:
            slot = env_state.model.seed_slots[slot_id]
            state = slot.state
            if state is None:
                slot_view[slot_id] = None
                continue
            slot_view[slot_id] = {
                "stage": state.stage.name,
                "alpha": float(slot.alpha),
                "blueprint_id": state.blueprint_id,
                "cf_contribution": state.metrics.counterfactual_contribution,
            }
        fused_result = result[0]
        probes[epoch] = {
            "slots": slot_view,
            "has_all_off": 0 in fused_result.all_disabled_accs,
            "n_pairs": len(fused_result.pair_accs.get(0, {})),
            "optimizer_lrs": [
                group["lr"]
                for opt in env_state.seed_optimizers.values()
                for group in opt.param_groups
            ],
        }
        r0c0 = env_state.model.seed_slots["r0c0"]
        if r0c0.state is not None and r0c0.seed is not None:
            weight_snapshots[epoch] = (
                next(r0c0.seed.parameters()).detach().clone()
            )
        return result

    monkeypatch.setattr(VectorizedPPOTrainer, "_run_fused_val_pass", probing_fused)

    config = TrainingConfig.for_cifar_minimal()
    config.task = "cifar_minimal"
    config.slots = list(_SLOTS)
    config.max_seeds = 3
    config.n_envs = 1
    config.n_episodes = 1
    config.max_epochs = 54
    config.chunk_length = 54
    config.seed = 42
    config.use_telemetry = False
    config.gradient_telemetry_stride = 1

    train_ppo_vectorized(
        **config.to_train_kwargs(),
        device="cpu",
        devices=["cpu"],
        num_workers=0,
        quiet_analytics=True,
        seed_lr_override=0.0,
        proof_baseline_mode=ProofBaselineMode.FIXED_SCHEDULE.value,
        proof_baseline_lifecycle_policy="apply_declared_lifecycle_schedule",
        proof_baseline_schedule_id=FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
        proof_baseline_schedule_hash=FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH,
        proof_baseline_schedule_version=FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_VERSION,
        proof_baseline_schedule_action_count=(
            FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_ACTION_COUNT
        ),
    )

    def stage(epoch: int, slot_id: str) -> str:
        view = probes[epoch]["slots"][slot_id]
        assert view is not None, f"no seed in {slot_id} at epoch {epoch}"
        return view["stage"]

    # 1. Forced germination delivered the PLACEBO blueprint in every slot
    #    (serial: each germinate is checked at its first post-action probe).
    for probe_epoch, slot_id in ((2, "r0c0"), (19, "r0c1"), (36, "r0c2")):
        view = probes[probe_epoch]["slots"][slot_id]
        assert view is not None and view["blueprint_id"] == "placebo"

    # 2. Exact-epoch SERIAL lifecycle (probe(E) reflects actions through E-1;
    #    D3 masks GERMINATE while any seed is GERMINATED/TRAINING, so each
    #    placebo reaches HOLDING before the next germinates; BLENDING ramps
    #    alpha over 5 STANDARD-tempo epochs — alpha_speed does NOT skip it):
    #    s0 germ@1 train@2 blend@12 hold@17; s1 18/19/29/34; s2 35/36/46/51.
    assert stage(2, "r0c0") == "GERMINATED"
    assert stage(3, "r0c0") == "TRAINING"
    assert stage(12, "r0c0") == "TRAINING"   # dwell held through epoch 11
    assert stage(13, "r0c0") == "BLENDING"   # G2 passed AT epoch 12
    assert 0.0 < probes[13]["slots"]["r0c0"]["alpha"] < 1.0  # mid-ramp
    assert stage(17, "r0c0") == "BLENDING"   # ramp window 12..16
    assert stage(18, "r0c0") == "HOLDING"    # G3 passed AT epoch 17
    assert probes[18]["slots"]["r0c0"]["alpha"] >= 0.99  # full amplitude
    assert stage(19, "r0c1") == "GERMINATED"
    assert stage(29, "r0c1") == "TRAINING"
    assert stage(30, "r0c1") == "BLENDING"
    assert stage(35, "r0c1") == "HOLDING"
    assert stage(36, "r0c2") == "GERMINATED"
    assert stage(46, "r0c2") == "TRAINING"
    assert stage(47, "r0c2") == "BLENDING"
    assert stage(52, "r0c2") == "HOLDING"
    for slot_id in _SLOTS:
        assert stage(54, slot_id) == "HOLDING"  # all co-resident, held
        assert probes[54]["slots"][slot_id]["alpha"] >= 0.99

    # 3. NEVER fossilized, anywhere, ever.
    for epoch, probe in probes.items():
        for slot_id, view in probe["slots"].items():
            if view is not None:
                assert view["stage"] != "FOSSILIZED", (
                    f"{slot_id} fossilized at epoch {epoch}: the placebo would "
                    "become ablation-invisible at shapley_synergy_scale=0"
                )

    # 4. G2's inputs: gradient health measured with margin during the dwell.
    assert health_records, "gradient health never measured"
    assert min(h for _, h in health_records) >= 0.7

    # 5. seed_lr=0 on the REAL run-path optimizers (WI-4 reviewer requirement)
    #    and the placebo delta never moved across the entire run.
    late = probes[52]["optimizer_lrs"]
    assert late and all(lr == 0.0 for lr in late)
    assert torch.equal(weight_snapshots[3], weight_snapshots[54])

    # 6. LOO measured once alpha > 0; the 2^3 factorial configs appear once
    #    2 (all_off) and 3 (pairs) placebos are active.
    assert probes[13]["slots"]["r0c0"]["cf_contribution"] is not None
    assert probes[31]["has_all_off"], "all_off (v(empty)) config missing at k=2"
    assert probes[52]["has_all_off"]
    assert probes[52]["n_pairs"] == 3, "expected all 3 pair configs at k=3"
