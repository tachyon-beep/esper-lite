"""Gradient-health measurement must be independent of use_telemetry.

Regression test for esper-lite-4fe98055f7: gradient stats are the G2 blending
gate's input (control-path state, KTS-001), but their collection and their
sync into seed state were both gated on ``use_telemetry``. With telemetry off,
gradient health stayed UNMEASURED forever, permissive G2 (correctly) denied
every seed, and the run silently lost ALL blending/fossilization — an
observability flag changing training behavior.

The contract locked here: a telemetry-OFF run still measures gradient health
and syncs it into seed state (emission stays telemetry-gated; measurement does
not). Uses the production FIXED_SCHEDULE proof-baseline lifecycle to germinate
a seed deterministically with the real agent.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from esper.kasmina.slot import SeedState
from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
    FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
    FIXED_SCHEDULE_GERMINATE_R0C0_V1,
    FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
    ProofBaselineMode,
)
from esper.simic.training.config import TrainingConfig
from esper.simic.training.vectorized import train_ppo_vectorized


@pytest.mark.integration
def test_seed_gradient_health_measured_with_telemetry_off(
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

    # Spy on the sync seam: gradient measurement reaches seed state ONLY via
    # sync_telemetry(gradient_health=...); the fallback path passes no gradient
    # kwargs and leaves the seed UNMEASURED (KTS-001 denies it at G2).
    sync_calls: list[dict[str, Any]] = []
    original_sync = SeedState.sync_telemetry

    def spying_sync(self: SeedState, *args: Any, **kwargs: Any) -> None:
        sync_calls.append(
            {
                "gradient_health": kwargs.get("gradient_health"),
                "stage": self.stage.name,
                "gradient_measured_after": None,
            }
        )
        original_sync(self, *args, **kwargs)
        telemetry = self.telemetry
        if telemetry is not None:
            sync_calls[-1]["gradient_measured_after"] = telemetry.gradient_measured

    monkeypatch.setattr(SeedState, "sync_telemetry", spying_sync)

    config = TrainingConfig.for_cifar_minimal()
    config.n_envs = 1
    config.n_episodes = 1
    config.max_epochs = 6
    config.chunk_length = 6
    config.seed = 42
    config.use_telemetry = False
    config.gradient_telemetry_stride = 1
    # Match the production regime (every run config in this experiment line
    # sets auto_forward_g1): the germinated seed must reach TRAINING, the
    # stage whose gradients the collector measures and G2 consumes.
    config.auto_forward_g1 = True

    train_ppo_vectorized(
        **config.to_train_kwargs(),
        device="cpu",
        devices=["cpu"],
        num_workers=0,
        quiet_analytics=True,
        # Production scripted lifecycle: germinate one NORM seed in the first
        # enabled slot at epoch 1, WAIT afterwards (leyline fixed schedule).
        proof_baseline_mode=ProofBaselineMode.FIXED_SCHEDULE.value,
        proof_baseline_lifecycle_policy="apply_declared_lifecycle_schedule",
        proof_baseline_schedule_id=FIXED_SCHEDULE_GERMINATE_R0C0_V1,
        proof_baseline_schedule_hash=FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
        proof_baseline_schedule_version=FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
        proof_baseline_schedule_action_count=(
            FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT
        ),
    )

    # The seed lived (fallback or measured sync fired at least once).
    assert sync_calls, "expected an active seed whose telemetry was synced"

    measured = [c for c in sync_calls if c["gradient_health"] is not None]
    stages_seen = sorted({c["stage"] for c in sync_calls})
    assert measured, (
        "gradient health never reached seed state in a use_telemetry=False run "
        f"(stages seen at sync: {stages_seen}): gradient measurement is G2 gate "
        "input (KTS-001), not observability, and must be collected regardless "
        "of telemetry (esper-lite-4fe98055f7)"
    )
    assert any(c["gradient_measured_after"] for c in measured), (
        "sync_telemetry received gradient_health but gradient_measured never "
        "became True on the seed state"
    )
