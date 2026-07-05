"""Phase 0: the per-env, per-epoch residency sweep (PIN A applied across all on-path seeds).

``accumulate_residency_for_env`` folds one epoch of per-slot state into the per-seed
``SeedResidencyAccumulator`` map an ``EnvState`` holds. It computes the PER-SEED LOO marginal
``seed_contribution = val_acc − baseline_accs[slot]`` (PIN A) for each slot, with ``None`` when
that slot's LOO was not measured this epoch (the first-BLENDING-step gap), and delegates the
on-output-path gate (stage ≥ BLENDING, α>0) to the accumulator. Keyed by ``seed_id`` so a
germinate→prune→re-germinate slot attributes residency to the correct seed instance.
"""

from __future__ import annotations

import pytest

from esper.simic.rewards.residency import (
    SeedResidencyAccumulator,
    SlotResidencySample,
    accumulate_residency_for_env,
)
from esper.simic.rewards.types import STAGE_BLENDING, STAGE_TRAINING


def test_sweep_applies_per_seed_loo_marginal() -> None:
    """An on-path seed with a measured LOO accrues (val_acc − baseline)·alpha."""
    accs: dict[str, SeedResidencyAccumulator] = {}
    samples = [
        SlotResidencySample(slot_id="r0c0", seed_id="sA", stage=STAGE_BLENDING, alpha=0.5, params=1000),
    ]
    accumulate_residency_for_env(
        accs, val_acc=50.0, baseline_accs_env={"r0c0": 46.0}, slot_samples=samples
    )
    # seed_contribution = 50.0 - 46.0 = 4.0; integrand = 4.0 * 0.5 = 2.0
    assert accs["sA"].cf_weighted_integral == pytest.approx(2.0)
    assert accs["sA"].n_on_path_steps == 1
    assert accs["sA"].n_none_steps == 0


def test_sweep_none_loo_when_slot_absent_from_baseline() -> None:
    """An on-path seed whose LOO was NOT measured this epoch contributes 0 but is counted
    as a None step (the None-fraction the readout reports)."""
    accs: dict[str, SeedResidencyAccumulator] = {}
    samples = [
        SlotResidencySample(slot_id="r0c0", seed_id="sA", stage=STAGE_BLENDING, alpha=0.5, params=1000),
    ]
    accumulate_residency_for_env(
        accs, val_acc=50.0, baseline_accs_env={}, slot_samples=samples  # slot not ablated
    )
    assert accs["sA"].cf_weighted_integral == pytest.approx(0.0)
    assert accs["sA"].n_on_path_steps == 1
    assert accs["sA"].n_none_steps == 1


def test_sweep_skips_empty_slots() -> None:
    """A slot with no seed (seed_id None) is not accumulated."""
    accs: dict[str, SeedResidencyAccumulator] = {}
    samples = [SlotResidencySample(slot_id="r0c0", seed_id=None, stage=0, alpha=0.0, params=0)]
    accumulate_residency_for_env(
        accs, val_acc=50.0, baseline_accs_env={}, slot_samples=samples
    )
    assert accs == {}


def test_sweep_accumulates_across_epochs_per_seed() -> None:
    """Repeated sweeps for the same seed accumulate; the per-seed key separates instances."""
    accs: dict[str, SeedResidencyAccumulator] = {}
    for alpha, baseline in ((0.25, 47.0), (0.5, 45.0)):
        accumulate_residency_for_env(
            accs,
            val_acc=50.0,
            baseline_accs_env={"r0c0": baseline},
            slot_samples=[SlotResidencySample(slot_id="r0c0", seed_id="sA", stage=STAGE_BLENDING, alpha=alpha, params=1000)],
        )
    # epoch1: (50-47)*0.25 = 0.75 ; epoch2: (50-45)*0.5 = 2.5 ; total 3.25
    assert accs["sA"].cf_weighted_integral == pytest.approx(3.25)
    assert accs["sA"].n_on_path_steps == 2


def test_sweep_off_path_seed_creates_no_residency() -> None:
    """A seed in TRAINING (stage < BLENDING) gets an accumulator but accrues nothing."""
    accs: dict[str, SeedResidencyAccumulator] = {}
    samples = [SlotResidencySample(slot_id="r0c0", seed_id="sA", stage=STAGE_TRAINING, alpha=0.0, params=1000)]
    accumulate_residency_for_env(
        accs, val_acc=50.0, baseline_accs_env={"r0c0": 46.0}, slot_samples=samples
    )
    assert accs["sA"].n_on_path_steps == 0
    assert accs["sA"].cf_weighted_integral == pytest.approx(0.0)


def test_sweep_updates_params_snapshot() -> None:
    """params(seed) is snapshotted from the latest sample (PIN B time-invariance: the seed's
    active-param count is fixed once instantiated; the sweep keeps the current value)."""
    accs: dict[str, SeedResidencyAccumulator] = {}
    accumulate_residency_for_env(
        accs, val_acc=50.0, baseline_accs_env={"r0c0": 46.0},
        slot_samples=[SlotResidencySample(slot_id="r0c0", seed_id="sA", stage=STAGE_BLENDING, alpha=1.0, params=2000)],
    )
    assert accs["sA"].params == 2000
