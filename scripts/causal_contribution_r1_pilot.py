#!/usr/bin/env python
"""Causal-contribution R1 pairing-pilot driver (SUPPRESS-SLOT estimand-invariant core).

Runs ONE arm of the parallel-control causal-contribution run across seeds 41-45
on ONE device, sequentially. The owner launches arms in parallel respecting the
<=2-concurrent-GPU budget (e.g. control on cuda:0 + suppress_slot_on on cuda:1).

Design: docs/plans/concepts/2026-06-28-causal-contribution-run-design-v2.md (§5).
Config:  configs/config-3slot-3seed-suppress-slot-r0c0.json (shared hyperparams,
         cloned from config-3slot-3seed-baseline-shaped.json, compile OFF).

THREE arms (the same SUPPRESS-SLOT cohort, toggled):
  control            -- RNG split ON, NO stationary mask (matched factual reference)
  suppress_slot_off  -- RNG split ON, suppression OFF => CRN NO-OP control: must be
                        compare-point-identical to `control` for the same seed at
                        every step (the no-op proof half of R1)
  suppress_slot_on   -- PRIMARY: r0c0 un-committable for the whole run

All three run with rng_three_domain_split=True so the controller/host/blueprint
streams are split and per-step compare-point hashes are emitted; pairing is
preserved by the shared controller init + data order. The intervention is
byte-identical-behavior when OFF (gated): suppress_slot_off mutates no mask.

This script intentionally does NOT auto-launch a sweep. Invoke per arm/device.

Usage (owner runs; do NOT run inside CI / shared-GPU windows):
  PYTHONPATH=src uv run python scripts/causal_contribution_r1_pilot.py \
      --arm suppress_slot_on --device cuda:1 --telemetry-dir /tmp/cc_r1_slot_on
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from esper.leyline import SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY
from esper.simic.training import TrainingConfig
from esper.simic.training.proof_baselines import (
    CAUSAL_CONTRIBUTION_R1_PAIR_ID,
    build_suppress_slot_cohort,
)
from esper.simic.training.vectorized import train_ppo_vectorized

DEFAULT_CONFIG = "configs/config-3slot-3seed-suppress-slot-r0c0.json"
SEEDS = (41, 42, 43, 44, 45)
ARMS = ("control", "suppress_slot_off", "suppress_slot_on")
_HARNESS_KEY = "_causal_contribution_r1"


def _load_base_kwargs(config_path: str) -> dict[str, object]:
    """Load shared hyperparams, stripping the driver-only harness metadata block."""
    raw = json.loads(Path(config_path).read_text())
    raw.pop(_HARNESS_KEY, None)  # not part of the TrainingConfig schema
    config = TrainingConfig.from_dict(raw)
    return config.to_train_kwargs()


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal-contribution R1 pilot (one arm)")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--device", required=True, help="e.g. cuda:0 / cuda:1 / cpu")
    p.add_argument("--telemetry-dir", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument(
        "--seeds",
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated seeds (default 41-45).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-seed plan and exit WITHOUT training.",
    )
    p.add_argument(
        "--gpu-preload",
        action="store_true",
        help=(
            "Enable gpu_preload (CIFAR data-pipeline fix, ~3x faster; data-pipeline "
            "only, orthogonal to the controller/blueprint RNG domains). Re-verify the "
            "compare-point gates hold on a gpu_preload smoke before trusting it."
        ),
    )
    return p.parse_args(argv)


def _arm_train_kwargs(arm: str, *, seed: int) -> dict[str, object]:
    """Harness-specific train() kwargs for one arm/seed.

    rng_three_domain_split is ON for ALL arms (the instrumented build): that is
    what makes the arms CRN-paired. The suppression toggle and the lifecycle
    policy differ per arm.
    """
    if arm == "control":
        return {
            "rng_three_domain_split": True,
            "proof_baseline_mode": None,
            "proof_baseline_lifecycle_policy": None,
            "proof_baseline_suppression_enabled": False,
            "proof_baseline_pair_id": CAUSAL_CONTRIBUTION_R1_PAIR_ID,
        }
    suppression_enabled = arm == "suppress_slot_on"
    cohort = build_suppress_slot_cohort(
        training_seed=seed, suppression_enabled=suppression_enabled
    )
    return {
        "rng_three_domain_split": True,
        "proof_baseline_mode": cohort.mode.value,
        "proof_baseline_lifecycle_policy": cohort.lifecycle_policy,
        "proof_baseline_suppression_enabled": suppression_enabled,
        "proof_baseline_pair_id": cohort.proof_baseline_pair_id,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    base = _load_base_kwargs(args.config)
    # The CRN compare-point + RNG-domain determinism are validated under eager
    # execution; refuse to silently run compiled and muddy the proof.
    if base.get("compile_mode") not in ("off", None):
        raise ValueError(
            f"R1 pilot requires compile_mode='off' for a clean CRN compare-point; "
            f"config has compile_mode={base.get('compile_mode')!r}"
        )
    base["compile_mode"] = "off"

    print(
        f"[cc-r1] arm={args.arm} device={args.device} seeds={list(seeds)} "
        f"policy={SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY} compile=off",
        flush=True,
    )
    for seed in seeds:
        arm_kwargs = _arm_train_kwargs(args.arm, seed=seed)
        run_kwargs = dict(base)
        run_kwargs.update(arm_kwargs)
        run_kwargs["seed"] = seed
        run_kwargs["device"] = args.device
        run_kwargs["telemetry_dir"] = f"{args.telemetry_dir}/{args.arm}_s{seed}"
        run_kwargs["group_id"] = f"cc_r1_{args.arm}_s{seed}"
        if args.gpu_preload:
            run_kwargs["gpu_preload"] = True
        _preflight(run_kwargs, arm=args.arm, seed=seed, canonical_seeds=seeds)
        if args.dry_run:
            print(f"[cc-r1] DRY-RUN seed={seed}: {arm_kwargs}", flush=True)
            continue
        print(f"[cc-r1] START arm={args.arm} seed={seed}", flush=True)
        train_ppo_vectorized(**run_kwargs)
        print(f"[cc-r1] DONE  arm={args.arm} seed={seed}", flush=True)
    # Post-run analysis gating (run by the pilot analysis, not here): for each
    # paired seed, load the arms' INTERVENTION_STEP records and call
    #   assert_constant_controller_stride(draw_counts)               # within-run
    #   assert_offset_free_parity(control_counts, off_counts)        # cross-arm
    #   assert_target_slot_never_committed(on_germination_slot_ids)  # suppression
    # from esper.simic.training.intervention — these are HARD pre-verdict gates.
    return 0


def _preflight(
    run_kwargs: dict, *, arm: str, seed: int, canonical_seeds: tuple[int, ...]
) -> None:
    """Fail loudly BEFORE a run if an arm would break pairing or drop telemetry.

    Governor #1/#2/#3: every arm must run with the RNG split (else arms are not
    CRN-paired); paired arms must share the master seed (the cross-arm join is by
    seed); and a FileOutput/events.jsonl backend must be attached (INTERVENTION_STEP
    is severity="debug" and a console-only run would silently drop it).
    """
    if run_kwargs.get("rng_three_domain_split") is not True:
        raise ValueError(
            f"arm {arm!r} must run with rng_three_domain_split=True (CRN pairing)"
        )
    if run_kwargs.get("seed") != seed or seed not in canonical_seeds:
        raise ValueError(
            f"seed pairing broken: run seed {run_kwargs.get('seed')} not the paired "
            f"arm seed {seed} in {canonical_seeds}"
        )
    if not run_kwargs.get("telemetry_dir") or run_kwargs.get("use_telemetry") is not True:
        raise ValueError(
            "INTERVENTION_STEP (severity='debug') needs a FileOutput backend: set "
            "telemetry_dir AND use_telemetry=True, else the compare-point is dropped"
        )
    if run_kwargs.get("compile_mode") not in ("off", None):
        raise ValueError("R1 requires compile_mode='off' for a clean CRN compare-point")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
