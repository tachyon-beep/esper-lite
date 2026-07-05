#!/usr/bin/env python
"""PIN-E placebo noise-floor run driver (esper-lite-94869250f1).

Runs the 3-slot near-inert placebo schedule for one epsilon arm (placebo init
std) across seeds on ONE device, sequentially. The measurement: three frozen
(seed_lr=0) tiny-random placebo seeds held at HOLDING alpha=1, never
fossilized; the fused val pass logs the full 2^3 counterfactual factorial
every epoch, from which scripts/pin_e_placebo_analyze.py assembles phi and
c_paid OFFLINE. shapley_synergy_scale stays 0.0 for every run this driver can
launch — the preflight refuses anything else.

Plan: docs/plans/ready/2026-07-03-pin-e-placebo-harness.md (dual-reviewed).
Config: configs/config-pin-e-placebo.json.

Epsilon-ladder mechanism: the placebo blueprint reads the module constant
esper.kasmina.blueprints.cnn.PLACEBO_INIT_STD at germination; this driver sets
it from --init-std BEFORE training and records the value in each run's
pin_e_preflight.json. The per-epoch lr==0 guard lives in the WI-5 integration
test (tests/integration/test_pin_e_placebo_schedule.py) against the real
optimizer path; no LR scheduler exists in this codebase.

Usage (one arm per device; both GPUs may run different arms in parallel):
  PYTHONPATH=src uv run python scripts/pin_e_placebo_run.py \
      --device cuda:0 --telemetry-dir telemetry/pin_e --init-std 1e-3 \
      --seeds 41,42,43 --gpu-preload
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_ACTION_COUNT,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_VERSION,
    ProofBaselineMode,
)
from esper.simic.training import TrainingConfig
from esper.simic.training.vectorized import train_ppo_vectorized

DEFAULT_CONFIG = "configs/config-pin-e-placebo.json"
SEEDS = (41, 42, 43)
_HARNESS_KEY = "_pin_e"
# The schedule holds all three placebos co-resident at HOLDING from epoch 51;
# a run must comfortably contain that window plus a terminal epoch.
_MIN_MAX_EPOCHS = 60


def _load_base_kwargs(config_path: str) -> dict[str, object]:
    """Load shared hyperparams, stripping the driver-only harness metadata."""
    raw = json.loads(Path(config_path).read_text())
    raw.pop(_HARNESS_KEY, None)
    config = TrainingConfig.from_dict(raw)
    return config.to_train_kwargs()


def _std_tag(init_std: float) -> str:
    return f"{init_std:g}"


def _build_run_kwargs(
    base: dict[str, object],
    *,
    seed: int,
    device: str,
    telemetry_base: str,
    init_std: float,
) -> dict[str, object]:
    run_kwargs = dict(base)
    tag = _std_tag(init_std)
    run_kwargs["seed"] = seed
    run_kwargs["device"] = device
    run_kwargs["telemetry_dir"] = f"{telemetry_base}/placebo_std{tag}_s{seed}"
    run_kwargs["group_id"] = f"pin_e_std{tag}_s{seed}"
    run_kwargs["seed_lr_override"] = 0.0
    run_kwargs["proof_baseline_mode"] = ProofBaselineMode.FIXED_SCHEDULE.value
    run_kwargs["proof_baseline_lifecycle_policy"] = "apply_declared_lifecycle_schedule"
    run_kwargs["proof_baseline_schedule_id"] = FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1
    run_kwargs["proof_baseline_schedule_hash"] = FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH
    run_kwargs["proof_baseline_schedule_version"] = (
        FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_VERSION
    )
    run_kwargs["proof_baseline_schedule_action_count"] = (
        FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_ACTION_COUNT
    )
    return run_kwargs


def _preflight(run_kwargs: dict) -> None:
    """Fail loudly BEFORE a run if a measurement invariant is broken.

    Every check here protects the noise-floor estimand: a violation would not
    crash the run — it would silently corrupt tau.
    """
    if run_kwargs.get("compile_mode") not in ("off", None):
        raise ValueError(
            "PIN-E requires compile_mode='off' (eager eval numerics); got "
            f"{run_kwargs.get('compile_mode')!r}"
        )
    if not run_kwargs.get("telemetry_dir") or run_kwargs.get("use_telemetry") is not True:
        raise ValueError(
            "PIN-E needs telemetry_dir AND use_telemetry=True: the counterfactual "
            "matrix events ARE the measurement, and gradient-health collection "
            "feeds the G2 gate (esper-lite-4fe98055f7)"
        )
    if run_kwargs.get("seed_lr_override") != 0.0:
        raise ValueError(
            "PIN-E requires seed_lr_override=0.0 (frozen placebo delta); got "
            f"{run_kwargs.get('seed_lr_override')!r}"
        )
    if run_kwargs.get("proof_baseline_schedule_id") != FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1:
        raise ValueError(
            "PIN-E requires the placebo declared schedule "
            f"{FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1!r}; got "
            f"{run_kwargs.get('proof_baseline_schedule_id')!r}"
        )
    if run_kwargs.get("shapley_synergy_scale") != 0.0:
        raise ValueError(
            "PIN-E measures with the term OFF: shapley_synergy_scale must be 0.0 "
            f"(got {run_kwargs.get('shapley_synergy_scale')!r}); enabling is a "
            "separate owner-gated decision (esper-lite-f22a1d48a7)"
        )
    for gate in ("auto_forward_g1", "auto_forward_g2", "auto_forward_g3"):
        if run_kwargs.get(gate) is not False:
            raise ValueError(
                f"PIN-E requires {gate}=False: auto_forward gates would advance "
                "seeds ahead of the declared schedule's ADVANCE epochs"
            )
    if run_kwargs.get("amp") is not False:
        raise ValueError(
            "PIN-E requires amp=False: autocast perturbs the eval numerics the "
            "quantization-grid measurement depends on"
        )
    if run_kwargs.get("slots") != ["r0c0", "r0c1", "r0c2"]:
        raise ValueError(
            f"PIN-E requires slots=['r0c0','r0c1','r0c2']; got {run_kwargs.get('slots')!r}"
        )
    if run_kwargs.get("max_seeds") != 3:
        raise ValueError(
            f"PIN-E requires max_seeds=3 (k=3 coalition); got {run_kwargs.get('max_seeds')!r}"
        )
    max_epochs = run_kwargs.get("max_epochs")
    if not isinstance(max_epochs, int) or max_epochs < _MIN_MAX_EPOCHS:
        raise ValueError(
            f"PIN-E requires max_epochs >= {_MIN_MAX_EPOCHS} (all-HOLDING window "
            f"starts at epoch 51); got {max_epochs!r}"
        )


def _write_preflight(
    run_kwargs: dict, *, init_std: float, config_path: str
) -> Path:
    """Write the provenance file the analyzer refuses to run without."""
    run_dir = Path(str(run_kwargs["telemetry_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schedule_id": run_kwargs["proof_baseline_schedule_id"],
        "schedule_hash": run_kwargs["proof_baseline_schedule_hash"],
        "seed_lr_override": run_kwargs["seed_lr_override"],
        "shapley_synergy_scale": run_kwargs["shapley_synergy_scale"],
        "placebo_init_std": init_std,
        "seed": run_kwargs["seed"],
        "group_id": run_kwargs["group_id"],
        "config": config_path,
        "n_episodes": run_kwargs["n_episodes"],
        "n_envs": run_kwargs["n_envs"],
        "max_epochs": run_kwargs["max_epochs"],
        "created_unix": time.time(),
    }
    path = run_dir / "pin_e_preflight.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PIN-E placebo noise-floor run (one arm)")
    p.add_argument("--device", required=True, help="e.g. cuda:0 / cuda:1 / cpu")
    p.add_argument("--telemetry-dir", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument(
        "--init-std",
        type=float,
        default=1e-3,
        help="Placebo final-conv init std (epsilon-ladder arm; recorded in provenance).",
    )
    p.add_argument(
        "--seeds",
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated run seeds (default 41,42,43).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gpu-preload", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.init_std <= 0.0:
        raise ValueError(f"--init-std must be > 0 (got {args.init_std})")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    base = _load_base_kwargs(args.config)

    # Epsilon-ladder knob: the blueprint factory reads this module constant at
    # germination. Recorded in provenance; visible here, nowhere else.
    import esper.kasmina.blueprints.cnn as cnn_blueprints

    cnn_blueprints.PLACEBO_INIT_STD = args.init_std

    print(
        f"[pin-e] device={args.device} seeds={list(seeds)} init_std={args.init_std:g} "
        f"schedule={FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1} scale=0.0 compile=off",
        flush=True,
    )
    for seed in seeds:
        run_kwargs = _build_run_kwargs(
            base,
            seed=seed,
            device=args.device,
            telemetry_base=args.telemetry_dir,
            init_std=args.init_std,
        )
        if args.gpu_preload:
            run_kwargs["gpu_preload"] = True
        _preflight(run_kwargs)
        if args.dry_run:
            print(f"[pin-e] DRY-RUN seed={seed}: {run_kwargs['group_id']}", flush=True)
            continue
        _write_preflight(run_kwargs, init_std=args.init_std, config_path=args.config)
        print(f"[pin-e] START seed={seed} -> {run_kwargs['telemetry_dir']}", flush=True)
        train_ppo_vectorized(**run_kwargs)
        print(f"[pin-e] DONE  seed={seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
