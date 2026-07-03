#!/usr/bin/env python
"""PIN-E placebo noise-floor analyzer (WI-7, esper-lite-94869250f1).

OFFLINE assembly of the tau (noise-floor) recommendation from the counterfactual
matrix telemetry the run driver (scripts/pin_e_placebo_run.py) produces. For
each terminal counterfactual-matrix event the analyzer reconstructs the full
2^3 coalition-value table v: frozenset(enabled slots) -> accuracy, then for each
of the three co-resident null placebo players computes:

  Shapley:  phi(s) = 1/3 (v{s}-v{})
                   + 1/6 [(v{s,t}-v{t}) + (v{s,u}-v{u})]
                   + 1/3 (v(C)-v{t,u})
  Paid credit (LOO-from-empty):  c_paid(s) = v{s} - v{}
  Signed excess:                 e(s)      = phi(s) - c_paid(s)   (KEEP THE SIGN)

tau = P99 of the SIGNED terminal excess distribution pooled per epsilon arm,
with a block-bootstrap CI (blocks = episodes). Only the positive tail leaks
credit through the live term's max(0, phi - c_paid - tau) deadband, so the
signed quantiles (not |excess|) are the estimand; |excess| is reported as
context only.

Two DISTINCT deliverables, kept separate in the report (do not conflate):
- D2 (tau): the terminal signed (phi - c_paid) distribution + P99 tau + CI,
  assembled from COUNTERFACTUAL_MATRIX_COMPUTED (above).
- D1 (GATE-0 line): per-STAGE mean (bias), std (noise floor), CoV of the
  CREDITED per-step LOO -- the reward system's own logged
  reward_components.seed_contribution in ANALYTICS_SNAPSHOT kind='last_action',
  grouped by reward_components.seed_stage. Source is the credited value (not a
  recomputation). Stages at alpha=0 (GERMINATED/TRAINING) log a null
  contribution and are labeled "structural zero (alpha=0)" so nobody reads them
  as measured noise-floor evidence. No bootstrap on D1.

Plan: docs/plans/ready/2026-07-03-pin-e-placebo-harness.md, especially section 2.6
(D1/D2 / tau definition / epsilon ladder / non-degeneracy).

Exit codes:
  0  success
  1  provenance refusal (a run dir's pin_e_preflight.json fails the hard gate)
  2  degenerate floor (every terminal excess sample is exactly zero across all
     arms) -- the PDR-0014 reversal trigger

Usage:
  uv run python scripts/pin_e_placebo_analyze.py telemetry/pin_e [more_roots...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from esper.leyline import SeedStage
from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
)

# Stages where the seed's blend alpha is 0 (STE output fully masked): the credit
# system logs a null seed_contribution because the leave-one-out marginal is
# structurally zero. Reported as such, never as measured noise floor.
_ALPHA_ZERO_STAGES = frozenset(
    {SeedStage.DORMANT.value, SeedStage.GERMINATED.value, SeedStage.TRAINING.value}
)

MATRIX_EVENT = "COUNTERFACTUAL_MATRIX_COMPUTED"
PREFLIGHT_NAME = "pin_e_preflight.json"
REPORT_NAME = "pin_e_noise_floor_report.json"

# Block bootstrap of the P99: resample episodes (each keeps its 3 per-seed
# samples together), fixed RNG seed for cross-run reproducibility.
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20260703
BOOTSTRAP_CI = (2.5, 97.5)

# Efficiency identity is algebraically exact for the Shapley formula; a fired
# check means the assembly is wrong. abs_tol carries it for null players where
# v(C) ~= v(empty) and the pure relative tolerance would blow up.
_EFF_RTOL = 1e-9
_EFF_ATOL = 1e-9

# Zero threshold for the D2 non-degeneracy trigger (owner-RATIFIED 2026-07-03).
# Accuracies sit on a ~100/|testset| pp grid (~0.12 pp for the smoke test set),
# but the 1/3,1/6-weighted Shapley assembly leaves ~1e-17 float residuals on
# excess values that are physically zero. Direction of safety: a tolerance only
# WIDENS what counts as zero, so it can only make the exit-2 reversal trigger
# fire MORE readily -- a deterministic-zero placebo whose excess is 1e-17 float
# dust must still trip exit-2, which a literal != 0.0 would miss. Real signal is
# grid-quantized at ~2e-2, seven orders of magnitude above this threshold, so no
# real sample can be misclassified as zero.
ZERO_TOL = 1e-9

# The hard provenance gate (field -> required value). shapley_synergy_scale and
# seed_lr_override must be exactly zero (the term OFF, the delta frozen); the
# schedule id/hash pin the exact declared placebo schedule that was run.
_REQUIRED_PROVENANCE: dict[str, object] = {
    "shapley_synergy_scale": 0.0,
    "seed_lr_override": 0.0,
    "schedule_id": FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
    "schedule_hash": FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH,
}


class ProvenanceError(Exception):
    """A run dir's provenance file is missing or fails the hard gate (exit 1)."""


# ---------------------------------------------------------------------------
# Provenance gate (HARD, runs before any events are read)
# ---------------------------------------------------------------------------
def read_preflight(run_dir: Path) -> dict:
    """Load run_dir/pin_e_preflight.json, refusing (not KeyError) if absent."""
    path = run_dir / PREFLIGHT_NAME
    if not path.exists():
        raise ProvenanceError(f"{run_dir}: missing {PREFLIGHT_NAME}")
    return json.loads(path.read_text())


def check_provenance(run_dir: Path, preflight: dict) -> None:
    """Refuse the run unless every measurement-critical field is exactly right.

    A wrong value here would not crash a run -- it would silently corrupt the
    noise floor -- so this gate fires BEFORE any events are read.
    """
    for field, expected in _REQUIRED_PROVENANCE.items():
        if field not in preflight:
            raise ProvenanceError(
                f"{run_dir}: pin_e_preflight.json missing required field {field!r}"
            )
        actual = preflight[field]
        if actual != expected:
            raise ProvenanceError(
                f"{run_dir}: pin_e_preflight.json {field}={actual!r} but PIN-E "
                f"requires {field}={expected!r}"
            )


# ---------------------------------------------------------------------------
# Shapley assembly (the arithmetic core)
# ---------------------------------------------------------------------------
def build_value_map(event_data: dict) -> dict[frozenset[str], float]:
    """v: frozenset(enabled slot ids) -> accuracy, from one matrix event's configs.

    seed_mask[i] is positional against slot_ids[i]; true == seed enabled.
    """
    slot_ids = event_data["slot_ids"]
    vmap: dict[frozenset[str], float] = {}
    for config in event_data["configs"]:
        mask = config["seed_mask"]
        enabled = frozenset(slot for slot, on in zip(slot_ids, mask) if on)
        vmap[enabled] = config["accuracy"]
    return vmap


def shapley_paid_excess(
    vmap: dict[frozenset[str], float], slot_ids: list[str]
) -> dict[str, tuple[float, float, float]]:
    """Per-slot (phi, c_paid, excess) for the k=3 coalition. Requires all 8 subsets."""
    coalition = frozenset(slot_ids)
    empty: frozenset[str] = frozenset()
    out: dict[str, tuple[float, float, float]] = {}
    for slot in slot_ids:
        other_t, other_u = (x for x in slot_ids if x != slot)
        phi = (
            (1.0 / 3.0) * (vmap[frozenset({slot})] - vmap[empty])
            + (1.0 / 6.0)
            * (
                (vmap[frozenset({slot, other_t})] - vmap[frozenset({other_t})])
                + (vmap[frozenset({slot, other_u})] - vmap[frozenset({other_u})])
            )
            + (1.0 / 3.0) * (vmap[coalition] - vmap[frozenset({other_t, other_u})])
        )
        c_paid = vmap[frozenset({slot})] - vmap[empty]
        out[slot] = (phi, c_paid, phi - c_paid)
    return out


def assert_efficiency(
    vmap: dict[frozenset[str], float],
    slot_ids: list[str],
    per_slot: dict[str, tuple[float, float, float]],
) -> None:
    """Sum_s phi(s) must equal v(C) - v(empty). A violation is an assembly bug."""
    sum_phi = sum(rec[0] for rec in per_slot.values())
    efficiency = vmap[frozenset(slot_ids)] - vmap[frozenset()]
    if not math.isclose(sum_phi, efficiency, rel_tol=_EFF_RTOL, abs_tol=_EFF_ATOL):
        raise RuntimeError(
            "Shapley efficiency identity violated (assembly bug): "
            f"Sum phi = {sum_phi!r} but v(C) - v(empty) = {efficiency!r}"
        )


# ---------------------------------------------------------------------------
# Event assembly + terminal selection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TerminalRecord:
    """The terminal (last usable) counterfactual matrix for one (env, episode)."""

    run_dir: str
    env_id: int
    episode_idx: int
    slot_ids: list[str]
    per_slot: dict[str, tuple[float, float, float]]  # slot -> (phi, c_paid, excess)


def is_usable(event_data: dict) -> bool:
    """A matrix event is usable iff it is the full k=3 2^3 factorial.

    Over three distinct slots there are exactly 8 possible masks, so 8 distinct
    enabled-sets == every coalition present. k<3 (early lifecycle) and duplicate
    -mask k=3 events fail this and are skipped.
    """
    if len(event_data["slot_ids"]) != 3:
        return False
    return len(build_value_map(event_data)) == 8


def find_events_file(run_dir: Path) -> Path:
    """The single events.jsonl the run wrote (under a timestamped subdir)."""
    hits = sorted(run_dir.rglob("events.jsonl"))
    if len(hits) != 1:
        raise RuntimeError(
            f"{run_dir}: expected exactly one events.jsonl, found {len(hits)}"
        )
    return hits[0]


def load_terminal_records(run_dir: Path) -> list[TerminalRecord]:
    """Assemble the terminal per-seed (phi, c_paid, excess) for each episode.

    Terminal == the LAST usable matrix event in file order for each
    (env_id, episode_idx) key (a converged-terminal measurement); earlier
    events are transient. Every terminal is cross-checked against the Shapley
    efficiency identity.
    """
    events_path = find_events_file(run_dir)
    terminal_data: dict[tuple[int, int], dict] = {}
    with events_path.open() as fh:
        for line in fh:
            if MATRIX_EVENT not in line:  # cheap pre-filter before json.loads
                continue
            event = json.loads(line)
            if event["event_type"] != MATRIX_EVENT:
                continue
            data = event["data"]
            if not is_usable(data):
                continue
            terminal_data[(data["env_id"], data["episode_idx"])] = data  # last wins

    if not terminal_data:
        raise RuntimeError(
            f"{run_dir}: no usable k=3 counterfactual-matrix terminals in "
            f"{events_path}"
        )

    records: list[TerminalRecord] = []
    for (env_id, episode_idx), data in terminal_data.items():
        slot_ids = data["slot_ids"]
        vmap = build_value_map(data)
        per_slot = shapley_paid_excess(vmap, slot_ids)
        assert_efficiency(vmap, slot_ids, per_slot)
        records.append(
            TerminalRecord(str(run_dir), env_id, episode_idx, slot_ids, per_slot)
        )
    return records


# ---------------------------------------------------------------------------
# Blocks + block bootstrap
# ---------------------------------------------------------------------------
def excess_blocks(
    records: list[TerminalRecord],
) -> list[tuple[tuple[str, int, int], list[float]]]:
    """One block per episode: ((run_dir, env_id, episode_idx), [3 per-seed excess]).

    Deterministically sorted so the fixed-seed bootstrap is reproducible across
    machines regardless of filesystem / dict iteration order.
    """
    blocks = [
        (
            (rec.run_dir, rec.env_id, rec.episode_idx),
            [rec.per_slot[slot][2] for slot in rec.slot_ids],
        )
        for rec in records
    ]
    blocks.sort(key=lambda block: block[0])
    return blocks


def bootstrap_p99_ci(
    sample_blocks: list[list[float]],
    *,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Block-bootstrap CI of the P99 of signed excess.

    Resamples whole episodes (each carrying its 3 per-seed samples) with
    replacement B times; every episode keeps its within-episode correlation.
    Returns the (2.5%, 97.5%) percentile CI of the resampled P99s.
    """
    if not sample_blocks:
        raise RuntimeError("block bootstrap needs at least one episode block")
    sizes = {len(block) for block in sample_blocks}
    if sizes != {3}:
        raise RuntimeError(
            f"block bootstrap expects uniform blocks of 3 per-seed samples; "
            f"got block sizes {sorted(sizes)}"
        )
    arr = np.array(sample_blocks, dtype=float)
    n_blocks, block_size = arr.shape
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_blocks, size=(b, n_blocks))
    resampled = arr[picks].reshape(b, n_blocks * block_size)
    p99s = np.percentile(resampled, 99, axis=1)
    lo, hi = np.percentile(p99s, BOOTSTRAP_CI)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# D1: per-stage credited LOO (ANALYTICS_SNAPSHOT kind='last_action')
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LastActionRecord:
    """One per-step credited snapshot: the reward system's own seed_contribution."""

    run_dir: str
    env_id: int
    seed_stage: int | None
    seed_contribution: float | None
    val_acc: float | None


def load_last_action_records(run_dir: Path) -> list[LastActionRecord]:
    """Every ANALYTICS_SNAPSHOT kind='last_action' -> credited (stage, contribution)."""
    events_path = find_events_file(run_dir)
    records: list[LastActionRecord] = []
    with events_path.open() as fh:
        for line in fh:
            if '"last_action"' not in line:  # cheap pre-filter before json.loads
                continue
            event = json.loads(line)
            if event["event_type"] != "ANALYTICS_SNAPSHOT":
                continue
            data = event["data"]
            if data["kind"] != "last_action":
                continue
            rc = data["reward_components"]
            records.append(
                LastActionRecord(
                    str(run_dir),
                    data["env_id"],
                    rc["seed_stage"],
                    rc["seed_contribution"],
                    rc["val_acc"],
                )
            )
    if not records:
        raise RuntimeError(
            f"{run_dir}: no last_action snapshots in {events_path}"
        )
    return records


def _stage_name(stage: int | None) -> str:
    return "NONE" if stage is None else SeedStage(stage).name


def _val_acc_grid_step(records: list[LastActionRecord]) -> float | None:
    """The measurement quantum: 100/|testset|.

    val_acc is on a clean per-env grid (each env may hold a slightly different
    split size); pooling across envs blurs it, so derive the step per env and
    take the finest. Returns None if no env has >=2 distinct val_acc values.
    """
    by_env: dict[int, set[float]] = defaultdict(set)
    for rec in records:
        if rec.val_acc is not None:
            by_env[rec.env_id].add(round(rec.val_acc, 10))
    per_env_steps: list[float] = []
    for accs in by_env.values():
        ordered = np.array(sorted(accs))
        if ordered.size < 2:
            continue
        gaps = np.diff(ordered)
        gaps = gaps[gaps > ZERO_TOL]
        if gaps.size:
            per_env_steps.append(float(gaps.min()))
    return min(per_env_steps) if per_env_steps else None


def _guarded_cov(
    mean: float, std: float, grid_step: float | None
) -> tuple[float | None, str | None]:
    """CoV = std/|mean|, nulled when |mean| is at or below the quantization grid.

    A null player's mean sits sub-grid, so the ratio would be a garbage number.
    """
    if grid_step is None:
        return None, "grid step undetermined (insufficient val_acc resolution)"
    if abs(mean) <= grid_step:
        return None, "|mean| <= grid step (sub-quantization); CoV undefined"
    return std / abs(mean), None


def d1_per_stage(records: list[LastActionRecord]) -> dict:
    """GATE-0 line: per-stage mean/std/CoV of the CREDITED seed_contribution."""
    grid_step = _val_acc_grid_step(records)
    contrib_by_stage: dict[int | None, list[float]] = defaultdict(list)
    n_by_stage: dict[int | None, int] = defaultdict(int)
    for rec in records:
        n_by_stage[rec.seed_stage] += 1
        if rec.seed_contribution is not None:
            contrib_by_stage[rec.seed_stage].append(rec.seed_contribution)

    stage_rows: list[dict] = []
    for stage in sorted(n_by_stage, key=lambda s: (s is None, s)):
        vals = np.array(contrib_by_stage[stage], dtype=float)
        n_credited = int(vals.size)
        if n_credited > 0:
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n_credited > 1 else 0.0
            cov, cov_note = _guarded_cov(mean, std, grid_step)
        else:
            mean = std = cov = None
            cov_note = "no credited samples (structural zero at alpha=0)"
        alpha_zero = stage in _ALPHA_ZERO_STAGES
        if alpha_zero:
            label = "structural zero (alpha=0)"
        elif stage is None:
            label = "no stage (pre-germination)"
        else:
            label = "credited noise floor"
        stage_rows.append(
            {
                "seed_stage": stage,
                "seed_stage_name": _stage_name(stage),
                "alpha_zero": alpha_zero,
                "label": label,
                "n_snapshots": n_by_stage[stage],
                "n_credited": n_credited,
                "mean": mean,
                "std": std,
                "cov": cov,
                "cov_note": cov_note,
            }
        )
    return {"source": "credited", "grid_step": grid_step, "stages": stage_rows}


# ---------------------------------------------------------------------------
# Arm analysis + report
# ---------------------------------------------------------------------------
def discover_run_dirs(roots: list[Path]) -> list[Path]:
    """Every dir containing pin_e_preflight.json under any root (root itself counts)."""
    found: set[Path] = set()
    for root in roots:
        for preflight_path in Path(root).rglob(PREFLIGHT_NAME):
            found.add(preflight_path.parent)
    return sorted(found, key=str)


def signal_mask(samples: np.ndarray) -> np.ndarray:
    """Boolean mask of samples that are real signal (|excess| > ZERO_TOL).

    Values within ZERO_TOL of zero are float roundoff from the Shapley assembly,
    not a measured contribution.
    """
    return np.abs(samples) > ZERO_TOL


def _quantiles(values: np.ndarray, *, signed: bool) -> dict[str, float]:
    stats = {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }
    if signed:
        stats["min"] = float(values.min())
    return stats


def _provenance_echo(run_dir: Path, preflight: dict) -> dict:
    return {
        "run_dir": str(run_dir),
        "schedule_id": preflight["schedule_id"],
        "schedule_hash": preflight["schedule_hash"],
        "seed_lr_override": preflight["seed_lr_override"],
        "shapley_synergy_scale": preflight["shapley_synergy_scale"],
        "placebo_init_std": preflight["placebo_init_std"],
        "seed": preflight["seed"],
        "group_id": preflight["group_id"],
    }


def analyze(roots: list[Path]) -> dict:
    """Gate provenance, group by epsilon arm, and assemble the tau report."""
    run_dirs = discover_run_dirs(roots)
    if not run_dirs:
        raise ProvenanceError(
            f"no run dir with {PREFLIGHT_NAME} found under {[str(r) for r in roots]}"
        )

    # HARD provenance gate on EVERY run dir before a single event is read.
    preflights: dict[Path, dict] = {}
    for run_dir in run_dirs:
        preflight = read_preflight(run_dir)
        check_provenance(run_dir, preflight)
        preflights[run_dir] = preflight

    arms_map: dict[float, list[Path]] = defaultdict(list)
    for run_dir in run_dirs:
        arms_map[float(preflights[run_dir]["placebo_init_std"])].append(run_dir)

    arm_reports: list[dict] = []
    total_nonzero = 0
    for init_std in sorted(arms_map):
        arm_dirs = sorted(arms_map[init_std], key=str)
        blocks: list[tuple[tuple[str, int, int], list[float]]] = []
        for run_dir in arm_dirs:
            blocks.extend(excess_blocks(load_terminal_records(run_dir)))
        blocks.sort(key=lambda block: block[0])

        sample_blocks = [block[1] for block in blocks]
        samples = np.array([x for block in sample_blocks for x in block], dtype=float)
        lo, hi = bootstrap_p99_ci(sample_blocks)

        d1_records: list[LastActionRecord] = []
        for run_dir in arm_dirs:
            d1_records.extend(load_last_action_records(run_dir))
        d1 = d1_per_stage(d1_records)

        mask = signal_mask(samples)
        n_nonzero = int(mask.sum())
        total_nonzero += n_nonzero
        nonzero_abs = np.abs(samples[mask])
        signed = _quantiles(samples, signed=True)

        arm_reports.append(
            {
                "placebo_init_std": init_std,
                "run_dirs": [str(d) for d in arm_dirs],
                "provenance": [_provenance_echo(d, preflights[d]) for d in arm_dirs],
                "n_episodes": len(blocks),
                "n_samples": int(samples.size),
                "tau_p99": signed["p99"],
                "tau_p99_ci": [lo, hi],
                "signed_excess": signed,
                "abs_excess": _quantiles(np.abs(samples), signed=False),
                "d1_per_stage_loo": d1,
                "non_degeneracy": {
                    "n_nonzero": n_nonzero,
                    "fraction_nonzero": float(n_nonzero / samples.size),
                    "min_nonzero_abs": (
                        float(nonzero_abs.min()) if nonzero_abs.size else None
                    ),
                    "spread": float(samples.max() - samples.min()),
                    "n_distinct_values": int(np.unique(samples).size),
                    "degenerate": n_nonzero == 0,
                },
            }
        )

    overall_degenerate = total_nonzero == 0
    return {
        "roots": [str(r) for r in roots],
        "run_dirs": [str(d) for d in run_dirs],
        "schedule_id": FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
        "schedule_hash": FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH,
        "bootstrap": {
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "ci_percentiles": list(BOOTSTRAP_CI),
        },
        "arms": arm_reports,
        "overall_degenerate": overall_degenerate,
        "exit_code": 2 if overall_degenerate else 0,
    }


def write_report(root: Path, report: dict) -> Path:
    path = Path(root) / REPORT_NAME
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return path


def _print_summary(report: dict, report_path: Path) -> None:
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("PIN-E placebo noise-floor (D2 / tau) — offline analysis")
    lines.append("=" * 74)
    lines.append(f"schedule_id   : {report['schedule_id']}")
    lines.append(f"schedule_hash : {report['schedule_hash']}")
    lines.append(f"run dirs      : {len(report['run_dirs'])}")
    lines.append(
        f"bootstrap     : B={report['bootstrap']['B']} seed={report['bootstrap']['seed']} "
        f"CI={report['bootstrap']['ci_percentiles']}"
    )
    for arm in report["arms"]:
        nd = arm["non_degeneracy"]
        se = arm["signed_excess"]
        lines.append("-" * 74)
        lines.append(
            f"ARM placebo_init_std={arm['placebo_init_std']:g}  "
            f"episodes={arm['n_episodes']}  samples={arm['n_samples']}"
        )
        lines.append(
            "  signed excess: "
            f"mean={se['mean']:+.5f}  P50={se['p50']:+.5f}  P90={se['p90']:+.5f}  "
            f"P95={se['p95']:+.5f}  P99={se['p99']:+.5f}  max={se['max']:+.5f}  "
            f"min={se['min']:+.5f}"
        )
        ab = arm["abs_excess"]
        lines.append(
            "  |excess| ctx : "
            f"P50={ab['p50']:.5f}  P90={ab['p90']:.5f}  P95={ab['p95']:.5f}  "
            f"P99={ab['p99']:.5f}  max={ab['max']:.5f}"
        )
        lines.append(
            f"  tau=P99(signed)={arm['tau_p99']:+.5f}  "
            f"block-bootstrap CI=[{arm['tau_p99_ci'][0]:+.5f}, {arm['tau_p99_ci'][1]:+.5f}]"
        )
        verdict = "DEGENERATE" if nd["degenerate"] else "non-degenerate"
        lines.append(
            f"  non-degeneracy: {verdict}  nonzero={nd['n_nonzero']}/{arm['n_samples']} "
            f"({nd['fraction_nonzero']:.1%})  distinct={nd['n_distinct_values']}  "
            f"spread={nd['spread']:.5f}  min_nonzero_abs={nd['min_nonzero_abs']}"
        )
        d1 = arm["d1_per_stage_loo"]
        gs = "n/a" if d1["grid_step"] is None else f"{d1['grid_step']:.5f}"
        lines.append(
            f"  D1 GATE-0 (credited LOO by stage; grid_step={gs}):"
        )
        for row in d1["stages"]:
            if row["mean"] is None:
                stat = f"n_credited=0 [{row['label']}]"
            else:
                cov = "null(<=grid)" if row["cov"] is None else f"{row['cov']:.3f}"
                stat = (
                    f"mean={row['mean']:+.5f} std={row['std']:.5f} CoV={cov} "
                    f"[{row['label']}]"
                )
            lines.append(
                f"    {row['seed_stage_name']:<11} N={row['n_snapshots']:<5} {stat}"
            )
    lines.append("-" * 74)
    if len(report["arms"]) > 1:
        lines.append("epsilon plateau (tau=P99 signed, ascending init_std):")
        for arm in sorted(report["arms"], key=lambda a: a["placebo_init_std"]):
            lines.append(
                f"  std={arm['placebo_init_std']:g}: tau={arm['tau_p99']:+.5f}  "
                f"CI=[{arm['tau_p99_ci'][0]:+.5f}, {arm['tau_p99_ci'][1]:+.5f}]"
            )
        lines.append("-" * 74)
    if report["overall_degenerate"]:
        lines.append(
            "VERDICT: DEGENERATE — every terminal excess is exactly zero. "
            "PDR-0014 reversal trigger; tau is vacuous."
        )
    else:
        lines.append(
            "tau is CONSERVATIVE-PROVISIONAL (lower bound); finalize at the first "
            "ON calibration run. Human judges the epsilon plateau above."
        )
    lines.append(f"report written: {report_path}")
    print("\n".join(lines))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PIN-E placebo noise-floor analyzer (D2 / tau recommendation)"
    )
    parser.add_argument(
        "roots",
        nargs="+",
        help="Telemetry root dir(s) to discover run dirs under, or explicit run dir(s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    roots = [Path(r) for r in args.roots]
    try:
        report = analyze(roots)
    except ProvenanceError as exc:
        print(f"PIN-E provenance refusal (exit 1): {exc}", file=sys.stderr)
        return 1
    # Write the report BEFORE deciding the exit code so a degenerate verdict
    # still leaves a durable artifact.
    report_path = write_report(roots[0], report)
    _print_summary(report, report_path)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
