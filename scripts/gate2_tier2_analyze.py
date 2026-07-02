"""GATE 2 Tier-2 paired readout (pre-registration:
docs/analysis/2026-07-02-gate2-learnability-probe-preregistration.md).

Reads telemetry/gate2_probe/tier2_{arm}_s{seed}.jsonl pairs and reports:
  - Primary: paired dP(FOSSILIZE|eligible) at the read batch, exact one-sided
    sign-flip permutation p across seed-pairs (dependency-free; N=5 => min
    p = 1/32).
  - Magnitude floor: median |paired dP| vs the control arm's own natural
    drift over the trailing 5 updates (self-calibrating floor).
  - Tertiary (L4 transient): adv_base at fossilize steps, retro vs control
    series (critic absorption of prior injections shows up here).
  - Tier-1 pooled screen: realized dadv/sigma_A for both routings.

Usage: uv run python scripts/gate2_tier2_analyze.py [--read-batch K] \
           [--dir telemetry/gate2_probe] [--seeds 51 52 53 54 55]
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

SERIES_KEYS = ("dadv_retro_at_foss", "dadv_term_at_foss")


def load_run(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def propensity_series(records: list[dict]) -> dict[int, float]:
    out = {}
    for r in records:
        if "skipped" in r:
            continue
        el = r["propensity"]["eligible"]
        if el.get("n", 0) > 0 and el["mean"] is not None:
            out[r["batch"]] = el["mean"]
    return out


def adv_base_series(records: list[dict]) -> dict[int, float]:
    out = {}
    for r in records:
        if "skipped" in r:
            continue
        ab = r.get("adv_base_at_foss", {})
        if ab.get("n", 0) > 0 and ab["mean"] is not None:
            out[r["batch"]] = ab["mean"]
    return out


def sign_flip_p_one_sided(deltas: list[float]) -> float:
    """Exact one-sided sign-flip permutation p for mean(deltas) > 0."""
    n = len(deltas)
    observed = sum(deltas) / n
    count = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=n):
        m = sum(s * d for s, d in zip(signs, deltas)) / n
        if m >= observed - 1e-15:
            count += 1
        total += 1
    return count / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("telemetry/gate2_probe"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 52, 53, 54, 55])
    parser.add_argument("--read-batch", type=int, default=None,
                        help="Batch index for the endpoint read (default: last "
                        "batch present in ALL runs)")
    args = parser.parse_args()

    runs: dict[tuple[str, int], list[dict]] = {}
    for seed in args.seeds:
        for arm in ("control", "retro"):
            path = args.dir / f"tier2_{arm}_s{seed}.jsonl"
            if not path.exists():
                print(f"MISSING: {path} — pair for seed {seed} incomplete, skipping seed")
                break
            runs[(arm, seed)] = load_run(path)

    seeds = [s for s in args.seeds if ("control", s) in runs and ("retro", s) in runs]
    if not seeds:
        raise SystemExit("No complete pairs found.")

    # Endpoint batch: latest batch present in every run's propensity series.
    prop = {key: propensity_series(recs) for key, recs in runs.items()}
    common = set.intersection(*(set(v.keys()) for v in prop.values()))
    if not common:
        raise SystemExit("No common batch with eligible propensity across all runs.")
    read_batch = args.read_batch or max(common)
    if read_batch not in common:
        raise SystemExit(f"--read-batch {read_batch} not present in all runs; "
                         f"common batches: {sorted(common)}")

    # Primary endpoint.
    deltas = []
    for s in seeds:
        d = prop[("retro", s)][read_batch] - prop[("control", s)][read_batch]
        deltas.append(d)
        print(f"seed {s}: P_retro={prop[('retro', s)][read_batch]:.4f} "
              f"P_control={prop[('control', s)][read_batch]:.4f} dP={d:+.4f}")
    p = sign_flip_p_one_sided(deltas)
    print(f"\nPRIMARY @ batch {read_batch}: mean dP = {statistics.mean(deltas):+.4f}, "
          f"median dP = {statistics.median(deltas):+.4f}, "
          f"one-sided sign-flip p = {p:.4f} (N={len(deltas)})")

    # Magnitude floor: control's own natural drift over trailing 5 batches.
    drifts = []
    for s in seeds:
        series = prop[("control", s)]
        past = read_batch - 5
        if past in series:
            drifts.append(abs(series[read_batch] - series[past]))
    if drifts:
        floor = statistics.median(drifts)
        med = statistics.median(deltas)
        print(f"MAGNITUDE FLOOR: median control 5-update drift = {floor:.4f}; "
              f"median paired dP = {med:+.4f} -> "
              f"{'ABOVE' if med >= floor else 'BELOW'} floor")

    # Persistence: dP at the two batches preceding the endpoint.
    for back in (2, 4):
        b = read_batch - back
        if all(b in prop[(a, s)] for a in ("control", "retro") for s in seeds):
            ds = [prop[("retro", s)][b] - prop[("control", s)][b] for s in seeds]
            print(f"persistence check @ batch {b}: mean dP = {statistics.mean(ds):+.4f}")

    # Tertiary: L4 transient (adv at fossilize steps, retro arm vs control arm).
    print("\nL4 TRANSIENT (mean adv_base at fossilize steps, per batch):")
    for s in seeds:
        ar = adv_base_series(runs[("retro", s)])
        ac = adv_base_series(runs[("control", s)])
        both = sorted(set(ar) & set(ac))
        line = " ".join(f"b{b}:{ar[b] - ac[b]:+.2f}" for b in both[-8:])
        print(f"  seed {s} (retro-control): {line}")

    # Tier-1 pooled screen across ALL runs (per-unit-delta, sigma_A units).
    print("\nTIER-1 POOLED SCREEN (realized dadv/sigma_A at delta*, per event batch):")
    for key, label in (("dadv_retro_at_foss", "retro"), ("dadv_term_at_foss", "terminal")):
        ratios = []
        for recs in runs.values():
            for r in recs:
                if "skipped" in r:
                    continue
                d = r.get(key, {})
                if d.get("n", 0) > 0 and d["mean"] is not None and r["sigma_A_raw"] > 0:
                    ratios.append(d["mean"] / r["sigma_A_raw"])
        if ratios:
            print(f"  {label}: median {statistics.median(ratios):.4f}, "
                  f"q10 {sorted(ratios)[int(0.1 * (len(ratios) - 1))]:.4f}, "
                  f"n_batches {len(ratios)} "
                  f"(G2a gate: CLEAR >= 0.10, FAIL < 0.05)")


if __name__ == "__main__":
    main()
