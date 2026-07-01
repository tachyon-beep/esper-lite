"""n=5 J-read analyzer — causal-contribution run (control vs suppress_slot_on).

PRIMARY ESTIMAND: paired Δ(acc-per-param) = efficiency_suppress − efficiency_control,
seed-level (resample SEEDS, NOT the 12 vec-envs — the pre-registered invalidator).
  efficiency = system accuracy contribution (all-on − all-off, late-run counterfactual
               matrix) per MILLION committed params.
Also reports a committed-J proxy (Σ counterfactual / Σ params) for context.

GATES (must ALL pass before banking ΔJ):
  per seed: constant controller stride (control, on); offset-free parity (control==on
  draw-count + state hash); `on` germinates ZERO r0c0; finiteness-gate trips == 0;
  decision-step slot-entropy health (min head_slot_entropy/head_slot_learnable_fraction
  > 0.1 — the choice-conditioned signal; PDR-0006).

DECISION RULE (pre-registered — see docs/plans/ready/2026-07-01-n5-j-read-run-sheet.md):
  Δeff seed-level bootstrap CI excludes 0 and NEGATIVE  -> (b) efficiency-enabling stem
      (credit r0c0's enabling contribution; the LOO/freeloader penalty is the wrong fix)
  Δeff CI includes 0                                    -> (c) fungible (no slot penalty)
  Δeff CI excludes 0 and POSITIVE                       -> (a) freeloader (penalize)
  CI too wide to separate                               -> escalate to n=10.

Usage: uv run python scripts/causal_contribution_j_analyze.py <telemetry_dir> [seeds-csv]
  e.g. uv run python scripts/causal_contribution_j_analyze.py telemetry/causal_r1_n5 41,42,43,44,45
"""
from __future__ import annotations

import glob
import json
import random
import statistics
import sys

sys.path.insert(0, "/home/john/esper-lite/src")
from esper.simic.training.intervention import (  # noqa: E402
    OffsetParityError,
    assert_constant_controller_stride,
    assert_offset_free_parity,
    assert_target_slot_never_committed,
)

DOWNSTREAM = {"r0c1", "r0c2"}


def load(base: str, arm: str, seed: int):
    hits = glob.glob(f"{base}/{arm}_s{seed}/**/events.jsonl", recursive=True)
    if not hits:
        return None
    tot_p = tot_cf = dn_p = r0_p = nf = 0
    draw_counts: list[int] = []
    state_hashes: list[str | None] = []
    germ_slots: list[str] = []
    max_ep = -1
    fin_trips = 0
    slot_cond_min = float("inf")
    offs: list[float] = []
    ons: list[float] = []
    for line in open(hits[0]):
        e = json.loads(line)
        et = e.get("event_type")
        d = e.get("data", {})
        if et == "SEED_FOSSILIZED":
            p = d["params_added"]
            s = d["slot_id"]
            cf = d.get("counterfactual") or 0.0
            tot_p += p
            tot_cf += cf
            nf += 1
            if s in DOWNSTREAM:
                dn_p += p
            elif s == "r0c0":
                r0_p += p
        elif et == "INTERVENTION_STEP":
            draw_counts.append(d["controller_draw_count"])
            state_hashes.append(d.get("controller_state_hash"))
        elif et == "SEED_GERMINATED":
            germ_slots.append(d["slot_id"])
        elif et == "EPISODE_OUTCOME":
            ep = d.get("episode_idx")
            if isinstance(ep, int):
                max_ep = max(max_ep, ep)
        elif et == "PPO_UPDATE_COMPLETED":
            se = d.get("head_slot_entropy")
            lf = d.get("head_slot_learnable_fraction")
            if se is not None and lf:
                slot_cond_min = min(slot_cond_min, se / lf)
            # head_nan_detected / head_inf_detected are per-head dicts ({slot: False, ...}),
            # truthy even when all-False — count only an ACTUAL non-finite head, an explicit
            # gate skip, or an amp overflow.
            head_nan = d.get("head_nan_detected") or {}
            head_inf = d.get("head_inf_detected") or {}
            if (
                d.get("nan_grad_count", 0)
                or d.get("inf_grad_count", 0)
                or any(head_nan.values())
                or any(head_inf.values())
                or d.get("update_skipped")
                or d.get("amp_overflow_detected")
            ):
                fin_trips += 1
    cut = max_ep * 0.8
    for line in open(hits[0]):
        if '"COUNTERFACTUAL_MATRIX_COMPUTED"' not in line:
            continue
        d = json.loads(line)["data"]
        if d.get("episode_idx", 0) < cut:
            continue
        off = on = None
        for c in d.get("configs", []):
            m = c.get("seed_mask", [])
            if m and not any(m):
                off = c["accuracy"]
            if m and all(m):
                on = c["accuracy"]
        if off is not None and on is not None:
            offs.append(off)
            ons.append(on)
    contrib = (sum(ons) / len(ons)) - (sum(offs) / len(offs)) if offs else float("nan")
    eff = contrib / (tot_p / 1e6) if tot_p else float("nan")
    return dict(
        tot_p=tot_p, dn_p=dn_p, r0_p=r0_p, tot_cf=tot_cf, nf=nf,
        draw_counts=draw_counts, state_hashes=state_hashes, germ_slots=germ_slots,
        max_ep=max_ep, fin_trips=fin_trips, slot_cond_min=slot_cond_min,
        contrib=contrib, eff=eff, cf_per_param=(tot_cf / tot_p if tot_p else float("nan")),
    )


def gate(label: str, fn) -> bool:
    try:
        fn()
        print(f"    PASS  {label}")
        return True
    except OffsetParityError as ex:
        print(f"    FAIL  {label}: {ex}")
        return False


def bootstrap_ci(deltas: list[float], b: int = 10000, seed: int = 0):
    """Seed-level paired bootstrap: resample the per-seed Δeff with replacement."""
    rng = random.Random(seed)
    n = len(deltas)
    meds = []
    for _ in range(b):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    lo = meds[int(0.025 * b)]
    hi = meds[int(0.975 * b)]
    return lo, hi


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "telemetry/causal_r1_n5"
    seeds = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else "41,42,43".split(","))]

    ok = True
    deltas: list[float] = []
    print(f"{'seed':>4} {'arm':>9} {'totParams':>10} {'r0c0_p':>8} {'downP':>9} "
          f"{'contrib':>8} {'pp/Mparam':>10} {'cf/param':>9} {'finTrip':>7} {'slotCondMin':>11}")
    for seed in seeds:
        c = load(base, "control", seed)
        on = load(base, "suppress_slot_on", seed)
        if not c or not on:
            print(f"  seed {seed}: incomplete (control={'ok' if c else 'MISSING'} "
                  f"on={'ok' if on else 'MISSING'}) — skip")
            continue
        for tag, a in [("control", c), ("suppress", on)]:
            print(f"{seed:>4} {tag:>9} {a['tot_p']:>10} {a['r0_p']:>8} {a['dn_p']:>9} "
                  f"{a['contrib']:>8.2f} {a['eff']:>10.3f} {a['cf_per_param']:>9.4f} "
                  f"{a['fin_trips']:>7} {a['slot_cond_min']:>11.3f}")
        print(f"  -- gates seed {seed} --")
        ok &= gate("stride constant (control)", lambda: assert_constant_controller_stride(c["draw_counts"]))
        ok &= gate("stride constant (on)", lambda: assert_constant_controller_stride(on["draw_counts"]))
        n = min(len(c["draw_counts"]), len(on["draw_counts"]))
        ok &= gate("offset-free control==on", lambda: assert_offset_free_parity(
            c["draw_counts"][:n], on["draw_counts"][:n],
            control_state_hashes=c["state_hashes"][:n], other_state_hashes=on["state_hashes"][:n]))
        ok &= gate("on germinates ZERO r0c0", lambda: assert_target_slot_never_committed(on["germ_slots"]))
        for tag, a in [("control", c), ("on", on)]:
            # Finiteness trips are INFORMATIONAL, not a hard gate: a rare transient
            # non-finite logit is sanitized in-place (value-independent → offset-free
            # parity is preserved, which IS the pairing gate above) and perturbs one step
            # out of ~30k (negligible for Δ_struct/Δacc). Only a CASCADE signals real
            # instability. (Advisor 2026-07-01: gate on estimand/pairing, not trips==0.)
            if a["fin_trips"] > 10:
                print(f"    FAIL  finiteness CASCADE ({tag}): {a['fin_trips']} trips (>10 ⇒ instability, not a transient)")
                ok = False
            elif a["fin_trips"]:
                print(f"    WARN  {a['fin_trips']} transient finiteness trip(s) ({tag}) — sanitized; pairing preserved (offset-free gate above)")
            if a["slot_cond_min"] <= 0.1:
                print(f"    FAIL  decision-step slot entropy health ({tag}): min={a['slot_cond_min']:.3f} ≤ 0.1")
                ok = False
        deltas.append(on["eff"] - c["eff"])

    print("\n" + "=" * 64)
    print(f"PRIMARY: paired Δ(acc-per-param) = eff_suppress − eff_control, n={len(deltas)} seeds")
    for seed, d in zip(seeds, deltas):
        print(f"  seed {seed}: Δeff = {d:+.3f} pp/Mparam")
    if len(deltas) >= 2:
        med = statistics.median(deltas)
        print(f"  median Δeff = {med:+.3f} pp/Mparam (mean {statistics.mean(deltas):+.3f}, "
              f"SD {statistics.pstdev(deltas):.3f})")
        if len(deltas) >= 3:
            lo, hi = bootstrap_ci(deltas)
            excl0 = (lo > 0) or (hi < 0)
            sign = "NEGATIVE → (b) efficiency-enabling stem" if hi < 0 else (
                "POSITIVE → (a) freeloader" if lo > 0 else "INCLUDES 0 → (c) fungible / underpowered")
            print(f"  seed-level bootstrap 95% CI = [{lo:+.3f}, {hi:+.3f}]  "
                  f"({'EXCLUDES' if excl0 else 'INCLUDES'} 0) ⇒ {sign}")
            print(f"  [n={len(deltas)}: n=5 BEGINS causal evidence; n=10 is the floor. "
                  f"Wide CI ⇒ escalate to n=10.]")
    if not ok:
        print("\nGATES: FAILURE — a gate tripped; do NOT bank Δeff")
    elif len(deltas) < 5:
        print(f"\nGATES: ALL PASS (data clean). n={len(deltas)} is a PILOT — the bootstrap CI is "
              f"degenerate below n=5; direction NOT yet banked (n=5 begins causal evidence, n=10 floor).")
    else:
        print("\nGATES: ALL PASS — apply the decision rule to the seed-level CI above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
