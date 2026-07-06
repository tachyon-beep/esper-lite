# Current State — Esper        Checkpoint: 2026-07-06 (checkpoint #23 — B-deferred branch consolidation locked in [PDR-0038]; re-init fixed; working on feat/ev-stab-stage2-hra)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet), at **Stage-2 acceptance**
(esper-lite-2a4b56e719, in_progress). Unchanged this session — no code/experiment work; this was a
**workspace-continuity + branch-topology** session. Next code action is still the MAJOR-1 telemetry
wrapper (see "Next session, start here"). Metric it ultimately moves: host-accuracy contribution
(guardrail) via the Stage-2 MAJOR-1 acceptance gate (metrics.md).

## Branch / workspace state (NEW — the point of this session)
- **Where we work:** `feat/ev-stab-stage2-hra` (all live work is here: Stage-2, finished sanctum UI,
  entropy Phase-1, current Stage-0). This is the branch to boot on.
- **Decision (PDR-0038, B-deferred, winner-takes-all):** long-term **exactly ONE of
  {`feat/ev-stab-stage2-hra`, current `main`} survives; the other is jettisoned (archived).** The
  survivor is picked at **Stage-2 completion / acceptance verdict** — NOT now. `main` is the
  *current* trunk of record (origin, all provenance/checkpoints) but *current ≠ guaranteed survivor*.
- **Why deferred:** the two branches are 15 days / 101 vs 61 commits divergent (merge-base
  2026-06-21); a merge conflicts on 13 core `simic`/`tamiyo` files, and main's unique code is mostly
  drain-duplicates + parked-Shapley (`scale=0.0`) + superseded — dormant. Merging now = reconciling
  delicate RL code, mostly dormant, on a **broken test gate**, to then build the wrapper on it.
- **Re-init fixed:** `docs/product/` was vendored onto ev-stab (`6db68192`) so `/own-product`
  RESUMES here instead of bootstrapping. The **living workspace runs on ev-stab** during the defer;
  it goes wherever the survivor is at the pick.
- **Informed jettison (mandatory):** before the loser is discarded, run a **salvage audit** —
  enumerate its unique code, decide KEEP (cherry-pick into survivor) vs DROP (with reason), owner
  signs off the manifest. Jettisoning code is fine; jettisoning it *uninformed* is not. The archive
  ref is the backstop, not a substitute for the audit.
- **Safety net (archives):** `archive-main-2026-07-06` @`2d296b6c`, `archive-stage2-hra-2026-07-06`
  @`6db68192`, `backup/ev-stab-stage2-hra-2026-07-06` @`2fad1fc4`. The jettisoned branch survives as
  an archive ref.
- **Redundant now:** `feat/sanctum-layout` + `feat/entropy-thermometer-fix` (0 unique commits,
  absorbed into ev-stab) + their worktrees — prune at the survivor pick.

## In flight
- **Stage-2 acceptance (esper-lite-2a4b56e719, in_progress):** pure scorer DONE (42 tests, drl-verified
  core) on `feat/ev-stab-stage2-hra@2fad1fc4`. **Telemetry wrapper is the remaining half** and is
  REQUIRED before freeze: §1 validity gates, §0 provenance, burn-in `W` discard, §8B floored-update
  exclusion, §8E `Var(returns)` covariate, G3/G4, tier-gated packet CLI, §9 diagnostic scalars
  (`value_main_target_scale`, `cf_value_target_scale` — touches `ppo_agent.py`, ON-leg-only, byte-identity check).
- **Branch survivor pick + cleanup (esper-lite-1f1e55f58f, blocked by Stage-2):** future work item;
  gated on a runnable verification gate + drl/pytorch review; owner-gated to act.

## Facts the next session must not relitigate
- **Branch:** we continue on ev-stab; the survivor pick is DEFERRED to Stage-2 completion (PDR-0038).
  Do NOT re-open "which branch" or attempt the merge now. Do NOT delete `main`/ev-stab or push — owner-gated.
- **Verification gate is broken:** full `pytest` wedges on GPU `test_data_opt`/`test_dual_ab`; any
  merge/survivor claim needs a *runnable* gate (targeted tests + `train ppo` smoke), not "tests-green".
- **Stage-2 HRA ON-leg is code-complete** (cf value + reward wired; per-stream GAE + EV) — do not
  rebuild it or re-fear "cf trained on zeros" (guard exists). MAJOR-1 rubric core is drl-verified.
- Gate is **pure-scorer-only so far**; NOT freezable until the wrapper exists + is tested. No A/B before freeze.

## Open questions / blocked-on-owner
- **Nothing blocked-on-owner right now.** Authority grant holds as written (reviewed 2026-07-05).
- **Future owner gates (not now):** the survivor pick / any reconciling merge, jettisoning the loser,
  git history rewrite / branch deletion, and any push to origin (local `main` is 2 checkpoints ahead
  of `origin/main`, unpushed — a standing owner-gated push).
- Standing placeholders: north-star / rent / host-acc-floor TARGETs (metrics.md) still owner-set;
  `Δparam_max` (G2 ceiling) to be frozen before the ON run.

## Last checkpoint did (checkpoint #23)
- **PDR-0038** — B-deferred branch consolidation (winner-takes-all; continue on ev-stab; survivor at
  Stage-2 completion), with reversal triggers + integration gates. Fixed the `/own-product` re-init
  (vendored `docs/product` onto ev-stab). Created 2 archives + confirmed the earlier backup.
- Tracker: created **esper-lite-1f1e55f58f** (survivor pick + cleanup), blocked by the Stage-2 task.
  Stage-2 task stays in_progress. No roadmap horizon change (EV-stab stays Now). No new metric readings.
- Committed on `feat/ev-stab-stage2-hra` (the living workspace during the defer). NOT pushed.

## Next session, start here
**Build the MAJOR-1 telemetry wrapper** (the remaining half of the gate), TDD throughout, on
`feat/ev-stab-stage2-hra`: §1 validity + §0 provenance + burn-in `W` + §8B floored-exclusion + §8E
covariate + G3/G4 + packet CLI (mirror `scripts/proof_packet.py` — duckdb over Karn `ppo_updates`) +
§9 diagnostic scalars (byte-identity check on `ppo_agent.py`). Then **freeze the gate doc** → then
the paired fresh-init HRA ON/OFF A/B (owner launches). Pointers: gate doc
`docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`; scorer
`src/esper/simic/telemetry/stage2_acceptance.py`; memory `ev-stab-stage2-impl-state`.
