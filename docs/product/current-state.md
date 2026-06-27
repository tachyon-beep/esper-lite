# Current State — Esper        Checkpoint: 2026-06-28 (resume #1 — bootstrap workspace persisted)

## The bet right now
**Reward credit-assignment redesign — Phase 0 (instrument-first).** Make J a computable
post-hoc yardstick and decide rescale-vs-redesign via the GATE −1 cheap-fix falsifier
sweep. Metric it moves: committed-J / corr(reward,J).

## In flight
- **GATE −1 cheap-fix falsifier sweep** — 16/25 runs done (control 5/5, unitnorm 5/5,
  clip2 5/5; clip5 + escrow finishing), **9 left + a placebo batch owed**; box alive and
  mid-sweep (latest run `telemetry_2026-06-27_231925` still appending past midnight),
  ~1 day wall-clock at 2/2 GPUs, self-driving. · tracker: esper-lite-a221da47ea (Phase −1),
  esper-lite-3d67b09687 (Stage-0 instrument)
- **The (a)/(b) synergy fork** — OPEN. n=5 co-resident synergy shows complementarity is
  real (mechanism exists, pro-(b)); the closer (does the *committed* neg-blendΔ early-conv
  cohort enable?) is **blocked on seed-identity reconstruction** (see open questions).
- **EV-stabilization** epic (Next). · tracker: esper-lite-f25b71c165

## Open questions / blocked-on-owner
- **(a)/(b) fork** (freeloader defect vs LOO-undervalued enabling stem) — decides
  redesign-vs-remeasure; opposite fixes. Needs the **seed-lifecycle-conditioned synergy**,
  which is blocked: `(env,slot)` is not single-occupancy (multi-candidate germination) and
  structured germ/foss events carry no stable `seed_id` — needs seed-identity
  reconstruction (likely morphogenesis schema / drl-expert). Detail: evidence packet §5.3.
- **metrics.md TARGET numbers are still `<owner-set>` placeholders** — *highest-value owner
  input.* No acceptance/kill decision can fire until the committed-J north-star and the
  guardrail floors carry a number + date (a target with no number is not falsifiable).
  Same for the inferred **vision.md audience/secondary**. *(blocked-on-owner)*
- **Escrow config fix uncommitted** (`configs/config-3slot-3seed-baseline-escrow.json`,
  n_envs 128→12) — awaiting owner word to commit; deliberately excluded from the workspace
  commit. *(blocked-on-owner)*
- **Dropped-events completeness check owed** before any paired GATE −1 verdict (several
  runs exited non-zero with DirectoryOutput trailing-event loss — data integrity, not
  config contamination).

## Last checkpoint did
- RESUME #1 (`/own-product` → `/product-checkpoint`): **clean resume, no drift** — HEAD
  unchanged (406aeb25), all 4 in-flight tracker IDs still `open`, box alive mid-sweep.
- **Authority grant re-confirmed as written** (Default research grant; Last reviewed
  2026-06-28) — no change.
- **Persisted the bootstrap workspace** — the 5 artifacts + PDR-0001 were untracked since
  bootstrap; this is their first commit. No new PDR (no new product decision this session).
- Note: Karn MCP binding returned empty this session — run-state verified by filesystem,
  not Karn queries, until it is pointed at the run dir.

## Next session, start here
**The seed-identity reconstruction for the lifecycle-conditioned synergy closer** — the
actual answer to the (a)/(b) fork (validate the fate-distribution before interpreting any
synergy). In parallel, the box finishes the arm sweep + placebo, then paired-bootstrap
GATE −1 scoring (resample the ≥5 seeds as the unit, NOT the 12 vec-envs — the
pre-registered invalidator). Substantive detail lives in
`docs/analysis/2026-06-25-phase0-objective-and-instrumentation.md` and project memory.
