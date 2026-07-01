# Current State — Esper        Checkpoint: 2026-07-01 (checkpoint #3 — measurement-artifact reversal + J-reframe; n=5 J-read pre-registered)

## The bet right now
**Reward credit-assignment redesign — causal resolution of the (a)/(b) fork, measured on J
(acc-per-param), not accuracy.** Cheap-rescale closed (GATE −1 SURVIVE); harness built + validated
+ committed. The "entropy collapse" was a **MEASUREMENT ARTIFACT** (PDR-0006) — policy healthy, no
training fix. n=3 pilot leans **(b)**: r0c0 is an efficiency-enabling stem (suppressing it ~halves
system efficiency). Active work: the **n=5 J-read** (PDR-0007).

## In flight — the n=5 J-read (three phases)
- **Phase 0 — DONE.** Preserved 41–43 telemetry (`telemetry/causal_r1_n5/`); promoted + validated
  the J analyzer (`scripts/causal_contribution_j_analyze.py` — all gates pass, reproduces the
  pilot, median Δeff −7.0); pre-registered the run sheet
  (`docs/plans/ready/2026-07-01-n5-j-read-run-sheet.md`).
- **Phase 1 — GPU, OWNER GO PENDING (~13h).** Run seeds 44–45 × {control, suppress}, gpu_preload,
  committed harness; ≤2 concurrent (1/card). Commands in the run sheet.
- **Phase 2/3 — analysis + record.** Gates + paired ΔJ/acc-per-param (seed-level CI) → apply the
  pre-registered decision rule (negative CI → (b); includes 0 → (c); positive → (a); wide → n=10);
  durable result doc + PDR. · tracker: esper-lite-8190ff1c95
- **Telemetry hygiene** (esper-lite-425dcc4ca2): detector read-path fix LANDED (a84f9a78);
  observability emit (relabel raw → density, emit conditional) is the open follow-up.

## Open questions / blocked-on-owner
- **Phase 1 GPU go (~13h)** — the n=5 J-read's only blocker. *(blocked-on-owner)*
- **PDR-0004 estimand SCOPE — RESOLVED 2026-07-01:** owner ratified **total-system** (the run
  claims r0c0-specific system dependence; mechanistic/placebo DEFERRED). No longer blocking.
- **metrics.md TARGET numbers** still `<owner-set>` placeholders. *(blocked-on-owner)*
- **Advantage-pathology family** (PDR-0006 `does_not_fix`): op-conditioned Q, global adv-norm
  diluting sparse-head credit — queued next per advisor/ChatGPT review, after the J-read.

## Last checkpoint did (checkpoint #3)
- **Reversed the entropy-collapse premise:** diagnose-first ultracode workflow + verification →
  MEASUREMENT ARTIFACT, not training collapse (PDR-0006 supersedes PDR-0005); policy is healthy.
- **Reframed the fork discriminator to J / acc-per-param** (not accuracy); n=3 pilot leans (b).
- **Shipped the telemetry read-path fix** (a84f9a78; 435 tests green; PPO bit-identical).
- **Commissioned the n=5 J-read** (PDR-0007); Phase 0 done; created esper-lite-8190ff1c95.

## Next session, start here
**Phase 1 of the n=5 J-read** (on owner go): run seeds 44–45 per the run sheet, then Phase 2
(`uv run python scripts/causal_contribution_j_analyze.py telemetry/causal_r1_n5 41,42,43,44,45`)
→ apply the decision rule. If no GPU go yet, settle PDR-0004 estimand scope and/or start the
advantage-pathology queue in parallel.
