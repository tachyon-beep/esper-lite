# Current State — Esper        Checkpoint: 2026-07-02 (checkpoint #5 — GATE 2 PASSED; reviewer APPROVED; build awaits owner)

## The bet right now
**Reward credit-assignment redesign — the credit term for the enabling stem.** Every gate that is not the
owner's has now passed: GATE 1 (proxy-vs-estimand), **GATE 2 (learnability, PDR-0011 — retro-write delivery
MANDATED; terminal-flush killed by the pre-registered screen)**, the owed signal verification
(fossilize_contribution_scale multiplies the per-seed LOO — premise intact), and the reward-function-reviewer
(**APPROVE_WITH_CHANGES**, PDR-0012). Metric it moves: committed-J / corr(reward,J) once the term is A/B'd.

## In flight
- **Reward-credit term** (esper-lite-254175df90, in_progress): **blocked-on-owner — sign-off to START the
  default-OFF build** as scoped in PDR-0012 (retro-write + the reviewer's 5 build-conditioning changes;
  shapley_synergy_scale stays 0.0; enablement is a separate later gate: tau/PIN-E placebo + hard off-switch-J
  efficiency + fossilize-count gate + entrenchment monitor + ON-run dormancy recheck).
- **Telemetry hygiene** (esper-lite-425dcc4ca2): observability-emit follow-up (relabel raw → density, emit
  conditional) still open; unchanged this session.

## Open questions / blocked-on-owner
- **BUILD SIGN-OFF (new, PDR-0012 proposed):** approve starting the default-OFF Committed-Shapley build? All
  evidence gates passed; nothing is enabled without the later enablement gate.
- **n=10 vs bank-at-n=5 (MAGNITUDE, PDR-0009):** unchanged — direction banked (5/5, sign-test p≈0.03); only
  n=10 banks the effect size. Untouched by GATE 2.
- **Authority grant re-confirmation:** surfaced at the 2026-07-02 /own-product resume, unanswered (owner AFK);
  session proceeded under the standing grant (last reviewed 2026-06-28, within cadence).
- **metrics.md TARGET numbers** still `<owner-set>` placeholders.
- **Working-tree hygiene (owner's, not committed by the agent):** ~100 lines of phase-0 evidence updates to
  docs/analysis/2026-06-25-phase0-objective-and-instrumentation.md (+ CLAUDE.md/AGENTS.md/.gitignore edits)
  remain uncommitted from before this session.

## Last checkpoint did (checkpoint #5)
- **GATE 2 run end-to-end and PASSED (PDR-0011):** pre-registered three-tier probe (drl-expert design);
  Tier-2 paired campaign on local GPUs (5 same-seed pairs, K=30) — ΔP(FOSSILIZE|eligible) +0.081 median,
  5/5 seeds, p=0.0312, persistent; terminal-flush killed (0.0397σ_A < 0.05 bar). Verdict transfers to
  candidate B. Result: docs/analysis/2026-07-02-gate2-learnability-result.md.
- **reward-function-reviewer: APPROVE_WITH_CHANGES (PDR-0012, proposed)** — safe to build default-OFF;
  5 build-conditioning changes banked (incl. the divide_by_std unclipped-channel clamp).
- **Bug found + FIXED + closed (owner-directed): esper-lite-4fe98055f7** — use_telemetry=False silently
  disabled ALL blending/fossilization (gradient-health collection was telemetry-gated; G2 hard-fails
  unmeasured per KTS-001). Fixed in both trainers (c750c6b0) with a RED/GREEN-verified regression test;
  273 training tests pass. Historical telemetry-off runs were implicit no-morphogenesis ablations.
- Banked regime facts: gae_lambda=0.95 (run config, not leyline 0.98) ⇒ γλ=0.94525; the design doc's
  scaffold-rail routing corrected in the prereg + design-doc status header.

## Next session, start here
**The owner's build sign-off (PDR-0012).** If granted → implement the term per the design + the 5 reviewer
changes (retro-write delivery + GAE unit test first), default-OFF, on esper-lite-254175df90. If the owner
prefers, the n=10 magnitude run can proceed in parallel (independent of the build). Independent small item:
the telemetry-hygiene emit follow-up (esper-lite-425dcc4ca2).
