# Telemetry Improvement Program (TIP)

```yaml
# Plan Metadata
id: telemetry-improvement-program
title: Telemetry Improvement Program — self-defending experimental reads
type: concept
created: 2026-07-09
updated: 2026-07-09
owner: john (captured by product owner-agent, PDR-0049)

urgency: high        # queued behind Stage-2 read; Phase A is the next infra bet after it
value: >
  Make every future experimental verdict traceable from typed event -> view ->
  metric definition -> validity gate -> PDR decision, with population and
  denominator explicit. Not "more logs" — fewer mis-scored, misinterpreted, or
  overclaimed reads.

complexity: XL       # program of 4 phases; individual phases are M/L
risk: medium
risk_notes: >
  Main risks are (a) duplicating machinery the Stage-2 harness already built
  (validity gates, provenance, byte-identity proofs) instead of generalizing it,
  and (b) telemetry-contract churn colliding with frozen training paths. Phase A
  is read-path/scorer-side only and freeze-safe; anything touching emitters or
  obs schema waits for the Stage-2 ON wave to complete (same-commit discipline).

depends_on: []       # Phase A startable during the Stage-2 freeze (scorer/Karn side only)
soft_depends:
  - stage2-major1-acceptance      # Phases B+ reconcile against what S1-S7 landed
blocks: []

status_notes: >
  CAPTURED as intent (PDR-0049) from an owner-provided nominal plan, adapted.
  Not implementation-ready: needs per-phase specialist review at promotion.
percent_complete: 0

# Expert Review (REQUIRED before promotion to ready)
reviewed_by: []      # at promotion: drl-expert (metrics semantics), pytorch-expert
                     # (per-site host probes), ordis-quality (regression suite)
```

## Program thesis

Tamiyo's telemetry is good enough to *discover* real effects but not yet good
enough to make every read *self-defending*. The recent experimental arc
(Shapley A/B, Stage-0, Stage-2 prep) kept hitting the same failure classes:

- wrong denominator / wrong population (all-step entropy read as decision
  entropy — the PDR-0006 false collapse)
- glob/directory contamination (stale dirs, aborted `on_s4*` matches,
  partial-run quarantines: `aborted-launch-1/`, `killed-by-session-reap-1/`)
- metric naming drift (G2 "fossilized params" that was terminal footprint;
  `episode_idx` per-env vs dense-global)
- single-env terminal reads; missing provenance columns; schema/view mismatch
- outcome-vs-mechanism read confusion on partial runs
- **metrics that read as constants under the current config** (new scar
  2026-07-09: `clip_fraction`/`ratio_*` are identically 0/1.0 because
  `recurrent_n_epochs=1` makes the PPO ratio degenerate — any gate or dashboard
  consuming them reads noise-free nothing)

**Goal:** every future verdict can state, mechanically: which runs were scored
and excluded, which metric definitions were used, which populations and
denominators, which validity gates passed, and whether a null was informative
or bandwidth-bound.

## Non-negotiable rules (program principles)

1. No glob-based scoring — manifest-driven only.
2. Every experiment has an explicit manifest.
3. Every metric has a machine-readable definition (units, denominator,
   population, source events, invalid-if conditions).
4. Every scorer runs validity gates before interpretation (`VALIDITY:
   PASS/FAIL/NULL` precedes any effect table).
5. Every telemetry-contract change gets round-trip tests through the
   production view the scorer reads (the "half-wired scar" rule).
6. Every read-path-only change ships an identity/no-behaviour-change proof
   (already practiced: byte-identity goldens; make it a checklist item).
7. Every mid-run monitor is classified mechanism-health or outcome-peek;
   outcome-peek is embargoed until scoring.
8. Every partial/aborted/crashed run is quarantined AND manifest-excluded
   (visible, impossible to score accidentally).
9. Every null verdict states informative vs bandwidth-bound (the PDR-0026
   lesson).
10. Every metric report names denominator, population, and units — including
    chart labels (the Sanctum "slot entropy, decision-eligible steps only"
    rule).
11. **(added)** Every registry metric declares the config conditions under
    which it is informative (e.g. ratio/clip metrics require
    `recurrent_n_epochs > 1`; EV_main requires non-degenerate
    `Var(R_main)`). A metric consumed outside its validity envelope is a
    validity-gate failure, not a datum.

## Phases (intent — sequencing via /axiom-program-management at commitment)

### Phase A — "stop hurting ourselves" (scorer/read-path; startable during freeze)
- **A1 Run manifest + manifest-enforced scoring.** `run_manifest.json` per
  experiment (arms, seeds, run_dirs, commit, config hash, status,
  excluded_runs+reasons). Scorers refuse: unmanifested dirs under the
  experiment root, incomplete manifest-valid runs, duplicate seeds per arm,
  commit/config mismatch, excluded runs matching legacy globs.
- **A2 Metric registry** (machine-readable YAML): source events, population,
  denominator, units, formula, invalid-if, validity envelope (rule 11),
  owner. Stage-2 acceptance metrics enter first. Names distinguish raw /
  conditional / normalized / post-normalized variants.
- **A3 ValidityReport layer.** Generalize the Stage-2 wrapper's §1 gates
  (S3 `validate_pair`) into a reusable pre-scoring object: run_complete,
  expected episode/env coverage, terminal rows per env, tracebacks,
  non-finite events, commit/config/seed/reward-mode/schema-version match,
  manifest status, required columns, no-excluded-run-used.
- **A4 Historical-scar regression suite.** Fixtures for the known bug
  classes (14 from the arc + the constant-metric scar); a scorer or contract
  change that reintroduces one fails CI.
- **A5 Mechanism-health vs outcome-peek classification file** + labelled
  monitor outputs (`MECHANISM-HEALTH ONLY` / `OUTCOME-PEEK — BLOCKED`).

### Phase B — critic/value-target telemetry reconciliation (post-ON-wave)
Audit-then-gap-fill, NOT greenfield: much of the nominal list already exists
in `PPO_UPDATE_COMPLETED` (ev/ev_return_variance, return_var_{cf,main,residual}
shares, value_nrmse, v_return_correlation, pre/post-norm advantage std,
per-head entropies incl. choice-conditional, learnable fractions, gradient
states) and `actor_advantage_source` provenance is S7 territory. Gaps to
close: per-head approx-KL (registry-marked conditionally-informative),
per-stream target scales as first-class columns, guard-channel rate readers,
volatility/IQR series as registry metrics rather than notebook conventions.

### Phase C — seed lifecycle identity (post-ON-wave)
Stable `seed_uid` across germinate -> select -> on-path -> stage changes ->
prune/fossilize -> residency; event set completed so lifecycle joins need no
regex; fate-distribution validation before any lifecycle analysis. Unlocks
the r0c0 freeloader/enabler class of questions properly.

### Phase D — observation-schema programme (per-site host perception)
The obs-signal deficit is *localization*: per-slot features describe the seed,
not the host tissue; dormant slots are observationally symmetric. Sequence:
1. **D1** emit per-site static+dynamic host descriptors (activation stats,
   local gradient norm, dead-channel fraction, saturation, grad-flow
   proxies) as telemetry only — `OBSERVED_BUT_NOT_POLICY_VISIBLE`.
2. **D2** offline information-ceiling probes on logged rollouts: predictors
   from (host-only | slot-only | full ObsV3 | ObsV3+site features) to
   (future return, J, fossilize success, terminal efficiency). Discriminates
   "information absent from obs" vs "critic fails to extract it" — the
   direct answer to the 2026-07-09 obs-ceiling question.
3. **D3/D4** Obs V4 proposal and A/B ONLY if the probes show signal value.
   Never silently add to the policy observation mid-experiment.
This phase merges with the research-verdict "escrow obs-feed" line — one
observation programme, not two.

### Explicitly folded elsewhere (not TIP deliverables)
- Dashboards/experiment-card + denominator-labelling → the Sanctum
  tri-domain review line (esper-lite-5fab041f3a) adopts rule 10.
- Counterfactual/Shapley calibration telemetry (tau placebo legs, coalition
  observability) → stays with the parked Shapley line's reversal triggers
  (PDR-0026/0027); TIP provides the registry/validity substrate it would use.
- GPU/runtime provenance: mostly landed (allocator stats, cadence,
  dataloader_wait_ratio, memory peaks); remaining gap = co-tenancy/packing
  descriptors, folded into A1's manifest fields.

## Success criteria

A future experimental report can state mechanically: exact run set scored,
exact exclusions, exact metric definitions, populations/denominators, validity
outcomes, informative-vs-bandwidth-bound nullity, no mid-run outcome read
influenced the decision, schema versioned and reproducible. The standard the
last three months converged on manually becomes the default.

## Provenance

Captured 2026-07-09 from an owner-drafted nominal plan (external-assistant
assisted), adapted per owner instruction "don't assume it will be exactly
this" — see PDR-0049 for the judgment applied (dedup against landed Stage-2
machinery, validity-envelope rule added, dashboard/Shapley/GPU items rehomed,
Phase-A freeze-safety noted). Tracker epic: see PLAN_TRACKER / PDR-0049.
