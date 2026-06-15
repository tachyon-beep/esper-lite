# Correctness Proof Strategy

> **For Claude:** Use this plan to decide what evidence is needed before claiming Esper is programmatically correct. Do not run the reward-efficiency exam as proof until each lower gate is green.

**Goal:** Separate four failure classes: bad instrumentation, bad training mechanics, bad comparison math, and bad underlying theory. A proof run must fail closed into one of those classes before it is allowed to say continue, revise, or stop.

**Architecture:** Treat correctness as a ladder. Karn/Nissa prove evidence integrity; precision and replay prove the run is attributable; Kasmina/Tolaria/Simic invariants prove lifecycle mechanics; proof baselines and micro-oracles prove the comparison math; the PPO oracle sandbox proves mechanics independent of learning; the final reward-efficiency exam tests the theory.

**Tech Stack:** Python 3.11, PyTorch, DuckDB/Karn MCP views, Nissa telemetry, pytest/Hypothesis, Filigree, Loomweave.

**Prerequisites:**
- Read `CLAUDE.md`, `README.md`, `ROADMAP.md`, `docs/coord/PLAN_TRACKER.md`, and `docs/plans/completed/2026-06-13-proof-confounder-drain.md`.
- Use Loomweave before broad reference edits.
- Do not use defensive bug-hiding patterns. Missing proof evidence must block, not default.
- Proof verdicts must cite precision-proven telemetry and must not cite quarantined pre-cutoff BF16 artifacts.

```yaml
# Plan Metadata
id: correctness-proof-strategy
title: Correctness Proof Strategy
type: planning
created: 2026-06-15
updated: 2026-06-15
owner: Codex

urgency: high
value: Gives Esper a programmatic path to decide whether a failure is instrumentation, mechanics, math, or theory rather than guessing from a noisy PPO run.

complexity: L
risk: high
risk_notes: The danger is overclaiming from incomplete evidence. Every stage must produce a machine verdict or block.

depends_on:
  - proof-confounder-drain
soft_depends:
  - morphogenesis-governor-integrity
  - proof-baseline-controls
blocks:
  - ppo-stability-oracle-sandbox
  - reward-efficiency
  - counterfactual-oracle

status_notes: Strategy artifact plus execution hardening. The proof packet emits typed machine verdicts, defaults CLI and programmatic callers to the reward-efficiency proof profile, rejects explicit final-exam precision/baseline opt-outs, and blocks missing precision, baseline provenance, outcome-empty controls, missing/mismatched fixed-schedule provenance, misplaced schedule metadata, missing/mismatched fixed-schedule realized morphology traces, missing/malformed/mismatched static-final source/replay topology manifests, static-final post-replay lifecycle mutations, and malformed lockstep reward A/B pairs for final-exam packets; normalizer checkpoint and HostSnapshot missingness bugs were hardened during strategy execution. Fixed-schedule runtime control is executable through action-mask forcing with a pinned schedule hash, static-final topology replay has materialization plus trainer/runner source-preflight handoff, and the live 2026-06-15 full-baseline rehearsal produced isolated run directories with complete control evidence. The rehearsal packet now fails closed as `BLOCKED_MECHANICS` on value-collapse/numerical-instability confounders; multi-seed statistical evidence and the oracle/PPO mechanics ladder remain open.
percent_complete: 74

reviewed_by:
  - reviewer: python-engineering
    date: 2026-06-15
    verdict: approved-with-changes
    notes: Keep gates executable and fail-closed; avoid prose-only acceptance.
  - reviewer: pytorch-expert
    date: 2026-06-15
    verdict: approved-with-changes
    notes: Precision provenance and normalizer resume contracts must be proof blockers.
  - reviewer: quality-engineering
    date: 2026-06-15
    verdict: approved-with-changes
    notes: Each rung needs focused regression tests plus a packet-level acceptance test.
```

---

## Verdict Taxonomy

`scripts/proof_packet.py` owns the machine verdict vocabulary:

- `BLOCKED_INSTRUMENTATION`: required telemetry is missing, malformed, or incomplete.
- `BLOCKED_PRECISION`: a run lacks proof-grade precision provenance.
- `BLOCKED_MATH`: outcome contracts or required comparison controls are invalid.
- `BLOCKED_MECHANICS`: training mechanics are unstable or confounded by value, gradient, rollback, reward-hacking, or degradation events.
- `REVISE_ALGORITHM`: clean evidence has a positive signal below the configured ROI threshold.
- `STOP_THEORY`: clean evidence has no positive ROI signal.
- `CONTINUE`: clean evidence meets the configured ROI threshold.

No other artifact may turn `BLOCKED_*` evidence into a product verdict.

## Proof Ladder

1. **Evidence Integrity Gate**
   - Required evidence: parse-clean JSONL, `TRAINING_STARTED`, `PPO_UPDATE_COMPLETED` learnability fields, `EPISODE_OUTCOME`, explicit missingness.
   - Current hardening: observation normalizer checkpoint metadata now carries the exact Obs V3 contract; `HostSnapshot.val_accuracy` preserves absence as `None`.
   - Blocker verdict: `BLOCKED_INSTRUMENTATION`.

2. **Precision and Reproducibility Gate**
   - Required evidence: every `TRAINING_STARTED` row records `amp_enabled`; AMP runs record `amp_dtype`; checkpoint resumes reject stale Obs V3 normalizer contracts.
   - Quarantine rule: pre-cutoff artifacts without precision provenance are not proof-grade.
   - Blocker verdict: `BLOCKED_PRECISION`.

3. **Mechanics Gate**
   - Required evidence: rollback cannot share a step with stale lifecycle mutation; value/ratio/gradient anomaly events absent; active-slot missing telemetry cannot look healthy; mutation event identity is traceable.
   - Main plan: `morphogenesis-governor-integrity`.
   - Blocker verdict: `BLOCKED_MECHANICS`.

4. **Math and Control Gate**
   - Required evidence: param ratio uses total/host semantics; required baseline controls are present with mode, pair ID, lifecycle policy, seed provenance, fixed-schedule provenance where applicable, fixed-schedule realized morphology traces where applicable, joined source-final and static-final replay topology manifests where applicable, and valid outcomes; lockstep reward A/B has two outcome-bearing sides with the same pair/seed and distinct reward modes; invalid outcomes fail closed; ROI is computed only from valid outcomes; final-exam profile gates cannot be silently disabled.
   - Current evidence: the live full-baseline rehearsal and `/tmp/esper-proof-baseline-packet-default.md` satisfy this gate and advance the proof to `BLOCKED_MECHANICS`.
   - Main plan: `proof-baseline-controls`.
   - Blocker verdict: `BLOCKED_MATH`.

5. **Oracle Sandbox Gate**
   - Required evidence: a hardcoded/oracle policy can drive the lifecycle to expected outcomes under the same Kasmina/Tolaria/Simic mechanics.
   - Purpose: prove the mechanics and reward accounting before PPO learning is blamed.
   - Expected output: a packet that can reach `CONTINUE` or `REVISE_ALGORITHM` without PPO instability.

6. **PPO Learning Gate**
   - Required evidence: PPO can learn the oracle-shaped task without value collapse, ratio collapse/explosion, or per-head learnability gaps.
   - Failure interpretation: if the oracle sandbox passes and PPO fails, revise algorithm/training, not theory.

7. **Reward-Efficiency Theory Gate**
   - Required evidence: precision-proven, control-complete, mechanics-clean reward-efficiency run on `cifar_impaired`.
   - Verdicts: `CONTINUE`, `REVISE_ALGORITHM`, or `STOP_THEORY`.

## Immediate Execution Plan

1. Keep the proof-packet verdict tests green:
   ```bash
   PYTHONPATH=src uv run pytest tests/scripts/test_proof_packet.py -q
   ```
2. Keep evidence-missingness tests green:
   ```bash
   PYTHONPATH=src uv run pytest tests/simic/test_reward_normalizer_checkpoint.py tests/karn/test_store_import.py tests/karn/test_collector_validation.py tests/karn/test_triggers.py tests/karn/test_health.py -q
   ```
3. ✅ Complete the proof-baseline-controls rehearsal:
   - source-run final topology manifest capture and handoff,
   - static-final replay manifest source fields,
   - fixed-schedule realized morphology trace,
   - lockstep reward A/B pair,
   - reward-efficiency profile default and fail-loud packet flag policy.
4. Promote `ppo-stability-oracle-sandbox` from tracker entry to a concrete planning artifact with:
   - deterministic task fixture,
   - hardcoded lifecycle policy,
   - expected mutation/rollback/reward trace,
   - proof-packet acceptance fixture.
5. Only after the lower gates are green, run the final reward-efficiency exam and accept the packet verdict as the strategic answer.

Current packet command shape for final-exam evidence:
```bash
PYTHONPATH=src uv run python scripts/proof_packet.py \
  --telemetry-dir telemetry/reward-efficiency-YYYY-MM-DD \
  --output docs/analysis/reward-efficiency-proof-packet.md \
  --proof-profile reward-efficiency
```

## Definition of Done

- A single proof-packet command emits one of the seven machine verdicts.
- Every `BLOCKED_*` verdict has a regression test that proves the run cannot accidentally produce a product verdict.
- Every clean product verdict has at least one fixture for `CONTINUE`, `REVISE_ALGORITHM`, and `STOP_THEORY`.
- No proof artifact cites a run with missing precision provenance.
- No proof artifact treats absent accuracy, absent normalizer contract, or absent learnability telemetry as measured evidence.
