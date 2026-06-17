# 06 — Architect Handover

**Purpose:** Transition from *understanding* (this analysis) to *improvement planning*. This document prioritizes findings into actionable workstreams with effort sizing, sequencing, required reviewers, and risk gates. It is the bridge into `docs/plans/` and the Filigree tracker.

---

## Prioritization model

Priority = **blast radius × silence × correctness-impact**, capped by the project's own rules (No-Legacy means *delete*, don't shim; RL/PyTorch changes require specialist sign-off per CLAUDE.md).

| Tier | Meaning | Gate |
|------|---------|------|
| **P0** | Correctness/trust risk that can silently invalidate results | Fix before trusting any run output or scaling |
| **P1** | Rule violations + high-coupling debt that amplifies every future change | Fix before Phase 3 (TinyStories pivot) |
| **P2** | Maintainability / clarity debt | Opportunistic, before the relevant module is next touched |
| **P3** | Polish / hygiene | Backlog |

---

## P0 — Correctness & trust (do first)

### P0-1 · Resolve the `Q(s,op)`-as-baseline bias *(Q1+Q2)*
- **What:** The PPO baseline is action-conditioned; the op-head advantage subtracts a baseline containing the op being scored, and the GAE bootstrap is a single behaviour-sampled `Q(s',op')`.
- **Do:** Either train an op-independent `V(s)` head for the baseline/GAE (keep `Q(s,op)` as aux/telemetry), **or** use the marginal `V(s)=Σ_op π(op|s)·Q(s,op)` — the batched all-ops path at `ppo_agent.py:792-798` already evaluates every op, so this is cheap.
- **Acceptance:** explained-variance before/after; an ablation comparing op-head policy-gradient quality with vs without the op-independent baseline; documented decision record either way.
- **Effort:** M · **Reviewers:** `drl-expert` (mandatory) + `yzmir-deep-rl` skills · **Files:** `tamiyo/networks/factored_lstm.py:557-576,730-757`, `simic/agent/rollout_buffer.py:509-557`, `simic/agent/ppo_update.py:341-353`

### P0-2 · Remove the silent synthetic-data fallback *(Q3)*
- **What:** `except Exception: …return mock=True` substitutes random noise on any dataset-load failure; a run can complete on garbage with only a warning.
- **Do:** Let the load error propagate. If a mock dataset is genuinely needed for tests, make it a narrowly-typed, explicit `--mock-data` opt-in — never an auto-suppressed broad catch.
- **Acceptance:** a deliberately broken data root raises and aborts the run; no `mock=True` path reachable without an explicit flag.
- **Effort:** S · **Reviewers:** `axiom-python-engineering` · **Files:** `utils/data.py:~933-1005`

### P0-3 · Confirm/repair governor-rollback credit assignment *(Q9)*
- **What:** `mark_terminal_with_penalty` zeroes the **entire** episode's intermediate rewards on rollback. A long, mostly-good episode with a late panic loses all learning signal.
- **Do:** Decide if full-zeroing is intended. If not, scale the penalty or apply it to the last-k transitions while preserving earlier signal. Add a telemetry counter for rollback frequency + steps zeroed (frequent rollbacks would silently starve learning).
- **Effort:** S–M · **Reviewers:** `drl-expert` · **Files:** `simic/agent/rollout_buffer.py:797-814`

---

## P1 — Rule violations & high-coupling debt (before Phase 3)

### P1-1 · Eliminate silent `except Exception` swallowing in Nissa *(Q5)*
Replace bare catches with specific types or `except E: log(e); raise`. Shutdown paths must propagate enough signal that a failed backend is visible. **Effort:** S · **Files:** `nissa/output.py:221,264,273,662,702,736,810,923,930`, `scripts/train.py:1064,1083` · **Reviewer:** `axiom-python-engineering`.

### P1-2 · Populate or delete `grad_norm_host`/`grad_norm_seed` *(Q6)*
The collector already computes them (`simic/telemetry/gradient_collector.py:435-500`); wire them into `SignalTracker.update()` and the 23-dim base feature vector — **or** delete the dead observation fields. Do not leave silently-zeroed fields in the observation contract. **Effort:** S · **Reviewers:** `drl-expert` (it's an observation-space change) · **Files:** `leyline/signals.py:52-53`, `tamiyo/tracker.py:186-200`, `tamiyo/policy/features.py:296-307`.

### P1-3 · Split `vectorized_trainer.py` (CRITICAL god-file) *(Q4)*
Extract along the four named seams: `FusedValidationRunner`, `ActionInputBuilder`, `Epoch/BatchExecutor`, thin wiring `VectorizedPPOTrainer`. Re-export to preserve import sites; keep the no-cycle invariant. **Effort:** XL · **Reviewers:** `pytorch-expert` + `drl-expert` + `axiom-python-engineering:refactoring-architect` · **Risk:** behaviour-preserving refactor on the hot path — must be covered by tests first.

### P1-4 · Decompose `PPOAgent.update` (~900-line god-method) *(Q4)*
Extract the minibatch loop into `compute_losses`/`apply_grads`/`collect_update_metrics`; move CUDA/value telemetry into a `PPOUpdateTelemetry` collaborator and checkpoint I/O into its own module. **Effort:** L · **Reviewers:** `drl-expert` (mandatory). **Risk:** highest-risk edit surface in RL training.

### P1-5 · Tighten reward policy-invariance framing + close hacking surfaces *(Q8+Q17)*
Rewrite docstrings (`shaping.py`, `contribution.py:1001-1009`) to state invariance applies **only** to the PBRS term. Add a property test asserting the per-step PBRS sum over a full episode telescopes to `γ^T·φ(s_T)−φ(s_0)`. Add telemetry on prune-bonus frequency vs seed age (cadence-farming) and reconsider the asymmetric drip negative-clip. **Effort:** M · **Reviewers:** `drl-expert` + `yzmir-deep-rl`.

### P1-6 · Type the PPO metrics flow *(Q12)*
Replace the `dict[str,Any]` metrics bag threaded through `ppo_coordinator`/`emitters` with the already-existing `PPOUpdateMetrics` TypedDict end-to-end, so a renamed/dropped key fails loud rather than `.get`-defaulting. **Effort:** M · **Reviewers:** `axiom-python-engineering`.

### P1-7 · Resolve leyline→simic back-dependency *(Q10)*
Move `RewardComponentsTelemetry`/`ObservationStatsTelemetry` (they are telemetry *contracts*) into leyline, eliminating the function-local late imports and giving leyline true zero outbound domain deps. **Effort:** M · **Reviewers:** `axiom-python-engineering`.

### P1-8 · Delete confirmed dead code *(Q-dead)*
Remove `PerformanceBudgets`/`DEFAULT_BUDGETS` (`leyline/telemetry.py:150-172` + `__init__.py` exports) and `SlotConfigProtocol` (`karn/contracts.py:76`) after a confirming grep. **Effort:** S. *Do not* mass-delete the Loomweave dead-list (false-positive heavy).

---

## P2 — Maintainability (opportunistic, when next touching the module)

| ID | Action | Files | Effort |
|----|--------|-------|:-----:|
| P2-1 | Split `leyline/telemetry.py` → `leyline/telemetry/` package (events / payloads_ppo / payloads_seed / payloads_analytics), re-exported | `leyline/telemetry.py` | M |
| P2-2 | Handler-registry refactor of `SanctumAggregator`; per-subsystem snapshot assemblers | `karn/sanctum/aggregator.py` | L |
| P2-3 | Extract `SeedMetrics`/`SeedState`/`QualityGates` + a `GradientTelemetry` collaborator out of `slot.py` | `kasmina/slot.py` | L |
| P2-4 | Decompose `execute_actions` into parse→resolve→mutate→emit pipeline | `simic/training/action_execution.py` | L |
| P2-5 | Rename `vectorized.py`→`ppo_run.py`; move training primitives + metric aggregation to clarify the leaky boundary | `simic/training/` | M |
| P2-6 | Unify Karn's two `RewardComponents`/env models (store vs sanctum schema) onto one | `karn/store.py`, `karn/sanctum/schema.py` | M |
| P2-7 | Wire the orphaned MCP/DuckDB surface as a live OutputBackend (or document it as offline-only by design) | `karn/mcp/` | M |
| P2-8 | Move ~50 hand-calibrated reward constants into leyline; consolidate the 3 rent formulas | `simic/rewards/contribution.py` | M |
| P2-9 | Split `train.py` dispatch closure; de-duplicate preset/profiler blocks | `scripts/train.py` | M |
| P2-10 | Inject blend channel dims via HostProtocol instead of `nn.Linear(channels)` in `blending.py` | `kasmina/blending.py` | S |

---

## P3 — Hygiene / polish

- Add regression tests locking the **BPTT construction-time guard** (`ppo_agent.py:273-297`) and the **truncation-vs-terminal** handling — the two correctness gems that must not be silently removed.
- Fix stale dimension docstrings in `tamiyo/policy/features.py` ("30" → 31).
- Remove vestigial `self.lstm_ln = nn.Identity()` "backwards compat" no-op (`factored_lstm.py:367-369`) and the "backward compatibility" vocabulary in `karn` (get_snapshot, schema legacy fields).
- Add a HOLDING age/timeout fallback in the heuristic so a missing counterfactual can't stall a slot indefinitely (`tamiyo/heuristic.py:291-296`).
- Re-test `torch.compile` on `loss_and_correct` against PyTorch 2.9 inductor (currently deliberately eager).

---

## Suggested sequencing

```
Sprint 1 (trust):      P0-1, P0-2, P0-3   ← gate: no run output is trustworthy until done
Sprint 2 (rules):      P1-1, P1-2, P1-8, P1-7   (small, high-value rule fixes)
Sprint 3 (RL hardening): P1-5, P1-6   + P3 regression tests for the correctness gems
Sprint 4 (debt, pre-Phase-3): P1-3, P1-4   ← the two CRITICAL god-files (tests first!)
Ongoing:               P2 items as their modules are touched
```

**Phase-3 readiness gate (from ROADMAP):** the TinyStories pivot is blocked on the Phase 2.5 reward exam and Tamiyo-Next stability (explained variance not persistently negative; sparse-head entropy not collapsing). **P0-1 directly affects explained variance** and **P1-5 directly affects reward-exam validity** — both should land before that exam is considered conclusive.

---

## Reviewer routing (per CLAUDE.md)

| Workstream | Required reviewers |
|-----------|--------------------|
| Anything touching PPO/GAE/reward/value (P0-1, P0-3, P1-2, P1-4, P1-5) | `drl-expert` + `yzmir-deep-rl` |
| Hot-path / tensor / compile refactors (P1-3, P2-3) | `pytorch-expert` + `yzmir-pytorch-engineering` |
| God-file extractions (P1-3, P1-4, P2-*) | `axiom-python-engineering:refactoring-architect` |
| Observation-space changes (P1-2) | `drl-expert` (feature contract) |

**Plan-tracking:** Per CLAUDE.md, file these as plans under `docs/plans/ready/` and register them in `docs/coord/PLAN_TRACKER.md`; the P0 items warrant Filigree issues with a Phase-3 dependency edge.

---

## What this analysis did NOT do (scope for follow-up)

1. **Test-coverage map** — which god-methods are actually exercised (informs refactor safety for P1-3/P1-4).
2. **Runtime profiling** — confirm the unvectorized per-env GAE loop and the per-RL-step D2H sync are not real bottlenecks at target scale.
3. **Reward-mode production audit** — confirm which of the 7 modes are live vs experimental scaffolding (drives the No-Legacy deletion decision in P2-8).
4. **Planned domains** (Emrakul/Narset/Esika) — designed, not yet shipped; out of scope.
