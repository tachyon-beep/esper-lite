# 05 — Code Quality Assessment

**Inputs:** 5 specialist cross-cutting passes (drl-expert RL soundness, pytorch-expert GPU-first, python-code-reviewer commandments/rules audit, debt-cataloger god-files, dependency-map) + 10 validated domain reviews. Severity is by **blast radius**: how widely a defect propagates and how silently it fails.

---

## Scorecard

| Dimension | Grade | Evidence basis |
|-----------|:----:|----------------|
| **Macro-architecture (layering, coupling)** | **A** | 0 import cycles / 27,976 edges; leyline a clean fan-in-197 apex; protocol-based domain decoupling |
| **GPU-first / performance discipline** | **A−** | Inverted control flow confirmed; per-env streams + accumulate-then-sync; minor unvectorized GAE loop |
| **Telemetry contract** | **B+** | Strong typed-payload union, raise-on-bad-data; eroded by untyped intermediate metrics dict + a sensor gap |
| **RL algorithmic soundness** | **B** | Excellent recurrent-PPO rigor (BPTT guard, truncation handling, factored trust region) — but **Q(s,op)-as-baseline** is a genuine bias source |
| **Defensive-programming hygiene (CLAUDE.md)** | **B−** | Mostly fail-fast; but a **silent synthetic-data fallback** and pervasive `except Exception` in Nissa shutdown |
| **Module size / maintainability** | **C+** | 15+ files >1,000 LOC; two CRITICAL god-files (3,033-LOC trainer; ~900-line `update()`) |
| **No-Legacy compliance (CLAUDE.md)** | **B** | No shims/version-branches; but residual "backward compat" *vocabulary* + 2 confirmed dead symbols |

**Overall: B+ for a research platform.** The architecture is unusually disciplined for an RL codebase — the macro structure, GPU-first execution, and typed telemetry are genuinely strong and rare-to-get-right. Quality risk is concentrated in (a) one RL-correctness design choice, (b) a handful of silent-failure seams that violate the project's own stated rules, and (c) god-file maintainability debt in the hottest, riskiest modules.

---

## Critical findings (fix before scaling / before trusting results)

### Q1 — Action-conditioned critic Q(s,op) used as the PPO baseline *(drl-expert, HIGH)*
**The single most important correctness finding.** The critic concatenates a one-hot of the sampled `op` into the value-head input (`factored_lstm.py:557-576`), and `evaluate_actions` uses the **stored** op as the baseline that feeds GAE (`rollout_buffer.py:509-557`) and the value loss (`ppo_update.py:341-353`).

Standard PPO requires the baseline `b(s)` to be **independent of the action whose advantage it scores**. Here `A^op = r + γ·V(s') − Q(s, op_taken)` subtracts a baseline that already contains `op_taken`, so for the **op head** the surrogate is no longer an unbiased advantage-weighted policy gradient — part of the op's own value is cancelled out of its advantage. (The other 7 heads are scored against an op-conditioned baseline that *is* independent w.r.t. them, so they're far less affected.)

Compounding it (**Q2**): the GAE bootstrap `V(s')` is a **single behaviour-sampled** `Q(s', op'~π_old)` (`factored_lstm.py:730-757`), i.e. a one-sample estimate of `E_op[Q]` that becomes stale/off-policy after the first epoch — injecting policy-dependent variance into the value target.

**Fix:** either (a) train a true op-independent `V(s)` for the baseline/GAE and keep `Q(s,op)` as a telemetry/aux head; or (b) use the marginal `V(s) = Σ_op π(op|s)·Q(s,op)` (the batched all-ops path at `ppo_agent.py:792-798` already evaluates all ops at once — cheap, NUM_OPS small). Validate with explained-variance + a V(s)-baseline ablation. *This is a deliberate design choice, not a bug — but it must be documented and validated, because the op-head policy gradient is the most likely place to see silent bias.*

### Q3 — Silent fallback to synthetic random data *(support domain + commandments audit, HIGH)*
`get_cifar10_datasets`/`load_cifar10` (`utils/data.py:~933-1005`) wrap CIFAR10 construction in `except Exception: warnings.warn(...); return …(mock=True)`, substituting `torch.randn/randint` noise. **A full training run can complete on garbage** with only a warning — making accuracy telemetry and any "proof" meaningless. Textbook bug-hiding per CLAUDE.md. **Fix:** let the failure propagate, or make the mock a narrowly-typed, explicit opt-in flag — never an auto-suppressed broad-catch fallback.

### Q4 — Two CRITICAL god-files on the riskiest paths *(debt-cataloger, CRITICAL ×2)*
- `vectorized_trainer.py` (3,033 LOC, ~80-field class, 200–466-line methods). Split along its four already-named seams: `FusedValidationRunner`, `ActionInputBuilder`, `Epoch/BatchExecutor`, thin wiring trainer. Effort **XL**.
- `PPOAgent.update` (~900-line single method, `ppo_agent.py:526`). Extract the minibatch loop into named steps (`compute_losses`/`apply_grads`/`collect_update_metrics`); move CUDA/value telemetry + checkpoint I/O out. Effort **L**. *Highest-risk edit surface in the RL loop — require drl-expert review.*

---

## High-severity findings

| ID | Finding | Location | Rule/Commandment |
|----|---------|----------|------------------|
| Q5 | `except Exception:` swallowing errors without re-raise, pervasive in telemetry/shutdown | `nissa/output.py:221,264,273,662,702,736,810,923,930`; `scripts/train.py:1064,1083` | No-Defensive (VIOLATED) |
| Q6 | `grad_norm_host`/`grad_norm_seed` declared in `TrainingMetrics` but **never populated** (silently 0.0) — collector computes them but they never reach the observation | `leyline/signals.py:52-53`; `tamiyo/tracker.py:186-200` | C1 Sensors (PARTIAL) |
| Q7 | Two parallel, divergent telemetry data models (`store.py` vs `sanctum/schema.py` both define `RewardComponents`/env state) | `karn/store.py:236-331` vs `karn/sanctum/schema.py` | single-source-of-truth |
| Q8 | Reward layered with large non-potential dense terms on top of PBRS — only PBRS is policy-invariant; docstrings can be read as claiming invariance for the whole reward | `contribution.py:816-863, 653-684, 1295-1329` | C2 / reward soundness |
| Q9 | Governor rollback **zeroes the entire episode's** intermediate rewards (coarse credit assignment — a long good episode + late panic loses all signal) | `rollout_buffer.py:797-814` | reward soundness |
| Q10 | leyline imports back into simic (the only outbound dep; function-local late imports admit the cycle) | `telemetry.py:610,1947,1968` | layering purity |
| Q11 | God-files (cluster): `slot.py` 2,831 · `leyline/telemetry.py` 2,524 · `aggregator.py` 2,277 · `schema.py` 1,717 · `contribution.py` 1,514 · `factored_lstm.py` 1,394 · `action_execution.py` 1,634 | multiple | maintainability |

---

## Medium-severity findings

- **Q12 — Untyped PPO metrics bag.** Metrics flow as `dict[str,Any]` through `ppo_coordinator`/`emitters` with `metrics.get(key, default)`; a renamed/dropped key silently degrades. The payload is typed only at the final emit boundary. *(simic-training, partial Telemetry)*
- **Q13 — SIMPLIFIED reward omits structural rent** yet shares the dispatcher with rent-bearing modes; a SIMPLIFIED run silently does not exercise C2. Self-declares non-evidence, but the dispatcher treats it identically. *(simic-rewards)*
- **Q14 — Reward-mode proliferation:** 7 modes, 3 different rent formulas, ~50 hand-calibrated magic floats embedded in code (should be in leyline). *(simic-rewards)*
- **Q15 — MCP/DuckDB persistence surface ORPHANED** from live training (post-hoc CLI only). *(karn)*
- **Q16 — `blending.py` instantiates `nn.Linear(channels,…)`** making `BlendAlgorithm` implicitly host-shape-aware — bypasses HostProtocol. *(commandments audit, C5 partial)*
- **Q17 — Residual reward-hacking surfaces:** PRUNE economy cadence-farming on ≥3-epoch cadence; asymmetric drip negative-clip under-penalizes post-fossilization degradation. *(drl-expert)*
- **Q18 — `train.py` god-file** with duplicated preset-selection chains and profiler-kwargs blocks (override-applied-in-one-path-only risk). *(support)*

---

## Confirmed dead code (delete per No-Legacy)

- **`PerformanceBudgets` + `DEFAULT_BUDGETS`** — `leyline/telemetry.py:150-172` (has a `# TODO: [DEAD CODE]` admitting it; still exported in `__init__.py:744-745,1087-1088`). No production consumer.
- **`SlotConfigProtocol`** — `karn/contracts.py:76` (no callers outside its own file).

> ⚠️ Loomweave's `entity_dead_list` reports 1,306 candidates, but in `src/esper` proper these are **almost all false positives** (test-mock closures, blueprint factory locals, attribute-accessed `Threshold` classes). Static reachability cannot see attribute access / dynamic dispatch. **Do not mass-delete** — confirm each with a repo-wide grep first.

---

## Genuine strengths (preserve these — they are the platform's spine)

`★ Insight ─────────────────────────────────────`
These are not "no issues found" placeholders — the specialists flagged them as *positively* engineered, addressing failure modes that most RL codebases get wrong.
`─────────────────────────────────────────────────`

1. **BPTT hidden-state leakage guarded at construction** (`ppo_agent.py:273-297` raises if `max_steps_per_env > chunk_length`). The single most common recurrent-PPO correctness bug — handled with rigor. *Add a unit test asserting the ValueError fires so the guard can't be silently removed.*
2. **Correct truncation-vs-terminal handling** (`action_execution.py:1364-1365`): fixed-epoch episode ends are treated as time-limit *truncations* that bootstrap `V(s_T+1)`, not as true terminals (which would bias late-episode values toward 0). Theoretically correct (Pardo et al. 2018) and rarely done right. The only true terminals are governor rollbacks.
3. **Per-head factored trust region** (`ppo_update.py:276-338`): each head's ratio clipped independently and surrogates summed, with a documented ruling on why clipping the *joint* ratio would couple heads. Correct factored-PPO formulation.
4. **Layered entropy-collapse defenses** using *availability* masks (regularize sparse heads whenever the action was *valid*, not only when chosen) + hard probability floors + scheduled penalty. Directly addresses the sparse-head death-spiral.
5. **GPU-first execution is real, not aspirational** — confirmed by reading the sync/stream code: single sequential thread, no queue/lock between host-train and policy, accumulate-then-sync, deferred async gradient telemetry, `set_to_none=True`, `loss.detach()`. No retained-graph or memory-growth bugs found.
6. **Typed telemetry contract** — discriminated payload union, isinstance-discriminate, raise on bad data, absent≠zero discipline, SanctumBackend latches the first failure into a fatal modal rather than swallowing.
7. **Zero import cycles** with visible lazy-import discipline; leyline centralization is intentional and single-sources the contracts.

---

## Quality risk map (where a careless change does the most damage)

```
                 high blast-radius
                        ▲
   leyline/telemetry.py │ ● Q(s,op) baseline (Q1)   ● vectorized_trainer.py (Q4)
   (fan-in 197)       ● │ ● PPOAgent.update (Q4)
   contribution.py      │ ● synthetic-data fallback (Q3)
   (fan-in 165)       ● │ ● reward policy-invariance (Q8)
   aggregator.py      ● │   ● untyped metrics bag (Q12)
   (fan-in 141/133)     │
   ──────────────────────────────────────────────►
   low silence                              high silence (fails quietly)
```
Top-right quadrant = highest priority: widely-coupled **and** fails silently. Q1, Q3, Q4, and Q8 all sit there.
