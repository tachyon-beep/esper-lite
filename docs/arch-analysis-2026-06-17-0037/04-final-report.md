# 04 — Final Report

**Esper-lite architecture archaeology** · 2026-06-17 · git `e259169` (Loomweave index) on branch `0.1.1`
**Deliverable scope:** Option C (Architect-Ready) · all 8 active domains
**Method:** ultracode multi-agent workflow — 10 domain reviews → 10 independent validations (3 PASS / 7 WARN / 0 BLOCK) → 5 specialist cross-cutting passes (RL soundness, GPU-first, policy/rules audit, tech-debt, dependency map). 25 agents, ~2.3M tokens.

---

## Executive summary

Esper-lite is a **morphogenetic neural-network training platform**: a recurrent PPO agent (Tamiyo, trained by Simic) controls *another network's training process*, deciding when to grow, blend, fossilize, or prune "seed" modules in a host network. Its defining property is a **nested-loop meta-RL design** — each RL step *is* one host training epoch, and the RL "environment" is literally a neural network mid-training.

The verdict: **this is an unusually disciplined research codebase.** The macro-architecture is genuinely strong — zero import cycles across ~28K edges, a clean contracts hub (leyline), protocol-based domain decoupling, a confirmed GPU-first non-blocking control flow, and a typed telemetry contract. The system's stated design principles (the ROADMAP "Nine Commandments") are, for the most part, *real in the code*, not aspirational.

The risks are concentrated and identifiable: (1) **one RL-correctness design choice** — an action-conditioned `Q(s,op)` critic used as the PPO baseline — that can silently bias the op-head policy gradient; (2) **a few silent-failure seams** that violate the project's own hard rules (most seriously, a fallback to synthetic random data that lets a run complete on noise); and (3) **god-file maintainability debt** concentrated in the hottest, highest-coupling, RL-critical modules.

`★ Insight ─────────────────────────────────────`
The headline tension of this codebase: **clean *between* modules, heavy *within* the hot ones.** The layering discipline (0 cycles, leyline apex) is textbook. But the same effort did not extend inside the load-bearing modules — `vectorized_trainer.py` (3,033 LOC) and `PPOAgent.update()` (~900-line method) are exactly where RL bugs are silent and edits are riskiest. The architecture passed the "draw the boxes" test brilliantly and the "open the box" test only partially.
`─────────────────────────────────────────────────`

---

## What the system is (one paragraph)

Eight domains, each a biological "organ": **Kasmina** (stem cells — the host model + seed lifecycle FSM), **Tamiyo** (brain — the LSTM/heuristic decision policy), **Simic** (evolution — PPO, the reward economy, and the vectorized trainer), **Tolaria** (metabolism — the host-training execution layer + the safety Governor), **Nissa** (senses — the telemetry bus), **Karn** (memory — TUI/web dashboards, analytics store, DuckDB SQL), and **Leyline** (DNA — the shared contracts every domain imports). Three more (Emrakul, Narset, Esika) are designed but not yet shipped. ~72K LOC; Simic (23.6K) and Karn (22.5K) are 64% of it.

---

## Constitutional audit — does the code uphold its own "Nine Commandments"?

The ROADMAP states nine falsifiable architectural claims. Measured against the code (with the CLAUDE.md hard rules tracked separately):

| # | Commandment | Verdict | Evidence |
|---|-------------|:-------:|----------|
| 1 | Sensors match capabilities | ⚠️ **Partial** | Rich gradient/landscape sensors, but 6 event types have no structured Nissa handler, and `grad_norm_host/seed` are declared-but-never-populated |
| 2 | Complexity pays rent | ✅ **Upheld** | Param-ratio rent wired into every economy mode; SPARSE is literal accuracy−rent. *Caveat: SIMPLIFIED mode omits rent by design* |
| 3 | GPU-first / inverted control flow | ✅ **Upheld** | **Confirmed by reading the sync/stream code** — single sequential thread, no queue/lock between host-train and policy, accumulate-then-sync |
| 4 | Progressive curriculum | — | Out of code scope (training-program design) |
| 5 | Train Anything (HostProtocol) | ✅ **Upheld** | Seed engine imports only generic `torch.nn`; host bound as `HostProtocol`. *Minor: `blending.py` instantiates `nn.Linear(channels)` — host-shape-aware* |
| 6 | One morphogenetic plane | ✅ **Upheld** | Single `MorphogeneticModel`, one ModuleDict, one `rNcN` coordinate space; legacy slot names rejected |
| 7 | Governor prevents catastrophe | ✅ **Upheld** | `TolariaGovernor` constructed unconditionally; **no enable/disable/bypass flag exists anywhere**; panic decisions use only loss stats; policy supplies zero input |
| 8 | Hierarchical scaling | — | Future (Narset/Emrakul not shipped) |
| 9 | Frozen Core economy | — | Future (PEFT/adapter strategy) |

**CLAUDE.md hard rules:** *No-Legacy* — ✅ largely upheld (no shims/version-branches; but residual "backward compat" *vocabulary* in Karn/Tamiyo + 2 confirmed dead symbols to delete). *No-Defensive-Programming* — ⚠️ **violated** in two places (synthetic-data fallback; `except Exception` swallowing in Nissa shutdown). *Telemetry-as-contract* — ✅ strong at the wire, eroded by an untyped intermediate metrics dict.

**Bottom line:** every testable, *shipped* commandment is upheld or upheld-with-a-named-caveat. The two clear rule violations are both **silent-failure** patterns, which is the most dangerous kind in a system whose entire purpose is to produce trustworthy measurements.

---

## The five things that matter most

1. **`Q(s,op)`-as-baseline (RL correctness).** The PPO baseline is action-conditioned, violating the action-independent-baseline requirement; the op-head policy gradient can be silently biased, compounded by a single-sample GAE bootstrap. *Highest-impact finding.* Fix: train a true `V(s)` or use the marginal `Σ_op π(op|s)Q(s,op)` (the batched path already computes all ops). Validate with explained-variance + ablation.

2. **Silent synthetic-data fallback.** A dataset-load failure auto-substitutes random noise; a run can reach "completion" on garbage. Fix: fail loud or make mock an explicit opt-in.

3. **Two CRITICAL god-files** (`vectorized_trainer.py` 3,033 LOC; `PPOAgent.update` ~900-line method) on the riskiest RL paths. They have clean extract-class seams already named in the code.

4. **Reward policy-invariance framing.** Only the PBRS component is policy-invariant; large dense terms (fossilize/terminal bonuses, prune economy) intentionally reshape the optimum. Tighten docstrings so "PBRS preserves optimal policy" is never read as "this reward preserves optimal policy", and watch the residual reward-hacking surfaces (cadence-farming, asymmetric drip).

5. **Telemetry typed-contract erosion.** The contract is strong at the emit boundary but flows through an untyped `dict[str,Any]` metrics bag with `.get(key, default)`; a renamed key degrades silently. Plus a sensor gap (`grad_norm_*` never populated) and two divergent Karn data models.

---

## Strengths worth protecting

The recurrent-PPO implementation is, in several respects, *better* than typical: BPTT leakage is guarded at construction time; time-limit truncation is handled correctly (rare); the factored per-head trust region is correctly formulated; entropy-collapse defenses use availability masks to prevent sparse-head death-spirals. The GPU-first execution is genuine. The contracts layer is a model of single-source-of-truth design. **Any refactor must preserve these — especially the BPTT construction-time guard and the truncation handling, which should get regression tests so they can't be silently removed.**

---

## Confidence & limitations

- **High confidence:** structure, sizing, coupling (measured via Loomweave + `wc`/grep); the control-flow spine and telemetry path (call sites read directly); GPU-first verdict (sync code read, not just comments); the RL-soundness findings (specialist read the value-head and GAE code).
- **Medium confidence:** completeness of the full ~110-edge simic-internal fan-out (load-bearing edges verified, not every edge); the dead-code negative (static reachability can't prove dynamic reach — per-item grep required before deletion); production-vs-experimental status of the 7 reward modes (not verified against run configs).
- **Not covered:** test-coverage mapping (which god-methods are actually exercised), runtime profiling, and the planned domains (Emrakul/Narset/Esika).
- **Index staleness:** Loomweave index is one commit old; line numbers came from direct reads where it mattered (e.g. `vectorized_trainer.run()` moved to L2910 after a Jun-16 edit).

See `06-architect-handover.md` for a prioritized, effort-sized improvement plan.
