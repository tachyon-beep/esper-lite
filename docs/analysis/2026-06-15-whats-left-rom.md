# What's Left — ROM Breakdown (shadow-review vantage)

**Written:** 2026-06-15, from 9 passes of read-only shadow review of the in-flight
work on branch `0.1.1` (HEAD `82155c14`, all uncommitted). Companion to
[shadow-review-followups.md](./2026-06-15-shadow-review-followups.md).

**Bottom line:** The *code* is ~done. What remains splits into three very
different buckets, and the honest "weeks" estimate is almost entirely **Bucket C
(experiments, compute-bound)** — plus possibly **Bucket B (self-assigned
prerequisite code)** if those prerequisites are kept. Bucket A (cleanup) is hours.

The estimate drifted days→weeks because the agent ran out of tickets and
self-expanded scope into a full reward-efficiency *proof regime*. That work is
high quality, but it's the agent setting its own homework — so the lever you
have over "weeks" is **deciding how much of Bucket B is actually required**.

---

## Bucket A — Code cleanup (hours, not weeks). None are blockers.

All found during shadow review; none affect the live training path.

| Item | What | Est. |
|------|------|------|
| F10 | Pin `FIXED_SCHEDULE_GERMINATE_R0C0_HASH` to a literal + assertion (tamper-evidence) | ~15 min |
| F12(a) | Delete dead `state_dict_exact` replay-policy constant; tighten validation to `topology_only` | ~15 min |
| F13 | Fix `source_group_id IS NULL` false-close in proof_packet static-final query + test | ~20 min |
| F14 | Delete dead `VectorizedPPOTrainer.proof_baseline_mode` field | ~5 min |
| F15 | Flag source-cohort forced steps as `proof_controlled_step` (or comment) | ~10 min |
| F16 | Delete unreachable `unsupported_policies` empty-tuple guard | ~5 min |
| F8/F9 | proof_packet API/CLI default-profile docstring + happy-path test | ~30 min |
| F2/F3 | bf16 masked-position symmetry (optional) + stale comment | ~20 min |
| F17 | Decide if 10s telemetry flush-timeout is right / make configurable | ~20 min |

**Bucket A total: ~2–3 hours of focused work.** Optional, can ship the verdict without most of them.

---

## Bucket B — Code that may still be unbuilt (DAYS — and the real scope question)

Per the agent's own `PLAN_TRACKER.md`, the critical path it has drawn is:

```
correctness-proof-strategy → morphogenesis-governor-integrity →
ppo-stability-oracle-sandbox → reward-efficiency verdict
```

- **proof-baseline-controls / proof regime** — ✅ essentially built this session
  (4 control cohorts, topology replay, lockstep A/B, fail-closed proof_packet).
- **morphogenesis-governor-integrity** — drafted, P1, *not built*. Rollback
  ordering, observation truthfulness, blueprint contracts. **Scope unclear from
  shadow vantage — could be days.**
- **ppo-stability-oracle-sandbox** — planning artifact only. Isolates
  value-collapse / gradient-anomaly proof blockers. **Days if pursued.**

**The decision that controls "weeks":** Are governor-integrity and the
oracle-sandbox *strictly required* to get a defensible reward-efficiency verdict,
or are they gold-plating the agent added because they're rigorous-and-adjacent?
If required → Bucket B is real days of coding. If not → cut them and Bucket B
collapses to ~0. **This is the single biggest lever on the timeline and it's a
judgment call only you can make.** (Recommend: ask the agent to justify each as a
hard prerequisite for the verdict vs. a nice-to-have.)

---

## Bucket C — The experiment / validation phase (the actual WEEKS — compute-bound)

This is wall-clock, not typing. Cannot be shortened by the agent coding faster.

- **F1 — run the validation.** The two deferred P2 fixes (rollback
  tail-dominance, op-head FP32+floor) are correct *code* but *unvalidated training
  dynamics*; they need an ablation/retraining run. Compute-bound.
- **Reward-efficiency exam (Phase 2.5 gate).** Infra is 100% done. Running it =
  ~100 episodes × 8–12 envs across the cohorts on `cifar_impaired`, then analysis.
  Per ROADMAP. Days of GPU + analysis + likely iteration.
- **Iteration.** RL exams rarely pass first try; budget for a couple of cycles.

**Bucket C: days→weeks of compute + analysis, irreducible.** This is the honest
core of the agent's "weeks" — and it's legitimate.

---

## Also (not in any bucket): commit hygiene — ~1–2 hours, do soon

HEAD has been `82155c14` for 9 review passes with **~5,500 uncommitted insertions
across 58 files**. Independent of the timeline question, this should be
checkpointed into a few logical commits soon: a stray `git reset`/crash loses it,
and it's getting hard to review as one monolith. Suggest splitting roughly:
(1) DRL-audit bugfixes, (2) telemetry/Karn surface, (3) proof regime +
proof_packet, (4) docs/plans. Run the full `uv run pytest` before/after.

---

## TL;DR for the timeline

- **"Weeks of me coding"** → push back; the code is near done (Bucket A = hours,
  Bucket B = days *only if* the self-assigned prerequisites are kept).
- **"Weeks until a defensible reward-efficiency verdict"** → plausibly honest,
  because that's Bucket C (an *experiment*), not a patch.
- **Biggest lever:** decide whether governor-integrity + oracle-sandbox are
  required prerequisites or gold-plating. That alone can swing the estimate from
  "days + compute" to "weeks."
