# n=5 J-read — RESULT: (b) r0c0 is an efficiency-enabling stem

**Date:** 2026-07-02 · **Scope:** the banked n=5 leg of the causal-contribution run (PDR-0007). · **Verdict:**
**(b) efficiency-enabling stem** — direction BANKED (5/5 seeds, sign-test p≈0.03); magnitude n=5-coarse.
**Decision record:** PDR-0007 (commission), PDR-0006 (J-reframe), PDR-0004 (total-system estimand).

## Result
`control` (RNG-split ON, no mask) + `suppress_slot_on` (r0c0 un-committable), seeds 41–45, full config, gpu_preload,
committed harness + telemetry crash fix. All 10 runs completed (2400 EPISODE_OUTCOME each). Analyzer:
`scripts/causal_contribution_j_analyze.py`.

**Primary estimand — paired Δ(acc-per-param) = eff_suppress − eff_control, seed-level:**

| seed | control pp/M-param | suppress pp/M-param | Δeff | control totParams | suppress totParams |
|------|--------------------|---------------------|------|-------------------|--------------------|
| 41 | 19.96 | 7.34 | −12.62 | 411k | 1,126k |
| 42 | 11.92 | 7.51 | −4.41 | 560k | 1,079k |
| 43 | 16.34 | 9.30 | −7.05 | 438k | 830k |
| 44 | 14.55 | 9.06 | −5.49 | 504k | 875k |
| 45 | 18.19 | 9.09 | −9.10 | 398k | 912k |
| **median** | | | **−7.05** (mean −7.73, SD 2.91) | | |

Seed-level bootstrap 95% CI **[−12.62, −4.41] — EXCLUDES 0**.

## Gates — ALL PASS (5 seeds)
offset-free (control==on draw-count + state-hash parity); within-run stride constant; `on` germinates ZERO r0c0;
**finiteness trips == 0** (the ep-144 seed-44 transient did not recur; the telemetry crash fix ensures no crash if it
had); decision-step slot-entropy health (min E/LF > 0.1 both arms, all seeds). ⇒ pairing intact, data clean, poolable.

## Verdict (pre-registered decision rule)
**Δeff CI excludes 0 and NEGATIVE → (b) efficiency-enabling stem.** Suppressing r0c0 robustly ~HALVES system
parameter-efficiency (same accuracy, ~2–2.7× the total params) across all 5 independent seeds. r0c0 **buys
efficiency** — the OPPOSITE of a free-droppable freeloader. This confirms the n=3 pilot and resolves the (a)/(b)/(c)
fork at the SYSTEM level (PDR-0004 estimand): **r0c0 is an LOO-undervalued enabling stem; the reward should CREDIT its
enabling contribution, not penalize it.**

## Caveats (held honestly)
- **DIRECTION banked, MAGNITUDE not.** 5/5 same sign is a solid directional result (**sign-test p≈0.03**). The reported
  "95% CI [−12.62, −4.41]" is literally the min/max of the 5 points — treat it as the sign test, **NOT** a confidence
  interval. n=10 (the morphogenesis floor) would give a real magnitude/CI; it is not needed to bank the *direction* (the
  effect is consistent), but it is the only way to bank the *size*. **Owner decision: bank the direction at n=5, or
  extend to n=10 for the magnitude** — that call is yours, not the analyzer's.
- **Healthy EXPLORATION ≠ fully converged.** The read is on a 200-episode policy (decision-step entropy healthy, PDR-0006).
- **System-level, not mechanistic — the gap is ENTIRELY the DENOMINATOR** (advisor). Contribution is ~8pp in BOTH arms;
  only total params move (~400k → ~1M). So this is "the system is less **param-efficient** without r0c0 *available*,"
  NOT "r0c0 **mechanistically enables** the downstream structure." The labels "enabling stem" + a reward named "synergy"
  will drift the reader toward the mechanistic reading — keep the verdict at total-system (PDR-0004).
- **This does NOT settle the REWARD-design question.** The system-level (b) confirmation is necessary but not sufficient
  for the reward-credit term: the terminal-co-dependence proxy the Shapley design uses may diverge from the
  developmental/trajectory effect (the scaffolding blind spot) — that is GATE 1 of the reward design, run next on this
  same data (docs/plans/concepts/2026-07-01-reward-credit-shapley-synergy-design.md).

## Next
1. **Reward-design GATE 1** (proxy-vs-estimand): per seed, correlate control's terminal r0c0-LOO drop vs the suppress
   Δeff — does terminal co-dependence track the developmental effect (build candidate A) or is it the scaffolding
   signature (pivot to candidate B)?
2. Owner: bank (b) at n=5 vs extend to n=10; ratify the reward-credit design path once GATE 1 + GATE 2 (learnability)
   report.
