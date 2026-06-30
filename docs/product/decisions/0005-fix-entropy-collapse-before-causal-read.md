# PDR-0005 — Fix the pervasive entropy collapse before banking any causal read

Date: 2026-06-30   Status: **superseded (by PDR-0006 — the "collapse" was a measurement artifact; no training collapse to fix)**   Author: Claude (agent)   Owner sign-off: yes (grant — reprioritize the active bet); owner chose "investigate first" via AskUserQuestion
Supersedes: —   Related: PDR-0003, metrics.md, docs/analysis/2026-06-30-r1-pilot-result.md

## Context
The R1 pilot (n=3) **validated the harness** (offset-free, zero-r0c0, no contamination, full
scale) but delivered **no causal read on (a)/(b)**: both arms' policies suffered slot-head
**entropy collapse from ~update 3**, so all structure commits under a degenerate controller —
the wrong population for a question about a *healthy* controller's structural choices. An
earlier "r0c0 is replaceable / relocation" read was caught and pulled (population mismatch).
The owner chose to **investigate the collapse before spending more GPU**.

## Options considered
1. **Fix the collapse first, then resume the causal track** — pro: a healthy policy is the
   correct population and the prerequisite for any trustworthy (a)/(b) read; con: a training-
   stability sub-project before the causal answer.
2. Push ahead to n=5/n=10 on the current config — REJECTED by the investigation: more seeds
   on a collapsing controller just buy more uninterpretable runs.
3. Re-prioritize / abandon the reward-redesign bet (binding constraint is elsewhere) —
   REJECTED by the investigation evidence (below).

## The call
Option 1. The investigation (cheap, from existing telemetry) was decisive: **(i)** the
morphogenesis is REAL — committed structure adds **+7.69pp mean across all 6 runs**
(host-alone ~38% → with-structure ~46%; confirms+exceeds the n=1 +6.1pp guardrail), so the
"hollow / accuracy-decoupled" story is refuted; **(ii)** the entropy collapse is **general and
pre-existing** — GATE-1's runs collapsed too (411 anomalies ~ R1's ~450), so it is NOT caused
by the causal harness and has been present throughout the line of work. ⇒ the collapse is a
**fixable training-stability issue, not fundamental brokenness** → fix it first; do not
re-prioritize. The fix is now FOUNDATIONAL (it gates the causal read, may raise the
contribution, and affects every prior result measured on a collapsed policy).

## Rationale
Banking a causal direction off a degenerate policy is the exact population-mismatch error
the discipline exists to prevent. The +7.69pp (n=6) evidence makes "abandon the bet"
unsupported; the GATE-1-collapsed evidence makes "the harness broke it" false. The only
coherent move is to restore policy health, then re-run.

## Reversal trigger
Re-prioritize to "binding constraint is elsewhere" (PDR-0003 Option-3 route / roadmap
re-shape) **only if** a bounded collapse-fix effort (scoped levers: anneal-window,
std-floor / entropy-floor penalty on the raw distribution, fp32 policy logits) **fails to
restore healthy slot-head entropy** (metrics.md policy-entropy guardrail stays collapsed)
across a small number of iterations, OR a healthy policy is reached but the host-accuracy
contribution does NOT change and the structural decisions stay degenerate — i.e. policy
health turns out not to be the lever. Until then, the collapse fix is the active Now work.
