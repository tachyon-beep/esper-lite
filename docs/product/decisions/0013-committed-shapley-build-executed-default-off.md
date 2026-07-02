# PDR-0013 — Committed-Shapley top-up BUILT (default-OFF) and ACCEPTED; enablement handed to its own gate

Date: 2026-07-02   Status: accepted (owner sign-off for the build received in-session — "kick off the
committed-shapley top up"; the build execution and its within-scope calls were within grant)   Author: Claude (agent)
Supersedes: — (completes PDR-0012's proposal)   Related: PDR-0011, PDR-0012,
docs/plans/completed/2026-07-02-committed-shapley-topup-build.md, esper-lite-254175df90 (closed),
esper-lite-f22a1d48a7 (enablement gate)

## Context
PDR-0012 proposed the default-OFF build pending owner sign-off. The owner granted it in-session
(2026-07-02, recorded as comment 93 on esper-lite-254175df90) with "use ultracode at your discretion,
as well as all useful skills and subagents to plan, review, implement and verify."

## The call (what was executed and the within-build decisions made)
**Built, same day, across commits 7ec67de7 (plan) + a479d776..2d7e0db5 (implementation):** exact 2^k
committed-coalition Shapley over FOSSILIZED slots in the terminal fused val pass; retro-write delivery
`buffer.rewards[env, t_f] += min(normalized_cap, top_up / max(std, std_floor))` at the pre-GAE seam;
five flags default 0.0; ~100 new tests including the mandated hand-computed GAE retro-write test,
bit-exact kernel masking, no-op-at-0, and live end-to-end (zero AND forced-nonzero payout).

Within-build decisions (specialist-reviewed, documented in the plan's §6):
1. **Sibling mutual exclusion** — scale>0 structurally disables the interaction bonus and hindsight
   credit (rejected: net-into-c_paid, assert-dormant). A/B coupling confound accepted (dormancy-bounded).
2. **F2 as two bounds** — normalized-space cap PRIMARY + std-floor secondary; `__post_init__` refuses
   scale>0 without cap>0 AND normalized_cap>0 (both plan reviewers independently found the original
   delivery formula inexpressible through divide_by_std — fixed via `current_std()`).
3. **HRA hard exclusion** — ValueError at scale>0 with hra_value_decomposition (no CF-stream routing).
4. **c_paid = v({s})−v(∅) terminal** (reviewer F4/F6) — design doc amended with the semantic shift.
5. **Naming (F8)** — legacy `synergy_bonus` → `interaction_bonus` repo-wide; old runs' telemetry keys
   not rewritten (documented cost).

## Acceptance evidence
drl-expert + pytorch-expert plan reviews (approved-with-changes, folded in); reward-function-reviewer
re-pass on the BUILT code: APPROVE_WITH_CHANGES — "merge/ship the default-OFF build," its one MEDIUM
(no live nonzero payout) closed in-build (2d7e0db5); adversarial code review: CLEAN (8/8 hazard areas,
scale=0 "airtight"). Suites: tests/simic+leyline 1718 / kasmina 641 / karn 763 passed; sole failures =
pre-existing Phase 0 seed_residency cluster (esper-lite-obs-7240279be3, not this build's).

## Rationale
Every reviewer condition (F1–F8, F5, GAE test) verified as built; the shipped default (scale=0.0) is
byte-identical to status quo; enablement stays a separate, further-gated owner decision
(esper-lite-f22a1d48a7 carries the banked criteria + build-time residuals).

## Reversal trigger
Remove the term (delete, per No-Legacy — not flag-rot) if the enablement A/B falsifies design criteria
(i)–(v) — in particular (v) failing reopens GATE 2 per PDR-0011's trigger — or if the enablement gate's
hard off-switch-J efficiency + fossilize-count guard trips on the first ON calibration runs.
