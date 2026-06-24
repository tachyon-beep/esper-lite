# Plan Metadata
id: ev-stabilization-stage1-marginal-v-verification
title: "Stage 1 — Marginal V(s) baseline + expected bootstrap (P0-1) — VERIFICATION/CLOSURE"
type: ready
created: 2026-06-23
updated: 2026-06-23
owner: Claude (ralph-loop)

urgency: high
value: Confirm the reports' #1 recommendation is already satisfied so the epic does not re-implement a landed fix; capture the acceptance evidence and close the Filigree task.

complexity: S
risk: low
risk_notes: Verification only. The only risk is a false-positive "done" — mitigated by reading the actual code + git, done below.

depends_on: []
soft_depends: []
blocks: []

status_notes: Implementation DONE (commit 6a27b8e3). This artifact records the verification; remaining work is to confirm acceptance evidence and close Filigree esper-lite-e6382020d2.
percent_complete: 80

reviewed_by:
  - reviewer: reality-check (self, code + git)
    date: 2026-06-23
    verdict: approved
    notes: P0-1 confirmed landed; see evidence below.

---

# Stage 1 — Marginal V(s) baseline + expected bootstrap (P0-1) — VERIFICATION

## Verdict: ALREADY IMPLEMENTED — do not write an implementation plan

The four research reports unanimously ranked "replace the action-conditioned
`Q(s,op)` baseline + single-sample `Q(s',op')` bootstrap with an op-independent
`V(s)` / marginal `V(s)=Σπ(op|s)Q(s,op)`" as the #1, cheapest, highest-leverage
move. **It has already landed**, as the *directly-learned op-independent V(s)*
variant the reports themselves called the cleaner long-term design.

## Evidence (verified 2026-06-23 against current branch)

- **Git:** `6a27b8e3 feat(tamiyo,simic): op-independent V(s) baseline replacing Q(s,op) (P0-1)` is in this branch's history.
- **Critic head:** `factored_lstm.py:451-461` defines `state_value_head`, an op-INDEPENDENT V(s); the comment at `:438-446` states it is "the scalar used EVERYWHERE the PPO baseline matters: the rollout-stored value, the GAE bootstrap (V(s') estimates E_a[Q(s',a)] directly instead of one sampled op), the value-loss target, and explained_variance."
- **Q(s,op) demoted:** `factored_lstm.py:463-494` — `q_head` "RETAINED for telemetry only… NO LONGER the PPO baseline," trained as a small detached aux regression.
- **Bootstrap is V(s'):** `rollout_buffer.py:557-578` — `delta = rewards[t] + gamma_t * next_value * next_non_terminal - values[t]`, where `next_value` is the stored state-value (`values[t+1]` or `bootstrap_values[t]`), i.e. `V(s')`, NOT a sampled `Q(s',op')`.
- **Both heads optimized, V(s) is the baseline:** `ppo_agent.py:414-420` (params), `:898-899` ("value" = op-independent V(s); "q" = telemetry aux), EV computed on V(s) (`:671`).

The "expected/all-action bootstrap E_op[Q]" the reports recommend is satisfied:
a directly-learned `V(s')` IS the learned expectation, with no per-op enumeration
needed.

## Remaining work (to close Filigree esper-lite-e6382020d2)

1. **Acceptance evidence.** P0-1 landed alongside the `gain=0.01→0.1` value-head
   init (`b1935d39`). Confirm the EV-liftoff acceptance result attributable to
   the op-independent V(s) (median EV ~0.31, positive post-anneal slope) is
   recorded — see memory `recurrent-ppo-next-steps` (the 2×2 substrate×K run) and
   the handover `docs/handovers/2026-06-18-recurrent-ppo-and-p0-followup.md`.
   If a clean P0-1-isolated A/B (op-independent V(s) vs the old Q(s,op) baseline)
   was never run, note it as a "nice-to-have, not blocking" — the architectural
   correctness is unambiguous and the live EV is healthy.
2. **Golden re-baseline check.** Confirm `tests/simic/test_ppo_update_golden.py`
   was re-baselined for the V(s) change (memory notes a prior gap after
   `b1935d39`; verify it is green now).
3. **Close** `esper-lite-e6382020d2` with reason pointing at `6a27b8e3` and this
   doc; redirect its downstream (Stage 2 / 1b / COMA) to depend on Stage 0 +
   review instead.

## Reviewers
drl-expert sign-off that the op-independent V(s) + V(s') bootstrap fully
discharges the reports' "Change B"; no further critic-baseline work needed.
