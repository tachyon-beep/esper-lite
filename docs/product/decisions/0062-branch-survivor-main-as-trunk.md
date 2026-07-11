# PDR-0062 — Branch-survivor call: `main` as single trunk, merge-based consolidation

Date: 2026-07-12
Status: accepted (owner-decided direction; EXECUTION gated on the diagnostic read +
owner sign-off on the retirement/excision — see reversal trigger and escalations)
Resolves: the B-deferred winner-takes-all disposition (PDR-0038), whose trigger was
"Stage-2 completion" — reached at the REJECT verdict (PDR-0059).

## Context

The Stage-2 REJECT unblocked the ev-stab-stage2-hra-vs-main survivor call. PDR-0038's
informed-jettison discipline forbids discarding either line blind. The salvage audit
(`docs/analysis/2026-07-12-branch-survivor-salvage-audit.md`, commit `ca4159b6`) found
the framing was wrong: this is a SHARED-FOUNDATION fork, not two rival lines. Main's
HRA (`575d2560`) is its own commit message's admission a *squash* of the branch's
Stage-2 work onto 0.3.0 ("granular history preserved on branch feat/ev-stab-stage2-hra").
`git cherry`'s "1/125 patch-equivalent" was a squash artifact; `advantages.py` is
byte-identical across lines and `partition.py` differs by only ±60 — same lineage.

## Options considered

- **A — Branch survives (feature branch becomes trunk).** Rejected: making the feature
  branch the default ref needs an escalation-gated default-branch swap / history
  rewrite; more git friction, not less.
- **B — Main survives (single trunk = main).** CHOSEN by owner.
- **C — Reconcile, discard neither.** Subsumed by B's mechanism (a merge preserves
  both sides' unique work; nothing is discarded except the rejected HRA).
- **D — Defer entirely.** Adopted only for EXECUTION timing, not for the direction.

## The call (owner + advisor-confirmed)

1. **Survivor = `main` as the single trunk.** It is the default ref PRs target and the
   remote tracks; branch-as-trunk would need an escalation-gated swap. No effort
   asymmetry exists — a consolidation discards nothing either way.
2. **Mechanism = history-preserving MERGE, not cherry-pick.** Merge
   `feat/ev-stab-stage2-hra` into `main`, then excise the rejected HRA as one
   deliberate change; verify main's reward-redesign/Shapley/PIN-E track survives. The
   product-workspace superset (PDRs 0038–0061 additive, newer vision.md) merges with
   zero conflict.
3. **DROP the rejected HRA from the trunk** (cf_value_head, split_reward_streams V_cf
   routing, per-stream GAE, V_cf normalizer, VALUE_HEAD_SCHEMA v3) — it is the same
   rejected design on both lines. The value-free Stage-0 *diagnosis* is KEPT.
4. **KEEP-and-salvage the branch-unique work** onto the trunk via the merge: Sanctum v2,
   schedule-fix + vg telemetry (`7ea84dc6`), the entropy fix (superset of main's —
   main has 0 references to `choice_conditional_head_entropies`), escrow-Markovian fix,
   oracle sandbox, per-head adv-norm (already identical), acceptance harness (pending
   owner keep/drop), product PDRs 0038–0061.
5. **EXECUTION is gated** on (a) the 600-round diagnostic read and (b) owner sign-off on
   the four manifest checklist items (DROP confirm, acceptance-harness keep/drop, branch
   retirement, anti-recurrence PDR). This PDR records the DIRECTION, not the merge.

## Escalations flagged (owner sign-off required before execution)

- Branch retirement / deletion (destructive git + subsystem deprecation) — gated.
- Excising the rejected HRA from the trunk (feature removal) — confirm.
- Acceptance-harness disposition: keep for a future re-pre-registered HRA gate, or drop
  with the dead design?

## Reversal trigger

- If the 600-round diagnostic read materially changes the epic direction toward a
  redesigned HRA (A′/B) that would REUSE the branch's HRA scaffolding or acceptance
  harness, revisit the DROP scope before executing (the harness/scaffold may become
  KEEP). The trunk choice (main) does not reverse; only the DROP set is contingent.
- If a merge dry-run reveals the reward-track (contribution.py/reward_variance.py) is
  NOT cleanly preserved, escalate before proceeding — do not force-resolve main's
  research track into a loss.
