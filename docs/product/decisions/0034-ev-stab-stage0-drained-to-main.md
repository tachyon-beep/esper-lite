# PDR-0034 — EV-stab Stage-0 telemetry drained to main; branch unification narrowed to "retire ev-stab"

Date: 2026-07-05   Status: accepted (owner-authorized in-session: "go ahead and do it now",
in response to the unification recommendation). Executes the deferred disposition of
PDR-0031 (ev-stab → main = MERGE, deferred) now that its Stage-0 gate has been met (PDR-0032/0033).
Author: Claude (agent)   Related: PDR-0031 (branch consolidation; this drain is the reserved
core-training-path merge), PDR-0028 (value-free gate is THE gate). Tracker: esper-lite-f25b71c165.

## Context

PDR-0031 consolidated 4 live branches → 2 and recorded the ev-stab → main merge as MERGE,
DEFERRED, gated on Stage-0 completion + OFF-leg byte-identity + drl/pytorch review. Stage-0
completed and its gate was accepted (PDR-0033). Investigation for the merge revealed the true
shape of "unification": **main already contained the squashed Stage-2 HRA work** (via the
Shapley-line merge `f3758236`), so ev-stab was NOT the sole home of Stage-2. The only
main-missing, validated content on the branch was the **5 value-free Stage-0 telemetry commits**;
the granular Stage-2 is behaviourally ≡ main's squash, and 6 oracle-sandbox commits are WIP.

So "merge the branch" was the wrong frame. The correct unit is: **drain the 5 Stage-0 commits
to main, then retire ev-stab.**

## Options

- **(a) Drain the 5 Stage-0 commits to main now; leave the granular Stage-2 (≡ main squash) and
  the oracle WIP on the branch; verify with the byte-identity golden + specialist review.**
- **(b) Defer the whole merge until Stage-2 acceptance.** Rejected: Stage-0 is value-free
  (PDR-0028), decoupled from whatever Stage-2 acceptance decides, and it is the deliverable of
  a just-closed task — stranding validated telemetry on an orphan-prone branch is the org's
  chronic anti-pattern (CLAUDE.md).
- **(c) Whole-branch merge.** Rejected: would drag in oracle WIP and move `feat/sanctum-layout`'s
  base (it is stacked on ev-stab@70753923); a targeted cherry-pick leaves sanctum undisturbed.

## Call

**(a), executed and landed on LOCAL main (not pushed).** The 5 commits (`f721a5b3`, `562cb5bf`,
`ac2b27c8`, `82977370`, `70753923`) were cherry-picked onto a scratch branch off main, the
reconciliation conflicts hand-resolved, verified, and fast-forwarded onto `main`
(`e5bab048` → `17b5f634`, 19 files, +965/−16).

**Reconciliation (main had advanced 90 commits past the branch base):**
- `reward_variance.py`: UNION of main's per-step `compute_variance_shares` + the drained
  per-return family — disjoint, pure append.
- `partition.py`: main already had `decompose_additends`/`ADDITEND_SIGN_MAP`/`split_reward_streams`;
  drain added ONLY `COMPONENT_TERMS`. The `synergy_bonus` (ev-stab) → `interaction_bonus` (main
  canonical, matching the `RewardComponentsTelemetry` field) rename was applied throughout —
  zero `synergy_bonus` remain except one negative test assertion.
- `action_execution.py` / `vectorized_trainer.py`: adjacent-addition UNIONs (kept both
  `extra_blueprints` and `return_variance_telemetry`).

**Verification (all green):**
- **Byte-identity golden `test_ppo_update_golden.py` (2/2)** — PDR-0031's named OFF-leg detector:
  main's squashed Stage-2 and the drained Stage-0, flag off, are bit-identical.
- Full Stage-0 surface (120), changed-domain regression (`tests/simic/{rewards,telemetry,agent}`
  + training 290), slow full-loop reducer smoke — all pass. (The pre-existing dual-GPU
  `test_dual_ab.py` hang is unrelated to this additive, default-off change.)
- **pytorch-expert review: GO** — empirically verified all reconciliation hazards, the reducer
  whitelist (`return_var_*` keys) via a DuckDB round-trip through the Karn `ppo_updates` view,
  flag composition, and no dropped hunks / defensive-programming smells.

**Consequence:** the value-free Stage-0 gate telemetry now lives on canonical `main`. Branch
unification narrows to: **retire `feat/ev-stab-stage2-hra`** once Stage-2 acceptance resolves
(the granular Stage-2 ≡ main's squash, confirmed by the golden; the oracle WIP finishes or is
abandoned separately). `feat/phase-minus1-scale-falsifier` (0 ahead of main) deleted as redundant.

## Rationale

Draining only the validated, value-free, default-off telemetry — rather than the whole branch —
is the smallest correct unit: it lands the deliverable of a closed task, is decoupled from the
still-pending Stage-2 acceptance, leaves `sanctum-layout`'s stacked base untouched, and quarantines
the oracle WIP. The byte-identity golden is what certifies the merge is faithful, not the absence
of git conflicts (the auto-merge silently produced a duplicate `ADDITEND_SIGN_MAP`, caught and
fixed before landing).

## Reversal triggers

- Push is OWNER-GATED (never pushed from an agent without explicit ask). If a pre-push full-suite
  run (`uv run pytest`, longer timeout) surfaces a regression the targeted sweep missed → fix on
  main before push; the drain does not reach `origin` until then.
- If Stage-2 acceptance (esper-lite-2a4b56e719) bounces the epic to a DPBA refold, the granular
  Stage-2 on ev-stab changes — but Stage-0 (value-free) is unaffected and stays on main.
- If retiring ev-stab would lose the still-unmerged oracle-sandbox WIP → finish-or-abandon those
  6 commits explicitly first (PDR-0030 disposition discipline); do not delete the branch blind.
