# Current State — Esper        Checkpoint: 2026-07-12 (~03:00) · checkpoint #40 (PDR-0062/0063; on `feat/ev-stab-stage2-hra`)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet), post-REJECT instrumentation
phase. Two threads live: (1) the **600-round seed-41 OFF diagnostic** whose read is
the last input to the epic-direction DECIDE; (2) the **branch-survivor consolidation**
(PDR-0038 winner-takes-all) now scoped and directionally decided, executing after the
diagnostic. The diagnostic's pre-committed reading (PDR-0055): stationary
EV/Var(returns)/cf-scale over rounds 250–600 ⇒ cf non-stationarity was
schedule-transient; continued drift ⇒ structural morphogenetic churn.

## In flight
- **600-round seed-41 OFF diagnostic** since 2026-07-11 19:29 AEST @ `7ea84dc6`
  (cuda:1, setsid — survives reboot). **~280/600 rounds, healthy, 0 crashes, inside
  the read window (250–600).** Log: `telemetry/stage2_off_longdiag/logs/longdiag-s41.log`
  (done = `exited rc=`); telemetry: `.../seed41/telemetry_2026-07-11_192943/`. QUIET
  BOX while it writes. First run emitting vg telemetry. ETA ~midday AEST 2026-07-12.
- **Branch-survivor consolidation** (esper-lite-1f1e55f58f, in_progress): audit DONE
  (manifest committed `ca4159b6`), direction decided (PDR-0062), EXECUTION gated on the
  diagnostic read + owner sign-off.

## Facts the next session must not relitigate
- **PDR-0062 (branch-survivor):** survivor = `main` as single trunk; mechanism =
  history-preserving MERGE (not cherry-pick), then excise the rejected HRA. Owner-decided
  + advisor-confirmed. KEY: main's HRA `575d2560` is a SQUASH of the branch's Stage-2
  work — SHARED-FOUNDATION fork, not rival lines; same rejected HRA drops on both.
  Manifest: `docs/analysis/2026-07-12-branch-survivor-salvage-audit.md`.
- **PDR-0063 (anti-recurrence, proposed):** single-trunk discipline, product workspace
  owned in one place — needs owner adoption to bind.
- **PDR-0061:** schedule fix + vg Gram telemetry landed `7ea84dc6` (drl-approved).
- **PDR-0059:** Stage-2 REJECT final for this HRA impl; redesign needs new pre-reg.
- Owner grant confirmed 2026-07-10, re-confirmed in-session 2026-07-11. Owner manages
  pushes to origin themselves; agent never pushes.

## Open questions / blocked-on-owner
- **Branch-survivor sign-off (4 items, gated; manifest checklist):** confirm DROP of the
  rejected HRA from trunk; **acceptance-harness keep-or-drop** (keep for a future
  re-pre-registered HRA gate, or drop with the dead design?); approve branch retirement
  (destructive git — escalation); approve the PDR-0063 anti-recurrence practice.
- **Epic direction DECIDE (after the diagnostic read):** TIP / reward-efficiency /
  redesigned HRA A′-or-B / Stage-1 flag. Serial focus ruled by owner.
- North-star target/date, rent ceiling, host-accuracy floor: owner-unset.
- Standing: no push/tag/release/branch-deletion/telemetry-deletion/remote action.

## Last checkpoint did (checkpoint #40)
- Ran the branch-survivor salvage audit (informed jettison, PDR-0038): found a
  shared-foundation fork (main = squash of the branch's HRA), not rival lines. Wrote the
  KEEP/DROP/CONFLICT manifest (`ca4159b6`); direction decided = main-as-trunk via merge
  (PDR-0062), advisor-confirmed.
- Recorded the anti-recurrence single-trunk discipline (PDR-0063, proposed).
- Committed the tooling-drift chore (`7d69dff0`). Two subagents died mid-audit;
  resolved the load-bearing questions directly.

## Next session, start here
Check the diagnostic: `grep "exited rc=" telemetry/stage2_off_longdiag/logs/longdiag-s41.log`.
Still running ⇒ quiet box, light work only (branch-survivor is audit-complete, blocked
on diagnostic + owner sign-off — nothing to execute yet). Completed rc=0 ⇒ score the
PDR-0055 reading (rounds 250–600) + first vg normalizer-lag reads, then bring the owner
BOTH gated decisions together: the epic-direction DECIDE and the branch-survivor sign-off
(4 manifest items). Completed rc≠0 ⇒ RCA first. Do NOT execute the merge before the
diagnostic read and owner sign-off (PDR-0062 gate).
