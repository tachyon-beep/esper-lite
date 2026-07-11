# Current State — Esper        Checkpoint: 2026-07-11 (~19:45) · checkpoint #39 (PDR-0061; on `feat/ev-stab-stage2-hra`)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet), post-REJECT instrumentation
phase: the **PDR-0055 penalty-schedule fix + PDR-0060 vg Gram telemetry LANDED at
`7ea84dc6`** (drl-expert APPROVE-WITH-NITS, both nits closed, 5341 tests green;
PDR-0061), and the **clean 600-round seed-41 OFF diagnostic is IN FLIGHT** on that
code. The diagnostic's pre-committed reading (PDR-0055, carried from 0053/0054) is
the last input to the epic-direction DECIDE: stationary EV/Var(returns)/cf-scale
over rounds 250–600 ⇒ the cf non-stationarity was schedule-transient; continued
drift ⇒ structural morphogenetic churn.

## In flight
- **600-round seed-41 OFF diagnostic** since 2026-07-11 19:29 AEST @ `7ea84dc6`
  (cuda:1, session-proof setsid — survives session reboot). Log:
  `telemetry/stage2_off_longdiag/logs/longdiag-s41.log` (completion line =
  `exited rc=`); telemetry: `telemetry/stage2_off_longdiag/seed41/telemetry_2026-07-11_192943/`.
  Verified healthy at launch (90% GPU util, epoch 41 by 19:33). **QUIET BOX while it
  writes** (PDR-0056). First run emitting vg telemetry ((v, g) family).
- Nothing else running; cuda:0 idle.

## Facts the next session must not relitigate
- **PDR-0061:** the landing is accepted and closed — schedule breakpoints absolute
  (leyline `ENTROPY_PENALTY_*`), bit-identity scoped to EXACTLY 200-round runs;
  `total_train_steps` retired (pre-fix checkpoints fail loud); vg telemetry
  end-to-end (20 columns in karn `ppo_updates`). Offline caveats: vg moments
  correction=0 vs EV correction=1; mean-of-Grams pooling needs equal counts.
- **PDR-0059 (verdict):** Stage-2 REJECT final for this implementation; redesigned
  HRA needs a NEW pre-registration. **PDR-0060:** banked vs provisional register for
  the pre-DECIDE evidence; path lean slightly B, A′ credible — ADVISORY ONLY.
- **PDR-0058 (§11.3 Option B)** validity envelope; **PDR-0057** entropy-floor
  population audit front-of-TIP.
- Owner grant confirmed 2026-07-10 and re-confirmed in-session 2026-07-11.
- Owner manages pushes to origin themselves (observed twice); the agent never
  pushes. Branch state at checkpoint: ahead of origin by this checkpoint commit
  (+ possibly `7ea84dc6` depending on owner's last push).

## Open questions / blocked-on-owner
- **Epic direction DECIDE (after the diagnostic read):** TIP (esper-lite-c62d4891b0)
  / reward-efficiency (esper-lite-a2abff5ec5) / redesigned HRA (new pre-reg, A′-or-B
  per the PDR-0060 discriminators — the vg telemetry now emits what the held-out
  calibration-rescue test needs on future ON runs) / Stage-1 per-head-norm flag
  (esper-lite-89983714fb). Serial focus ruled by owner.
- **Branch-survivor disposition (esper-lite-1f1e55f58f) UNBLOCKED** — informed
  jettison (salvage audit + owner sign-off) before any discard (PDR-0038).
- North-star target/date, rent ceiling, host-accuracy floor: owner-unset.
- Standing: no push/tag/release/branch-deletion/telemetry-deletion/remote action.

## Last checkpoint did (checkpoint #39)
- Executed the PDR-0055 licensed work under TDD: schedule fix + vg telemetry, both
  drl-review-gated in one landing (`7ea84dc6`); review nits closed pre-landing.
- Launched the 600-round diagnostic on the fixed code and verified it live.
- Tracker: created + closed esper-lite-00ff305247 (schedule fix) and
  esper-lite-657fec0019 (vg telemetry) with full review trail; PDR-0061 appended.

## Next session, start here
Check the diagnostic first: `grep "exited rc=" telemetry/stage2_off_longdiag/logs/longdiag-s41.log`.
Still running ⇒ quiet box, light work only (e.g. branch-survivor salvage audit prep,
entropy-floor audit concept — read-only). Completed rc=0 ⇒ score the PDR-0055
pre-committed reading over rounds 250–600 (plus the rounds-0–200 continuation
corroboration and the first vg normalizer-lag/scale-slope reads), then bring the
owner the epic-direction DECIDE with the full evidence base. Completed rc≠0 ⇒ RCA
before anything else; the run is the epic's critical path.
