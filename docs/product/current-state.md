# Current State — Esper        Checkpoint: 2026-07-11 (~01:00) · checkpoint #37 (PDR-0057..0059; on `feat/ev-stab-stage2-hra`)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet): the **Stage-2 MAJOR-1 n=5
screen returned REJECT (PDR-0059)** — LEG-A FAIL (median Δ_A ≈ −0.093, 4/5 seeds
regress) ∧ MECH fail ∧ G4 fail; G1/G2/G3 pass. No n=10 for HRA objective-A as
implemented. The Stage-0 de-shaping DIAGNOSIS is untouched — the cf stream still
dominates return variance; what failed is this treatment (decomposition without
stabilizing the non-stationary cf target: scale drifts ~2–4 → ~24–28 within every
run). Packet of record: `docs/analysis/2026-07-11-stage2-major1-packet.txt`
(`bac4cc7f`). **The freeze is LIFTED** (wave complete, read banked).

## In flight
- Nothing running. GPUs idle. Quiet-box rule applies only while evidence runs write.
- **Next licensed work (PDR-0055, unchanged):** penalty-schedule fix — absolute
  update-round breakpoints pinned to the 200-round shape, leyline constants, retire
  `total_train_steps` from the schedule; **drl-expert review REQUIRED before
  landing**; test plan now includes the 0–199 coefficient-identity regression +
  off-by-one cases (PDR-0057 arc). Then relaunch the clean 600-round seed-41 OFF
  diagnostic (pre-committed readings in PDR-0055; fallback = PDR-0054 window).
- **Then (post-diagnostic):** the owner DECIDE on epic direction — see open questions.

## Facts the next session must not relitigate
- **§11.3 (PDR-0058, owner-ruled Option B):** §1 EV-finiteness is evaluated on the
  scored population; explained_variance NaN is a valid marker ONLY on §8B-floored
  updates; ±Inf and NaN-unfloored and non-finite ev_sum/ev_main/ev_cf are hard
  INVALID. One shared `decompose_population`. Envelope violations re-open the design,
  never widen it. Scorer fix + 10 contract tests: `d187402a` (332 tests green).
- **PDR-0059 (verdict):** REJECT is final for THIS implementation per §7. A
  redesigned HRA (cf-target stabilization first) needs a NEW pre-registration.
  Mediator caveat travels with the verdict (HRA estimated under current
  entropy-floor objective + 200-round schedule).
- **PDR-0057 (audit):** entropy-floor population audit is front-of-TIP; feasibility
  margin `d − floor` first; three-track separation (schedule fix / observability /
  objective change — objective change needs own PDR + experiment). Concept:
  `docs/plans/concepts/2026-07-10-entropy-floor-population-audit.md`.
- **PDR-0056 trigger resolved:** rerun 5/5 clean, zero drops — co-tenancy
  attribution CONFIRMED; quiet-box rule stays standing for future evidence runs.
- **Owner grant confirmed 2026-07-10 in-session** ("Yes") — wording stands as
  recorded in vision.md.

## Open questions / blocked-on-owner
- **Epic direction DECIDE (after the 600-round diagnostic):** TIP
  (esper-lite-c62d4891b0) / reward-efficiency (esper-lite-a2abff5ec5) / redesigned
  HRA (new pre-registration) / Stage-1 per-head-norm flag (esper-lite-89983714fb).
  Owner ruled serial focus; the read now exists, the diagnostic completes the arc.
- **Branch-survivor disposition (esper-lite-1f1e55f58f) is now UNBLOCKED** — it was
  deferred to "Stage-2 completion" (PDR-0038); the screen verdict is in. Owner call,
  informed-jettison discipline applies (salvage audit before any discard).
- **~16 commits unpushed** on `feat/ev-stab-stage2-hra` — no push without explicit
  ask (standing).
- North-star target/date, rent ceiling, host-accuracy floor: owner-unset.
- Standing: no push/tag/release/branch-deletion/telemetry-deletion/remote action.

## Last checkpoint did (checkpoint #37)
- ON rerun completed 5/5 clean; wave stewarded under quiet-mode (log-tails only).
- First scoring INVALID → diagnosed as checker-population artifact (2/2000 emitter
  NaN-markers) → owner ruled Option B → §11.3 envelope implemented, tested,
  committed pre-rescore with hashed inputs and no-peek held (PDR-0058).
- Rescore: §1 VALID → **REJECT** read banked + delivered (PDR-0059); packet,
  metrics, roadmap, tracker all updated.
- Entropy-floor population audit adopted from 3-round external review, all claims
  code-verified; concept doc + tracker row landed (PDR-0057).

## Next session, start here
Run the penalty-schedule fix under PDR-0055 (drl-expert review gate) and relaunch
the 600-round diagnostic — session-proof launcher, quiet box. If the owner is
present, surface the two unblocked DECIDEs (epic direction; branch survivor)
before starting long work.
