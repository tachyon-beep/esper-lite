# Current State — Esper        Checkpoint: 2026-07-04 (checkpoint #12 — ON-arm GATE crash adjudicated + fixed; relaunch gates running; PDR-0023)

## The bet right now
**Reward credit-assignment redesign — A/B in controlled recovery.** The 6dd80716 ON
wave crashed 5/5 (designed fail-loud: coalition builder rejected GATE-fossilized
slots; GATED_GATE is an ordinary policy style). Adjudicated per PDR-0023
(owner-ratified): fix (a) SHIPPED (c2e500d3 — admission + invariant preservation,
no estimand change; both adversarial reviews SHIP; crash bug esper-lite-fbeead4efc
CLOSED, GPU-verified on the production crash class). Metric: unchanged — PDR-0019
criteria (i)–(v) primary; episode-level paired Δcorr(reward,J) ≥ +0.10; asymmetric null.

## In flight
- **OFF set:** s41/s42/s44/s45 banked clean at 6dd80716 (2400/2400 each). s43 solo
  rerun at the same frozen commit (worktree ab-frozen-6dd80716) in progress —
  ~880/2400 at checkpoint, ETA this evening. Its OOM predecessor is quarantined
  (off_s43_oom_partial; nothing deleted).
- **Relaunch patch-set BUILT:** `.worktrees/ab-on-relaunch` = 6dd80716 + f98b6b3e
  (episode_idx cherry-pick) + d6514416 (fix+telemetry cherry-pick); 40/40 Shapley
  tests green in-tree.
- **Relaunch gates (PDR-0023 §6):** CPU OFF no-op PASSED (byte-identical digest
  d468d3e7…); GPU OFF replay of s41 at the patch-set RUNNING
  (telemetry/relaunch_gates/off_replay_s41); ON smoke RUNNING and already passing
  (GATE fossils in r0c0 — the production crash class — completing terminal evals,
  finite TOPUP, alpha_algorithms/tau_used present, true episode ids).
- Crashed ON attempts quarantined: telemetry/shapley_ab_n5/gate_crashed_on_wave_6dd80716/.

## Scoring rules (prereg + 2026-07-04 gate-fossil ADDENDUM — do not relitigate)
All PDR-0019 rules stand. NEW (addendum, owner-ratified): the stored×12+env_id
episode_idx decode is RETIRED for relaunched ON TOPUP events (applies ONLY to the
quarantined crashed attempts — applying it to the re-run double-corrupts criterion
(v)); gate/non-gate stratification reads come from TOPUP `alpha_algorithms` +
`tau_used`; tau=0.28 is ADD-placebo provenance — GATE-credit MAGNITUDE claims are
unlicensed without a gated null-player read (direction unaffected); ON-run P99
recalibration is stratified gate vs non-gate; population-validity gates carried
forward (R1's entropy-collapse diagnosis stays retired as telemetry artifact —
PDR-0006). NO outcome-conditioned rerun clause exists: n=5 = direction,
n=10 = magnitude/banking.

## Open questions / blocked-on-owner
- **May the ON wave fire on CPU-byte-identity + passing smoke alone, or must it
  wait for the full GPU off_s41 replay comparison?** Default = wait (stricter
  reading of ratification §6). If the replay shows drift, attribution needs a
  same-commit control before the substitute justification is accepted (PDR-0023
  reversal trigger).
- Standing: north-star/rent TARGET placeholders (metrics.md); enabling scale>0
  BEYOND this experiment stays owner-gated; G3's paired noise floor still unbanked
  (bank it or state absence at scoring).

## Last checkpoint did (checkpoint #12)
- **PDR-0023:** full adjudication recorded — fix (a) mechanism (admission, not
  save/restore), (b)/(c) rejected, minimal-patch-set deviation, decode retirement,
  tau licensing, borderline-rerun clause STRUCK, OOM kept operational (≤2 runs/GPU
  constraint), no-ON-outcome-data attestation.
- Shipped c2e500d3 (fix + gate-identity TOPUP telemetry + 9 test files; 2407 green;
  mypy/lint baselines unchanged) and 9ddf1542 (prereg addendum). Both adversarial
  reviews (drl, pytorch) returned SHIP; all review clauses closed same-session
  (k=2 paying-path test, H1 differential pin, schedule identity pins).
- Tracker: esper-lite-fbeead4efc closed (fix_verification recorded); enablement
  gate esper-lite-f22a1d48a7 updated (comment 113); H3 latent-gate-reset filed as
  observation esper-lite-obs-84562d1a16.
- OFF no-op CPU proof run and banked (control determinism + cross-tree byte-identity).

## Next session, start here
**Check the relaunch gates.** (1) ON smoke exit status + final TOPUP assertions;
(2) GPU off_s41 replay vs banked off_s41 — compare decision/reward/lifecycle
streams + final acc (strict identity may break on CUDA/co-tenancy nondeterminism;
if drift, run the same-commit control before attributing). Gates green + owner's
answer on the wait-question → **launch the ON wave: 5 seeds at the patch-set
worktree, ≤2 runs per GPU**, telemetry to telemetry/shapley_ab_n5/on_s4*, monitor
armed. Then score STRICTLY per prereg + addendum. Queued behind the A/B decision:
advantage-pathology / EV track (owner sequence).
