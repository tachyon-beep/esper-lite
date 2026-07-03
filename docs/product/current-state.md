# Current State — Esper        Checkpoint: 2026-07-04 (checkpoint #11 — external-review batch cleared; A/B still RUNNING; PDR-0022)

## The bet right now
**Reward credit-assignment redesign — the enablement A/B is RUNNING.** Committed-Shapley
top-up: F2 knobs FROZEN (PDR-0020), build accepted (PDR-0021), both arms at launch
commit **6dd80716**. Metric: PDR-0019 — design criteria (i)–(v) primary; episode-level
paired Δcorr(reward,J) ≥ +0.10 floor; asymmetric null.

## In flight
- **Paired n=5 OFF/ON A/B RUNNING on both GPUs** (owner GPU-go 2026-07-03): OFF wave
  deep in training (episodes ~770–1090 across seeds at this checkpoint, GPUs ~100%,
  zero failure markers); ON wave chains automatically per GPU. Telemetry →
  telemetry/shapley_ab_n5/. Monitor + launcher notifications armed.
- **SCORING PREREQUISITE (PDR-0022): the TOPUP `episode_idx` decode.** Runs at
  6dd80716 stamp the batch index into `episode_idx`; true id = stored × 12 + env_id
  (flagged ADDENDUM in the scoring prereg doc). Any per-episode join over TOPUP
  events — criterion (v) especially — MUST apply it. Per-event reads unaffected.
- Gate criteria 4–6 (entrenchment monitor, dormancy recheck, first-ON-run residuals
  incl. MANDATORY tau recalibration) execute during/after the ON runs per the prereg.

## Scoring rules (pre-registered — do not relitigate at scoring time)
Primary (i)–(v); Δcorr floor; G2 backstop (3× paired fossilize rate), G3 HARD (10%
efficiency, provisional pending a banked paired noise floor — bank it before scoring
or state its absence), G4 paid-event review trigger (>36/run forces adjudication);
std<1.0 confound gate; G1-abort pairing rule (aborted seed = pair dropped — the
paired-J analyzer now handles dropped pairs correctly, esper-lite-03a5351609); a
NULL n=5 fires NOTHING (asymmetric null, PDR-0019). Decode note above applies.

## Open questions / blocked-on-owner
- None for the runs. Standing: north-star/rent TARGET placeholders (metrics.md);
  enabling scale>0 BEYOND this experiment stays owner-gated (gate criteria 4–7);
  G3's paired noise floor for the efficiency read not yet banked (drl MINOR-3).

## Last checkpoint did (checkpoint #11)
- **PDR-0022:** external 8-finding review dispositioned — all verified, 3 bugs fixed
  (TDD, real-data replays reproduce banked results bit-for-bit), CI-gate hygiene
  cleared (all 3 AGENTS.md gates green), gate-2 estimand annotated NOT changed
  (GATE 2 PASS shown robust to the behavior-policy translation). A/B ruled NOT
  invalidated; the decode deviation flagged in the prereg, never silent.
- Tracker: 5 items filed and closed same-session (esper-lite-e780981fe7,
  -03a5351609, -72fb7074ca, -5d0c1620d7, -da189467f1); enablement-gate scoring path
  unblocked. Commits bbc8c6bc, 9fb674bb, 4a0a0d14, 18518f29, fd3e96d5 (+ 6c65792e
  tooling sync). Banked PDR-0009 and GATE-2 results audited — both stand.

## Next session, start here
**Monitor the A/B runs** (Karn: run_dir filter telemetry/shapley_ab_n5/). On
completion: verify commit hash, score STRICTLY per the prereg doc **applying the
episode_idx decode**, run the ON tau recalibration (P99 vs 0.56pp trigger), bank or
state absence of the G3 noise floor, bring the verdict + G4/confound reads to the
owner. If any ON run G1-aborts: drop the pair, investigate before re-run. Queued
behind the A/B decision: advantage-pathology / EV track (owner sequence).
