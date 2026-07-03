# PDR-0022 — external code-review disposition: A/B validity preserved, deviations flagged, estimand critique rejected as post-hoc

Date: 2026-07-04   Status: accepted (within grant — no outward-facing action; the
running experiment was not touched)   Author: Claude (agent)
Related: PDR-0019, PDR-0021; tracker esper-lite-e780981fe7, -03a5351609,
-72fb7074ca, -5d0c1620d7, -da189467f1 (all closed); commits bbc8c6bc, 9fb674bb,
4a0a0d14, 18518f29, fd3e96d5;
docs/analysis/2026-07-03-shapley-ab-scoring-preregistration.md (decode ADDENDUM);
docs/analysis/2026-07-02-gate2-learnability-probe-preregistration.md (estimand ADDENDUM).

## Context
Owner supplied an external (ChatGPT) review of the branch's recent changes — 8
findings — while the paired n=5 OFF/ON A/B was RUNNING at frozen launch commit
6dd80716. All 8 were verified against code and the documented CI gates before
acceptance (session discipline: external analysis is input, not verdict). All 8
were factually real; three needed reframing (mypy baseline was already red; the
gate-2 "fix" would have been a methods violation; one bug had two contamination
surfaces, not one).

## The calls
1. **The running A/B is NOT invalidated and was not restarted.** The one finding
   touching run telemetry (TOPUP `episode_idx` stamped with the batch index) is
   recoverable at read time because the payload independently carries `env_id`:
   `episode_idx_true = stored × 12 + env_id`, exact for every batch. Fixes land
   on the branch for future runs; the in-flight runs stay at 6dd80716
   (PDR-0015 same-commit rule).
2. **Pre-registration deviations are FLAGGED, never silent.** The decode formula
   was written into the scoring prereg as a dated ADDENDUM (commit bbc8c6bc);
   scoring MUST apply it for any per-episode join over TOPUP events.
3. **Scoring-blockers were enforced structurally:** the two defects that could
   corrupt scoring reads (episode_idx decode; the paired-J analyzer's
   zip-misalignment armed by the pre-registered G1-abort pair-drop) were wired
   as tracker dependencies of the enablement gate before fixing. Both fixed
   (TDD; real-data replays reproduce banked results bit-for-bit) and closed —
   the gate's scoring path is unblocked.
4. **The gate-2 estimand critique was REJECTED as a post-hoc estimand change.**
   The probe's unfloored readout is pre-registered, identical in both arms, and
   deliberately more sensitive (the 0.15 op floor censors below-floor movement).
   Resolved by annotation: probe docstrings, a dated prereg ADDENDUM, and a
   result-doc caveat recording the behavior-policy translation — control ~0.13
   floor-pinned, retro ~0.21 above floor, behavioral ΔP ≈ +0.06 — **the GATE 2
   PASS is robust to the estimand distinction** (fd3e96d5). A supplementary
   floored column is recipe-documented for future reruns, never a replacement.
5. **Banked results audited and stand:** r1_n5 causal verdict (PDR-0009)
   reproduced bit-for-bit under the fixed analyzer; the K=30 Tier-2 campaign is
   orphan-free (all 10 pair files present) so the GATE 2 result was never
   exposed to the orphan bug.

## Options considered
(a) Pause/restart the A/B on the telemetry defect — rejected: the defect is
bijectively decodable, the arms are internally consistent, and a restart burns
a day of GPU and the same-commit pairing for zero information gain.
(b) Fix silently and decode at scoring without flagging — rejected: violates the
prereg's own deviation rule.
(c) Adopt the reviewer's gate-2 fix (floor the readout) — rejected: post-hoc
estimand change after the campaign ran; the annotation records why.

## Reversal trigger
If at scoring time the decoded episode ids fail to reconcile with the
episode-scoped views (any TOPUP event whose decoded id has no matching episode
row, or a mid-run partial batch is discovered), STOP scoring, re-derive ids from
raw events, and treat criterion (v) as unread until reconciled. If any future
gate-2 rerun shows the unfloored/floored readouts DISAGREE on direction (not
magnitude), the estimand annotation escalates to a methods review before that
rerun is scored.
