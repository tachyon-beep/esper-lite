# PDR-0055 — Confounded diagnostic tossed pre-start; clean redesign under the experiment-value principle

Date: 2026-07-10
Status: accepted (first application of the owner's experiment-value principle)
Supersedes: PDR-0054 (the salvaged read-window design; the transient-vs-structural
QUESTION and the pre-committed outcome readings carry over unchanged)

## The principle being applied (owner, 2026-07-10, recorded in vision.md)

"I'd rather toss a week and start over with a new experiment that gives us 100% than
save a near-done run that gives us 20%." Test: will this give us insights that inform
future decisions — ruling things in/out or verifying a theory?

## Context

PDR-0054 salvaged the schedule-confounded 600-round diagnostic by shrinking the read
window to rounds 251–451 (the only all-schedules-constant span) and retracting the
determinism cross-check. That salvage was made under the constraint that the confound
(horizon-normalized entropy-floor penalty schedule) could not be fixed while the
training path was frozen. The freeze lifts TONIGHT when the ON wave completes —
which makes a clean instrument buildable for ~7–10h of delay. Under the owner
principle, the salvaged 60%-instrument loses to the clean 100%-instrument.

## The call

1. **Cancelled the queued diagnostic BEFORE it started** (waiter killed ~05:20; zero
   data produced; config + launcher retained on disk).
2. **Post-freeze fix (tonight, after the ON wave completes + drl-expert review):**
   make `_get_penalty_schedule`'s breakpoints ABSOLUTE in update rounds, pinned to the
   current 200-round shape — boost 1.5× for rounds < 50, 1.0× for 50–150, decay
   1.0→0.5 over 150–200, hold 0.5× thereafter; breakpoints as leyline constants;
   retire `total_train_steps` from the schedule (its only reader). **Shape-preserving
   property:** every 200-round run (the completed A/B arms AND any future n=10 arms)
   behaves bit-identically, so comparability across tiers is untouched; longer runs
   become true continuations instead of different systems.
3. **Relaunch the 600-round OFF diagnostic (seed 41)** on the fixed code. What the
   clean instrument buys over the salvage:
   - **350 all-schedules-flat rounds (250–600)** vs 200 (entropy flat ≥ 250, penalty
     flat ≥ 200) — vs the salvage's 251–451 window with a boost-phase caveat.
   - **True continuation semantics:** rounds 0–200 run the same schedule as the
     existing seed-41 OFF arm — a low-sensitivity determinism corroboration (GPU
     nondeterminism caveat per the PDR-0035 replay lessons; not claimed bitwise).
   - **No external-validity caveat** against the A/B arms.

## Pre-committed reading (carried from PDR-0053/0054, on the better window)

- EV/Var(returns)/cf-scale stationary within rounds 250–600 → non-stationarity was
  schedule-transient → a post-anneal scored window can re-power LEG-B and re-pose the
  cf-plateau criterion in future pre-registrations.
- Continued drift at constant entropy + constant penalty → structural morphogenetic
  churn → no horizon fixes it; mechanistic claims need detrended/relative metrics.

## Costs and discipline

- Cost: diagnostic answer arrives ~13:00 2026-07-11 instead of ~06:00 (+7h), cuda:1
  idles ~12:15–19:30 today. Accepted under the principle.
- The schedule fix is a TRAINING-PATH change: lands only after the ON wave completes
  (freeze intact), with drl-expert review per CLAUDE.md, and cannot affect the A/B
  read (scoring is read-path over already-complete runs).
- If drl-expert review rejects the shape-preserving fix for cause, fall back to
  PDR-0054's salvaged window on unmodified code (still decision-informative, just
  weaker) — do not ship an unreviewed training change to save the timeline.

## Reversal trigger

None beyond the fallback above — the cancelled run had no data; nothing is lost but
time already priced in.
