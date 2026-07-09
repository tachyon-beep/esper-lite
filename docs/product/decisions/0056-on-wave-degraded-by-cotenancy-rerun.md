# PDR-0056 — ON wave degraded by co-tenant analysis workload; full rerun at same commit

Date: 2026-07-10
Status: accepted (run-authorization grant; experiment-value principle applied)

## What happened

ON seed 41 trained to completion (200 rounds, final scoreboards printed) but exited
rc=1 on the fail-loud teardown check: `requested telemetry backend(s) degraded
(dropped events): DirectoryOutput`. Evidence store losses: **199/200 PPO updates,
2392/2400 episode outcomes** (101–200 events dropped in one burst). Seed 42 (still
running at diagnosis) shows the same drop signature (≥101 events, same minutes).
The gpu0 queue aborted per design; seeds 43/45 never started.

## Root cause — operational, not code

Drop bursts timestamped 04:40–04:43 AEST — exactly when the owner-requested
high-effort code review ran on this box: 8 finder agents + 8 verifier agents doing
multi-GB greps over telemetry (including reads of the live `events.jsonl` files the
trainers were appending), repeated full test suites, and DuckDB scans. The telemetry
writer thread starved; the bounded queue overflowed; DirectoryOutput dropped.
Corroboration: the OFF wave ran the identical code on a quiet box overnight with
ZERO drops; disk has 202G free; no tracebacks; training math unaffected.

The teardown check and the §6 coverage gates (hardened this very morning —
`n_rows`-per-env would flag the 8 missing outcomes) worked exactly as designed:
degraded evidence fails loud instead of scoring.

## The call

1. **Quarantine** both partial ON arms to `telemetry/stage2_ab_on/degraded-cotenancy-1/`
   (per the run-integrity convention; visible, never scorable).
2. **Rerun the full 5-seed ON wave at the SAME commit** (no code change — the cause is
   operational, so same-commit discipline with the OFF arms is preserved). Same
   session-proof launchers, same device pairing (41/43/45 cuda:0; 42/44 cuda:1).
3. **No telemetry-queue code hardening now** — a queue-size bump would put the ON arms
   on a different commit than OFF (PDR-0048: never score cross-commit pairs). Queue
   capacity / blocking-writer hardening is banked as a post-Stage-2 TIP item.

## Standing operational rule (the lesson)

**No heavy disk/CPU work on this box while evidence runs are writing.** That includes:
review-agent fleets, multi-GB greps, DuckDB scans over telemetry, full test suites,
and ANY read of a live events.jsonl. Light log-tails only. All analysis, scoring, and
review work queues until the wave completes. (Extends the PDR-0051 launcher rule:
that one made runs survive session death; this one makes them survive ME.)

## Consequences for the timeline

Wave rerun ≈ 18.5h (gpu1 done ~18:30 today, gpu0 ~00:45 tomorrow). The post-freeze
sequence (penalty-schedule fix → drl review → clean 600-round diagnostic, PDR-0055)
and the packet scoring both shift behind wave completion — scoring is itself a heavy
DuckDB read and now explicitly waits for all arms to finish. Cost accepted under the
experiment-value principle: a degraded arm that happens to squeak past the update
budget is a 20% instrument; the rerun is the 100% one.

## Reversal trigger

If the rerun ALSO drops events on a quiet box, the co-tenancy attribution is wrong —
stop, treat as a telemetry-writer defect, and take the cross-commit rerun path
(fix + rerun BOTH waves) rather than iterating on operational guesses.

## Provenance postscript (rerun launch, 05:52)

The rerun launched at commit `a0ad968e` (HEAD; the day's harness/docs commits landed
between waves). Training-path tree diff vs the OFF arms' commit `fe177844` across
simic/{training,agent,rewards}, tolaria, kasmina, tamiyo, nissa, scripts, leyline, and
both arm configs: **exactly one file — `leyline/episode_outcome.py`, +11 lines of
module-level contract CONSTANTS** (unread by any training code; all other changes are
read-path harness/tests/docs). Training behavior is byte-equivalent; empirically, the
rerun's epoch-1 losses are bit-identical to the OFF arms' (s41 2.3090133666992188,
s42 2.2991979122161865). §10 pairing compares run config from telemetry, not commits;
this note pre-declares the commit difference so the packet's §0 provenance block reads
as expected, not as a surprise.
