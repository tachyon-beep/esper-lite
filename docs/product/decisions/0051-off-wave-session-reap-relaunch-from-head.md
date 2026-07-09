# PDR-0051 — OFF wave killed by session reap; relaunched from HEAD with session-proof launcher

Date: 2026-07-08 (decision); recorded at checkpoint 2026-07-09
Status: accepted (owner-ratified via explicit choice)
Decider: owner (john); diagnosis by owner-agent

## Context

The first OFF wave (launched 11:34 2026-07-08 from Claude session `97d358f5` as
background Bash tasks) died at 16:04:04 — both trainers AND wrapper shells SIGKILLed
same-second, no exit banners, seeds 43/44/45 never started. Diagnosis: the launcher
session ended at 16:04:08 (owner closed VS Code) and the harness reaped its background
process tree. Journal clean (no OOM/oomd). This retroactively explains the Stage-0
control-run reap PDR-0033 could only classify as "external SIGKILL". Partials
(s41 ~149 updates, s42 ~145 — just under the 150-scored floor before burn-in)
quarantined to `telemetry/stage2_ab_off/killed-by-session-reap-1/`.

## Options

(a) Relaunch all 5 seeds from HEAD `fe177844` (frozen `04b58ed4` + 2 training-inert
gate-fix commits) — provenance tip == branch tip. (b) Relaunch from `04b58ed4` via
detached checkout (strictest pre-registration reading). (c) Hold.

## The call

**(a) Relaunch from `fe177844`** with a session-proof launcher: `setsid`/`nohup`,
scripts stored under `telemetry/stage2_ab_off/` (not session-scoped /tmp), own session
ids, reparented to init. Verified training within minutes; wave completed 5/5 valid
(~6h/run, not the estimated 9–13h).

## Rationale

The two extra commits are provably training-inert (TypedDict key, TUI, lint whitelist,
docs); same-commit discipline is about training-path identity, which is preserved.
Keeping run provenance == branch tip avoids a detached-checkout special case for the
ON wave.

## Standing lesson (operational rule)

**Never launch multi-hour GPU runs as Claude-session background tasks.** Always
setsid/nohup (or tmux) with scripts outside session scratchpads. Verified environment
guarantees: logind `KillUserProcesses` default-no, `Linger=yes` — only reboot or
explicit kill can stop a detached wave.

## Reversal trigger

None — operational. If a future wave dies with the session-proof launcher in place,
diagnose fresh (do not assume session reap).
