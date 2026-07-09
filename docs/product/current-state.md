# Current State — Esper        Checkpoint: 2026-07-09 (late) · checkpoint #35 (PDR-0049/0050/0051; on `feat/ev-stab-stage2-hra`)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet) is one wave from its read: the OFF
wave completed **5/5 valid**, §10 step-3 scalars are **FROZEN** (PDR-0050: W=17,
δ=0.050311, Δparam_max=1527.8, τ_acc=0.3pp), and the **5-seed ON wave is IN FLIGHT**
(launched 23:33 2026-07-09 @ `fe177844`, ~6h/run; gpu1 done ~12:15, gpu0 ~18:00
2026-07-10). The metric is the Stage-2 MAJOR-1 composite under the **AMENDED screen
predicate: LEG-A ∧ MECH ∧ G1–G4** — LEG-B is demoted to descriptive (both pre-registered
escalations fired: ε_rel spread 0.5716 ≥ 0.10; stationarity 2.4–13.6×; gate doc §11.1).

## In flight
- **Stage-2 ON wave (running):** logs `telemetry/stage2_ab_on/logs/`; session-proof
  launcher (setsid/nohup — PDR-0051 standing lesson: NEVER launch waves as
  Claude-session background tasks); devices mirror OFF partners (41/43/45 cuda:0,
  42/44 cuda:1 — required by §11.1 per-pair placement equality). Epoch-1 losses
  bit-identical to OFF (pairing sanity). Completion monitor armed (session-scoped).
- **After the ON wave (next session's work):** validity-check all 5 ON arms (200
  updates, 0 skips, tracebacks) → extend `telemetry/stage2_ab_off/packet_root/` with
  `on_s4X` symlink dirs → build score spec (pairs; thresholds: delta=0.050311,
  eps_rel=0.10, tau_acc=0.3, delta_param_max=1527.8, w=17, budget=150; g3_ratio_max=1.5,
  g4_ratio_max=2.0, g4_abs_floor=2) → `scripts/stage2_packet.py score` → **read the
  verdict from leg_a/mech_hold/g1–g4 fields (the emitted composite verdict includes
  LEG-B and is SUPERSEDED at the screen tier, §11.1)** → bring the owner the read.
- **Queued (owner ruled "slow and steady" — nothing starts before the Stage-2 read):**
  TIP (esper-lite-c62d4891b0, PDR-0049) and reward-efficiency stats
  (esper-lite-a2abff5ec5) compete at the post-Stage-2 DECIDE.

## Facts the next session must not relitigate
- **Step-3 is FROZEN (PDR-0050 + gate doc §11.1).** Do not reopen W/δ/Δparam_max or the
  LEG-B demotion; no retro-promotion even if LEG-B would have passed.
- **Calibration surface:** views key `run_dir` by BASENAME of events.jsonl's parent;
  `scan_ingestion_integrity` globs ONE level — hence the depth-1 symlink
  `packet_root/` (events.jsonl + stdout.log per run). Reuse it for ON.
- **Harness amendment landed** (RunMeta.placement split; per-pair placement equality;
  3 TDD tests; 182 harness tests green) — read-path only, training path untouched,
  same-commit discipline intact (ON and OFF both ran commit `fe177844`).
- **First OFF wave died by session reap** (owner closed VS Code; PDR-0051) — partials
  in `killed-by-session-reap-1/`, excluded from scoring; do not "recover" them.
- **Branch:** stay on `feat/ev-stab-stage2-hra`. No push without an explicit owner ask.

## Open questions / blocked-on-owner
- **Packet read (standing gate):** ON wave completes → I score and bring the verdict;
  ACCEPT is only bankable at n=10 (screen tier is direction-only).
- **Post-Stage-2 direction pick** (owner: "finish Stage-2, then pick"): TIP vs
  reward-efficiency vs whatever the packet motivates.
- **Filigree dashboard on :9189 refuses connections** (wardline finding-sink warning;
  esper-lite-29233429a9 has the probe details). Needs a restart decision — I did not
  restart shared infra mid-wave.
- **North-star target/date, rent ceiling, host-accuracy floor:** still owner-unset.
- **Standing owner gates:** no push, tag, release, branch deletion, telemetry
  deletion, or remote action without explicit approval.

## Last checkpoint did (checkpoint #35)
- Diagnosed the OFF-wave session-reap kill, relaunched session-proof, wave completed
  5/5 valid; PDR-0051 records the relaunch + the standing launcher rule.
- Ran §10 step-3: froze W/δ/Δparam_max; both pre-registered escalations fired → owner
  ruled LEG-B descriptive + device-placement provenance (PDR-0050, gate doc §11.1);
  harness amended TDD-style (182 tests green); official calibrate report saved.
- Launched the ON wave from the same commit as OFF; verified bit-identical epoch-1.
- Captured the Telemetry Improvement Program as a Next-bet intent (PDR-0049, epic
  esper-lite-c62d4891b0, concept doc) with the owner's "no early start" ruling.
- Closed the residual mypy/gpu-sync gates (commits `2ef832b5`/`fe177844`); filed
  wardline inertness (esper-lite-29233429a9).

## Next session, start here
Check the ON wave (`telemetry/stage2_ab_on/logs/wave_gpu*.out` — exit banners or
ABORTING). If 5/5 clean → validity-check, extend packet_root, build the score spec
from the §11.1 frozen thresholds above, run `stage2_packet.py score`, and read the
verdict per the AMENDED predicate. If a run died → diagnose before any relaunch
(PDR-0051: with the session-proof launcher, a death is NOT session reap — investigate
fresh). Nothing else starts before the read (owner ruling in PDR-0049).
