# Current State — Esper        Checkpoint: 2026-07-10 (early am) · checkpoint #36 (PDR-0052..0056; on `feat/ev-stab-stage2-hra`)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet): the **5-seed ON RERUN is in
flight** (launched 05:52 2026-07-10 @ `a0ad968e`; training tree byte-equivalent to the
OFF arms' `fe177844` — PDR-0056 provenance postscript; gpu1 done ~18:40, gpu0 ~00:50
2026-07-11). The first ON wave was DEGRADED by co-tenant analysis load starving the
telemetry writer (operational, not code) and is quarantined. Metric = Stage-2 MAJOR-1
composite under the amended screen predicate **LEG-A ∧ MECH ∧ G1–G4** (LEG-B
descriptive per §11.1; cf-plateau descriptive per §11.2).

## QUIET-MODE RULE (in force until the wave completes)
**No heavy disk/CPU work on this box while evidence runs write** (PDR-0056): no
review-agent fleets, multi-GB greps, DuckDB scans, full test suites, or reads of live
events.jsonl. Log tails only. Packet scoring itself is a heavy DuckDB read — it WAITS
for wave completion.

## In flight
- **ON rerun (running):** logs `telemetry/stage2_ab_on/logs/`; session-proof
  launchers; devices mirror OFF partners; epoch-1 losses bit-identical to OFF
  (pairing sanity). Exit monitor armed. Degraded first-wave data in
  `telemetry/stage2_ab_on/degraded-cotenancy-1/` — never score it.
- **After the wave (in order):** (1) validity-check all 5 ON arms (200 updates,
  2400 outcomes, rc=0, no drop lines) → (2) extend `packet_root/` with on_s4X symlink
  dirs (events.jsonl + stdout log each) → (3) score spec (pairs; thresholds
  delta=0.050311, eps_rel=0.10, tau_acc=0.3, delta_param_max=1527.8, w=17,
  budget=150; g3_ratio_max=1.5, g4_ratio_max=2.0, g4_abs_floor=2) →
  `scripts/stage2_packet.py score` → **read verdict from leg_a/mech_hold/g1–g4
  fields (composite verdict field superseded at screen tier, §11.1)**; the packet now
  carries the §11.2 cf-warmup descriptive block → (4) bring the owner the read →
  (5) freeze lifts → penalty-schedule fix (absolute rounds pinned to the 200-round
  shape; drl-expert review REQUIRED) → clean 600-round seed-41 OFF diagnostic
  (PDR-0055; fallback = PDR-0054 salvaged window on unmodified code).
- **Queued (owner: "slow and steady", nothing before the read):** TIP
  (esper-lite-c62d4891b0) vs reward-efficiency (esper-lite-a2abff5ec5) at the
  post-Stage-2 DECIDE.

## Facts the next session must not relitigate
- **§11.2 (PDR-0052):** cf-plateau is DESCRIPTIVE; hard gates = cf_value_loss
  presence (ON) / absence (OFF) + finiteness on scored updates. No re-gating without
  a new pre-registration.
- **Acceptance reader hardened** (d1349158 + f78cf095 + 710aa760): outcome bounds
  live in leyline (FINAL_ACCURACY_MIN/MAX, PARAM_RATIO_MIN...), consumed by harness +
  karn view; duplicate-terminal/NULL-env guards; §11.2 block in render_packet;
  validity-vs-consumption scope lesson encoded in comments. 315 telemetry tests.
- **PDR-0055/0056 rulings:** confounded diagnostic tossed pre-start (zero data);
  degraded ON wave quarantined; reruns at same training tree. PDR-0056 reversal
  trigger ARMED: if the rerun drops events on a quiet box, co-tenancy attribution is
  wrong → telemetry-writer defect path (fix + rerun BOTH waves cross-commit).
- **Owner grant (vision.md, 2026-07-10):** create/extend/add runs at discretion
  (PDR + pre-committed reading each); experiment-value principle (toss for 100% over
  saving 20%; test = informs future decisions). Acceptance gates stay owner-gated.
- **Branch:** stay on `feat/ev-stab-stage2-hra`; ~17 commits unpushed; no push
  without an explicit owner ask.

## Open questions / blocked-on-owner
- **Confirm vision.md grant wording** (owner dictated both additions in-session;
  the file is recorded, not silently rewritten — please skim the two new grant
  paragraphs and the experiment-value principle for fidelity).
- **Packet read (standing gate):** verdict comes to the owner; ACCEPT bankable only
  at n=10.
- **Filigree dashboard :9189 still down** (esper-lite-29233429a9); wardline
  boundary adoption pending same issue.
- North-star target/date, rent ceiling, host-accuracy floor: owner-unset.
- Standing: no push/tag/release/branch-deletion/telemetry-deletion/remote action.

## Last checkpoint did (checkpoint #36)
- Ran the owner-requested high-effort review of the 6-hour window: 10 verified
  findings → all fixed same-session (harden batch d1349158, reviewer-found crash
  paths f78cf095, fixture consolidation 710aa760).
- §11 cf-plateau gap: implemented faithfully, proved unsatisfiable on real data,
  owner demoted to descriptive (§11.2, PDR-0052).
- Diagnostic arc: schedule confound found and corrected pre-data (PDR-0054), then
  tossed pre-start for a clean post-freeze redesign under the owner's new
  experiment-value principle (PDR-0055); principle + broadened run grant recorded in
  vision.md.
- First ON wave degraded by my own review workload (co-tenancy) → quarantined,
  same-tree rerun launched and verified; quiet-box rule made standing (PDR-0056).

## Next session, start here
Check the ON rerun (`telemetry/stage2_ab_on/logs/wave_gpu*.out`). If 5/5 clean
(rc=0, no "dropped" lines, 200/2400 counts) → run the After-the-wave sequence above.
If any arm dropped events on the quiet box → PDR-0056 reversal trigger: STOP, treat
as telemetry-writer defect, do not iterate operationally. Honor QUIET-MODE until the
wave is done.
