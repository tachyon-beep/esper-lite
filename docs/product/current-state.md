# Current State — Esper        Checkpoint: 2026-07-07 (checkpoint #24 — MAJOR-1 wrapper pure layer S1–S4 landed [PDR-0039]; on feat/ev-stab-stage2-hra)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet), at **Stage-2 acceptance**
(esper-lite-2a4b56e719, in_progress). This session BUILT the pure layer of the MAJOR-1 telemetry
wrapper — the "remaining half" of the acceptance gate from PDR-0037/0029. Metric it ultimately
moves: host-accuracy contribution (guardrail) via the Stage-2 MAJOR-1 acceptance gate (metrics.md).

## In flight
- **Stage-2 acceptance (esper-lite-2a4b56e719, in_progress):** pure scorer DONE (42 tests,
  PDR-0037). **Wrapper PURE LAYER S1–S4 now DONE** (PDR-0039, commit `da2c5f00`): series
  primitives · `leg_series` per-leg reduction · `validate_pair` (§1 gates) · `calibrate_off`/`score`
  freeze-order split + §0 provenance. **80 tests green (38 wrapper + 42 scorer), ruff+mypy clean,
  torch-free, NO training-code touch.** Module `src/esper/simic/telemetry/stage2_acceptance_packet.py`.
  **REMAINING (the impure tail):** S5 duckdb reader (dict→UpdateRow at boundary, fail-loud;
  `runs`→RunMeta + `episode_outcomes`→G1/G2/G3/G4) · S6 CLI + markdown packet (mirror
  `proof_packet` structure) · S7 emission (§9 scalars + `actor_advantage_source`; ON-leg-only,
  byte-identity OFF-leg, drl/pytorch review). **Gate NOT freezable until S7; no A/B before freeze.**
- **Branch survivor pick + cleanup (esper-lite-1f1e55f58f, blocked by Stage-2):** unchanged (PDR-0038).

## Facts the next session must not relitigate
- **Wrapper architecture is settled — see PDR-0039** (caller-supplied pairing verified via the ev_*
  telemetry signature, NOT a flag; type-enforced freeze order; §8B total-level exclusion + emitted
  distribution caveat; `actor_advantage_source` as provenance constant). Do NOT re-derive it.
- **Branch:** continue on ev-stab; survivor pick DEFERRED to Stage-2 completion (PDR-0038). Do NOT
  merge / delete / push.
- **Verification gate is BROKEN** (full pytest wedges on GPU test_data_opt/test_dual_ab). Verify via
  TARGETED tests (`uv run pytest tests/simic/telemetry/test_stage2_acceptance*`) + ruff/mypy — never
  "tests-green".
- **Stage-2 HRA ON-leg code is complete** (PDR-0037); this is an ACCEPTANCE task. Pure scorer +
  wrapper pure layer done; only the impure tail (S5–S7) + freeze + A/B remain.

## Open questions / blocked-on-owner
- **Nothing NEW escalated this session.** Grant holds as written (re-confirmed 2026-07-07).
- **Standing owner gates (not now):** any push to origin (local ev-stab is ahead by `da2c5f00` +
  this checkpoint; local `main` still 2 ahead of origin — all unpushed); the survivor pick /
  reconciling merge / branch deletion / jettison salvage audit (PDR-0038).
- **Freeze-before-ON placeholders:** `Δparam_max` (G2 ceiling) and burn-in `W` still owner-set
  (δ/ε_rel are computed by `calibrate_off` on the OFF arms at run time); north-star / rent /
  host-acc-floor TARGETs still owner-set (metrics.md).
- **S7 gate (advisor check 2, still open):** the §9 emission's OFF-leg byte-identity must be verified
  by a runnable GPU-free test BEFORE it lands.

## Last checkpoint did (checkpoint #24)
- **PDR-0039** — MAJOR-1 wrapper architecture + pure layer (S1–S4). Committed code `da2c5f00` on
  feat/ev-stab-stage2-hra (NOT pushed). metrics.md Stage-2-gate row advanced (pure layer built). No
  roadmap horizon change (EV-stab stays Now). No new experimental readings; no reversal trigger tripped.
- Tracker: Stage-2 task stays in_progress (4/7 slices); progress comment 143 logged.

## Next session, start here
**S5 — the duckdb telemetry reader** for the wrapper (TDD): `read_leg(dir/conn, run_dir) ->
list[UpdateRow]` via `create_views` + `scan_ingestion_integrity` + `_rows`, ordered by
(inner_epoch, batch), dict→UpdateRow at the boundary (fail loud on missing column); plus `runs`→RunMeta
and `episode_outcomes`→G1/G2/G3/G4 readers. Then S6 CLI packet → S7 emission (specialist review +
byte-identity check) → freeze the gate doc → paired fresh-init HRA ON/OFF A/B (owner launches).
Pointers: module `stage2_acceptance_packet.py`; scorer `stage2_acceptance.py`; gate doc
`docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`; reference `scripts/proof_packet.py`;
PDR-0039; memory `ev-stab-stage2-impl-state`.
