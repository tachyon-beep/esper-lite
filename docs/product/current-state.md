# Current State — Esper        Checkpoint: 2026-07-07 (checkpoint #25 — S5 impure reader + dual-review hardening published [PDR-0040]; on feat/ev-stab-stage2-hra)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet), at **Stage-2 acceptance**
(esper-lite-2a4b56e719, in_progress). Building the MAJOR-1 telemetry-acceptance harness that must
FREEZE before the paired HRA-ON/OFF A/B can run. Metric it ultimately moves: host-accuracy
contribution (guardrail, +7.69pp n=6) via the Stage-2 MAJOR-1 acceptance gate (metrics.md).

## In flight
- **Stage-2 acceptance (esper-lite-2a4b56e719, in_progress — 5/7 slices):** pure scorer DONE
  (42 tests, PDR-0037); wrapper pure layer S1–S4 DONE (PDR-0039); **S5 impure duckdb reader now
  DONE + dual-review-hardened (PDR-0040):** `stage2_acceptance_io.py` — `read_leg` fail-loud
  dict→typed boundary + §6 safety readers (G1/G2/G3). Dual review (drl-expert + external) found &
  fixed Findings 1–4 (**F1 = a verified-live soundness bug: the G1/G2 hard-REJECT gate was reading
  one env, not an env-average**; F3 = §4 confound-downgrade added; F4 = §8F(i) per_head_norm fence).
  **56 wrapper + 216 telemetry-suite green, torch-free, NO training-code touch.** **REMAINING:**
  **S6** CLI + markdown verdict packet (tier-gated, mirror `proof_packet`; env-count-completeness
  validity lands here — needs `runs.n_envs`) · **S7** emission (§9 scalars + `actor_advantage_source`,
  ON-leg-only; GPU-free byte-identity OFF-leg test; G4 reader + lr/entropy schedule-match; drl+pytorch
  review). **Gate NOT freezable until S7; no A/B before freeze.**
- **Branch survivor pick (esper-lite-1f1e55f58f, blocked by Stage-2):** unchanged (PDR-0038).

## Facts the next session must not relitigate
- **Wrapper architecture settled — PDR-0039** (caller-supplied pairing, telemetry-verified via the
  `ev_*` signature; type-enforced freeze order; §8B total-level exclusion + emitted distribution
  caveat). Do NOT re-derive.
- **S5 reader + Findings 1–4 disposition settled — PDR-0040.** G1/G2 use per-env-terminal
  aggregation; §4 confound-downgrade in; §8F(i) `per_head_advantage_norm` fence in. Do NOT re-open.
- **Branch:** continue on ev-stab; survivor pick DEFERRED to Stage-2 completion (PDR-0038). Do NOT
  merge / delete / push.
- **Verification gate is BROKEN** (full pytest wedges on GPU test_data_opt/test_dual_ab). Verify via
  TARGETED tests (`uv run pytest tests/simic/telemetry/test_stage2_acceptance*`) + ruff/mypy — never
  "tests-green".
- **Stage-2 HRA ON-leg code complete** (PDR-0037); this is an ACCEPTANCE task.

## Open questions / blocked-on-owner
- **Nothing NEW escalated this session.** Grant holds as written (re-confirmed 2026-07-07).
- **Standing owner gates (not now):** any push to origin (local ev-stab is ahead by S5 + hardening
  + this checkpoint; local `main` still 2 ahead of origin — all unpushed); the survivor pick /
  reconciling merge / branch deletion / jettison salvage audit (PDR-0038).
- **Freeze-before-ON placeholders:** `Δparam_max` (G2 ceiling) and burn-in `W` still owner-set
  (δ/ε_rel are computed by `calibrate_off` on the OFF arms at run time); north-star / rent /
  host-acc-floor TARGETs still owner-set (metrics.md).
- **S7 gate (advisor check 2, still open):** the §9 emission's OFF-leg byte-identity must be verified
  by a runnable GPU-free test BEFORE it lands.
- **Definitional (pending §6/drl):** G3/G4 "materially elevated" thresholds; the G1/G2 definitions
  are flagged for §6 confirmation.

## Last checkpoint did (checkpoint #25)
- **PDR-0040** — published the drift: recorded the S5 impure reader (commit `49536a08`) + dual-review
  hardening (`424bc7e2` Findings 1–3, `e019665c` Finding 4) that had landed in code post-checkpoint-#24
  but were unwritten. metrics.md Stage-2 row advanced (S5 + dual-review). No roadmap horizon change
  (EV-stab stays Now). No new experimental readings; no reversal trigger tripped.
- Tracker: Stage-2 task stays in_progress (5/7 slices; S6 + S7 remain); progress comment logged.

## Next session, start here
**S6 — CLI + markdown verdict packet** (tier-gated, mirror `scripts/proof_packet.py` structure):
assemble RunMeta + LegSeries → `validate_pair` → `calibrate_off` (OFF arms) → `score` → render the
§-structured markdown packet; the **env-count-completeness validity gate lands here** (needs
`runs.n_envs`). Then S7 emission (specialist review + GPU-free byte-identity check) → freeze the
gate doc → paired fresh-init HRA ON/OFF A/B (owner launches). Pointers: modules
`stage2_acceptance_packet.py` (pure) + `stage2_acceptance_io.py` (duckdb) + scorer
`stage2_acceptance.py`; gate doc `docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`;
reference `scripts/proof_packet.py`; PDR-0039/0040; memory `ev-stab-stage2-impl-state`.
