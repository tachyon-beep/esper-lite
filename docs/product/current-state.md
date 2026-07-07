# Current State — Esper        Checkpoint: 2026-07-07 (checkpoint #26 — S6 verdict packet + env-count validity + assembler landed [PDR-0041]; on feat/ev-stab-stage2-hra)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet), at **Stage-2 acceptance**
(esper-lite-2a4b56e719, in_progress — **6/7 slices done, only S7 remains**). Building the MAJOR-1
telemetry-acceptance harness that must FREEZE before the paired HRA-ON/OFF A/B. Metric it ultimately
moves: host-accuracy contribution (guardrail, +7.69pp n=6) via the Stage-2 MAJOR-1 acceptance gate.

## In flight
- **Stage-2 acceptance (esper-lite-2a4b56e719, in_progress — 6/7):** pure scorer (PDR-0037) +
  wrapper pure layer S1–S4 (PDR-0039) + S5 impure reader & dual-review-hardened (PDR-0040) + **S6 now
  DONE (PDR-0041, commit `890ba364`):** `render_packet` (pure §-structured markdown verdict packet),
  `env_count_completeness_reasons`/`read_run_n_envs` (env-count validity — backstops Finding 1),
  `SeedPairing`/`build_report` (scoring-phase assembler; caller-supplied RunMeta, never calibrates /
  never reads `actor_advantage_source`). **236 telemetry-suite green, torch-free, NO training-code
  touch.** **REMAINING — S7 ONLY (the training-code tail):** emission (§9 scalars +
  `actor_advantage_source` near `ppo_agent.py:824`, ON-leg-only) + `read_run_meta` (runs view) + G4
  guard-channel reader + the runnable CLI (moved S6→S7, PDR-0041) + lr/entropy schedule-match;
  **GPU-free byte-identity OFF-leg test** (advisor check 2); **drl-expert + pytorch-expert review**.
  **Gate NOT freezable until S7; no A/B before freeze.**
- **Branch survivor pick (esper-lite-1f1e55f58f, blocked by Stage-2):** unchanged (PDR-0038).

## Facts the next session must not relitigate
- **Wrapper architecture settled — PDR-0039** (caller-supplied pairing, telemetry-verified via the
  `ev_*` signature; type-enforced freeze order; §8B total-level exclusion + emitted caveat). Do NOT re-derive.
- **S5 reader + Findings 1–4 disposition settled — PDR-0040** (per-env-terminal G1/G2, §4
  confound-downgrade, §8F(i) fence). Do NOT re-open.
- **S6 settled — PDR-0041:** render_packet is pure; env-count is an assembly-layer check merged into
  the Validity passed to `score()`; **the runnable CLI is S7, not S6** (RunMeta needs
  `actor_advantage_source`, unreadable honestly in S6). `build_report` never calibrates (no peeking)
  and takes caller-supplied RunMeta; g3/g4 holds are injected. Do NOT re-scope.
- **Branch:** continue on ev-stab; survivor pick DEFERRED to Stage-2 completion (PDR-0038). Do NOT
  merge / delete / push.
- **Verification gate is BROKEN** (full pytest wedges on GPU test_data_opt/test_dual_ab). Verify via
  TARGETED tests (`uv run pytest tests/simic/telemetry`) + ruff/mypy — never "tests-green".
- **Stage-2 HRA ON-leg code complete** (PDR-0037); this is an ACCEPTANCE task.

## Open questions / blocked-on-owner
- **Nothing NEW escalated this session.** Grant holds as written (re-confirmed 2026-07-07).
- **Standing owner gates (not now):** any push to origin (local ev-stab is ahead by S5 + hardening +
  S6 `890ba364` + checkpoints #25/#26; local `main` still 2 ahead of origin — all unpushed); the
  survivor pick / reconciling merge / branch deletion / jettison salvage audit (PDR-0038).
- **Freeze-before-ON placeholders:** `Δparam_max` (G2 ceiling) and burn-in `W` still owner-set
  (δ/ε_rel computed by `calibrate_off` on OFF arms at run time); north-star / rent / host-acc-floor
  TARGETs still owner-set (metrics.md).
- **S7 gate (advisor check 2, still open):** the §9 emission's OFF-leg byte-identity must be verified
  by a runnable GPU-free test BEFORE it lands.
- **Definitional (pending §6/drl — fold into the S7 specialist review):** env-count hard-reject
  threshold; G1/G2/G3/G4 definitions + the G3/G4 "materially elevated" thresholds.

## Last checkpoint did (checkpoint #26)
- **PDR-0041** — S6 landed (commit `890ba364`): render_packet + env-count validity + build_report;
  recorded the advisor-reviewed scope correction (runnable CLI moved S6→S7). metrics.md Stage-2 row
  advanced (6/7 slices). No roadmap horizon change (EV-stab stays Now). No new experimental readings;
  no reversal trigger tripped.
- Tracker: Stage-2 task stays in_progress (6/7 slices; S7 remains); progress comment 145 logged.

## Next session, start here
**S7 — the emission slice (LAST, training-code, specialist-reviewed).** Emit §9 diagnostic scalars +
`actor_advantage_source` near `ppo_agent.py:824` (ON-leg-only); add `read_run_meta` (runs view →
RunMeta) + the G4 guard-channel reader; wire the **runnable CLI** (`main()` over `telemetry_dir` +
pairing spec → build_report → render_packet, two-phase: calibrate OFF → freeze → score ON). MANDATORY
before it lands: a **GPU-free byte-identity OFF-leg test** (advisor check 2) + **drl-expert +
pytorch-expert review** (CLAUDE.md simic mandate). Then freeze the gate doc → paired fresh-init HRA
ON/OFF A/B (owner launches). Pointers: modules `stage2_acceptance_packet.py` (pure) +
`stage2_acceptance_io.py` (duckdb + assembler) + scorer `stage2_acceptance.py`; gate doc
`docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`; reference `scripts/proof_packet.py`
(CLI idiom); PDR-0039/0040/0041; memory `ev-stab-stage2-impl-state`.
