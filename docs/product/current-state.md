# Current State — Esper        Checkpoint: 2026-07-07 (checkpoint #27 — S7 built at `9666fe0c`, pending specialist review [PDR-0042]; on feat/ev-stab-stage2-hra)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet), at **Stage-2 acceptance**
(esper-lite-2a4b56e719, in_progress — **7/7 slices code-complete, pending specialist review**).
The MAJOR-1 telemetry-acceptance harness is built through S7 at `9666fe0c`, but it is **not
accepted, not frozen, and not A/B-ready** until the holistic + `pytorch-expert` + `drl-expert`
review finishes and findings are addressed. Metric it ultimately moves: host-accuracy contribution
(guardrail, +7.69pp n=6) via the Stage-2 MAJOR-1 acceptance gate.

## In flight
- **Stage-2 acceptance (esper-lite-2a4b56e719, in_progress — 7/7 built, review gate open):** pure scorer (PDR-0037) +
  wrapper pure layer S1–S4 (PDR-0039) + S5 impure reader & dual-review-hardened (PDR-0040) + **S6 now
  DONE (PDR-0041, commit `890ba364`):** `render_packet` (pure §-structured markdown verdict packet),
  `env_count_completeness_reasons`/`read_run_n_envs` (env-count validity — backstops Finding 1),
  `SeedPairing`/`build_report` (scoring-phase assembler; caller-supplied RunMeta, never calibrates /
  never reads `actor_advantage_source`). **236 telemetry-suite green, torch-free, NO training-code
  touch.** **S7 BUILT (PDR-0042, commit `9666fe0c`):** §9 `value_main_target_scale` /
  `cf_value_target_scale` emission inside the HRA-ON block, strict reducer whitelist, leyline
  `actor_advantage_source` provenance pipe into `TrainingStartedPayload` + runs view, `read_run_meta`,
  G4 guard-channel reader, explicit G3/G4 threshold predicates, and runnable `scripts/stage2_packet.py`
  (`calibrate` / `score`, fail-closed JSONL corruption path). **Code-complete is not accepted:**
  holistic + `pytorch-expert` + `drl-expert` review must confirm/reject OFF-leg byte identity,
  G1/G2/G3/G4/env-count definitions, G4 reader, and the training-code touch. **Gate NOT freezable
  until review findings are addressed; no A/B before freeze.**
- **Branch survivor pick (esper-lite-1f1e55f58f, blocked by Stage-2):** unchanged (PDR-0038).

## Facts the next session must not relitigate
- **Wrapper architecture settled — PDR-0039** (caller-supplied pairing, telemetry-verified via the
  `ev_*` signature; type-enforced freeze order; §8B total-level exclusion + emitted caveat). Do NOT re-derive.
- **S5 reader + Findings 1–4 disposition settled — PDR-0040** (per-env-terminal G1/G2, §4
  confound-downgrade, §8F(i) fence). Do NOT re-open.
- **S6 settled — PDR-0041:** render_packet is pure; env-count is an assembly-layer check merged into
  the Validity passed to `score()`; **the runnable CLI is S7, not S6** (RunMeta needs
  `actor_advantage_source`, unreadable honestly in S6). `build_report` never calibrates (no peeking)
  and takes caller-supplied RunMeta; g3/g4 holds are injected. **`build_report` returns
  `BuiltReport{report, validity}`** — the S7 CLI MUST thread `built.validity` into `render_packet`
  (else an INVALID packet lists no breaches; review fix `13589813`). Do NOT re-scope.
- **S7 built, not accepted — PDR-0042:** `9666fe0c` delivered the emission/provenance/readers/G4/CLI
  tail, and the task is now review-gated rather than implementation-gated. Do NOT freeze, close, or
  launch A/B until holistic + drl + pytorch review findings are resolved.
- **Branch:** continue on ev-stab; survivor pick DEFERRED to Stage-2 completion (PDR-0038). Do NOT
  merge / delete / push.
- **Verification gate is BROKEN** (full pytest wedges on GPU test_data_opt/test_dual_ab). Verify via
  TARGETED tests (`uv run pytest tests/simic/telemetry`) + ruff/mypy — never "tests-green".
- **Stage-2 HRA ON-leg code complete** (PDR-0037); this is an ACCEPTANCE task.

## Open questions / blocked-on-owner
- **Nothing NEW escalated this session.** Grant holds as written (re-confirmed 2026-07-07).
- **Standing owner gates (not now):** any push to origin (local ev-stab is ahead by S5 + hardening +
  S6 `890ba364` + review-fix `13589813` + checkpoints #25/#26; local `main` still 2 ahead of origin
  — all unpushed); the survivor pick / reconciling merge / branch deletion / jettison salvage audit (PDR-0038).
- **Freeze-before-ON placeholders:** `Δparam_max` (G2 ceiling) and burn-in `W` still owner-set
  (δ/ε_rel computed by `calibrate_off` on OFF arms at run time); north-star / rent / host-acc-floor
  TARGETs still owner-set (metrics.md).
- **S7 review gate:** tracker says the 3-agent review is running. Treat the OFF-leg byte-identity
  evidence and the G1/G2/G3/G4/env-count definitions as **pending specialist acceptance** until the
  review output is in hand.
- **Definitional (pending §6/drl — fold into the S7 specialist review):** env-count hard-reject
  threshold; G1/G2/G3/G4 definitions + the G3/G4 "materially elevated" thresholds.

## Last checkpoint did (checkpoint #27)
- **PDR-0042** — reconciled live git/tracker drift after S7 landed at `9666fe0c`: Stage-2 is now
  **7/7 slices code-complete, pending review**, not 6/7 with S7 remaining. metrics.md Stage-2 row
  advanced to the review-gated state. No roadmap horizon change (EV-stab stays Now). No new
  experimental readings; no reversal trigger tripped.
- Tracker: Stage-2 task stays in_progress; latest progress comment says S7 is built and a 3-agent
  review is running.

## Next session, start here
**Review and accept/rework S7.** Collect the holistic, `pytorch-expert`, and `drl-expert` review
outputs for `9666fe0c` (changes since `f5f4c7f3`). Fix findings first, with targeted tests. The
review must explicitly resolve OFF-leg byte identity, G1/G2/G3/G4/env-count definitions, the G4
reader, and the training-code emission/provenance touch. Only after the reviewed code is clean:
freeze the gate doc → paired fresh-init HRA ON/OFF A/B (owner launches). Pointers: modules
`stage2_acceptance_packet.py` (pure) + `stage2_acceptance_io.py` (duckdb + assembler) + scorer
`stage2_acceptance.py`; script `scripts/stage2_packet.py`; gate doc
`docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`; PDR-0039/0040/0041/0042.
