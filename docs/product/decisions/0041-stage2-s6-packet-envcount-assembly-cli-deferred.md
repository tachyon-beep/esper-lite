# PDR-0041 — Stage-2 MAJOR-1 acceptance: S6 verdict packet + env-count validity + assembly (CLI deferred to S7)

Date: 2026-07-07   Status: accepted (within grant: build + accept acceptance-harness code; a DISPATCH-level scope refinement of PDR-0039's slice map)
Author: Claude (Opus 4.8)   Owner sign-off: n/a (owner dispatched "carry on" this session; the scope correction was advisor-reviewed before any test was written)
Related: PDR-0039 (slice map S1–S7 — this executes S6 and refines it), PDR-0040 (S5 + dual-review disposition), gate doc `docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`, task esper-lite-2a4b56e719, epic esper-lite-f25b71c165. Commit `890ba364`.

## Context

PDR-0039's slice map called S6 "CLI + markdown verdict packet". Building it (TDD) surfaced a scope
contradiction, confirmed by an advisor pass before any code: **a runnable CLI cannot be honestly
driven in S6.** A CLI must construct `RunMeta`, which requires `actor_advantage_source` — a column
that does not exist in telemetry until S7 emits it. The only S6-available sources for that field are
(i) reading a nonexistent column (fails-loud on every invocation) or (ii) taking it from the CLI's
JSON spec — the *trusted-flag* pattern PDR-0039 explicitly rejected in favour of telemetry
verification. Per the TDD/verify discipline, one does not ship an entry point that cannot be driven.

## Options considered

1. **Keep the CLI in S6, read `actor_advantage_source` from the spec.** Rejected — reintroduces the
   trusted flag PDR-0039 replaced with the `ev_*`-signature verification; the CLI would be
   un-drivable end-to-end until S7 anyway.
2. **Keep the CLI in S6, read the (nonexistent) column.** Rejected — always fails loud; a
   non-functional entry point is not a slice.
3. **Scope S6 = the three drivable pieces; move the runnable CLI to S7.** Chosen — S6 lands the pure
   packet, the env-count validity, and the scoring assembler (all synthetic/duckdb-testable now);
   the CLI lands in S7 next to `read_run_meta` + the emission that make it drivable.

## The call — S6 = three pieces (CLI → S7)

1. **`render_packet` (pure).** Markdown verdict packet over the already-scored artifacts (no I/O, no
   torch): §7 verdict + tier caveat (an n=5 clean run is `SCREEN_PASS`, **never** a banked ACCEPT),
   §3/§4 leg results + frozen δ/ε_rel + per-seed Δ, §5 MECH, §6 G1–G4, §8B floored distribution +
   the per-stream-blindness caveat, §4 covariates + confound-downgrade note, §9
   diagnostic-scalars-**UNAVAILABLE** marker, §0 provenance **verbatim**. INVALID lists every §1
   breach and withholds all interpretation.
2. **env-count-completeness validity** (`read_run_n_envs` + `env_count_completeness_reasons`).
   Backstops Finding 1: the per-env-terminal G1/G2 mean silently averages over *whatever envs are
   present*, so the pair is rejected unless each leg's `episode_outcomes` covers all `runs.n_envs`
   distinct `env_id`s AND `ON n_envs == OFF n_envs`. It reads `runs`/`episode_outcomes` — telemetry
   the pure `validate_pair` never sees — so it lives at the assembly layer and its reasons are merged
   into the `Validity` passed to `score()` (whose INVALID short-circuit is the seam). **DEFINITIONAL:**
   hard-reject-on-incomplete is the conservative stance; the exact threshold rides with the G1/G2
   definitions **pending §6/drl confirmation**.
3. **`SeedPairing` + `build_report` (scoring-phase assembler).** Reads both legs' `ppo_updates` →
   `leg_series`, runs `validate_pair` + merges env-count reasons, reads the §6 paired safety deltas
   (ON − OFF), and `score`s against the **already-frozen** thresholds. It **never calibrates** (δ is
   frozen input — "no peeking" is mechanical) and **never reads `actor_advantage_source`** from
   telemetry (**caller-supplied `RunMeta`**, option a). `g3_hold`/`g4_hold` are **injected**: the G4
   guard-channel reader and the churn threshold are the S7/§6-definitional tail — the assembler
   refuses to fake safety it cannot yet read.

**236 telemetry-suite tests green** (incl. 11 render + 6 env-count + 4 build_report), ruff + mypy
clean, torch-free, no training-code touch. Module homes: `render_packet` in
`stage2_acceptance_packet.py` (pure); the readers/assembler in `stage2_acceptance_io.py` (duckdb).

## Rationale

Keeping the CLI out of S6 preserves the telemetry-verified-pairing invariant (a validated invariant
beats a trusted flag) and keeps the single training-code touch confined to S7, where the mandated
drl/pytorch review lands. The pure/impure split kept the packet fully synthetic-fixture-testable and
the env-count/assembly duckdb-testable — the same discipline that made Finding 1 catchable.

## Reversal trigger

- If S7's `read_run_meta` / emission shows `build_report`'s caller-supplied-`RunMeta` seam is wrong
  (e.g. the `runs` view can carry the pairing identity such that telemetry verification is
  unnecessary) → revisit the assembly boundary before freeze.
- The env-count hard-reject threshold and the G1/G2/G3/G4 definitions remain **pending §6/drl
  confirmation** — fold into the S7 specialist review, before freeze.
- **Advisor check-2 (PDR-0039) still open:** the §9 emission's OFF-leg byte-identity must be verified
  by a runnable GPU-free test before S7 lands.
- Unchanged: gate **NOT-FREEZABLE until S7**; **no paired A/B before freeze** (MAJOR-1 pre-registration).
