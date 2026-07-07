# Current State — Esper        Checkpoint: 2026-07-07 22:08 AEST · commit `f0ee27fe` · checkpoint #28 (PDR-0043; on `feat/ev-stab-stage2-hra`)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet) is at **Stage-2 MAJOR-1
acceptance** (esper-lite-2a4b56e719, in_progress). The acceptance harness is now
**7/7 slices built and review-remediated** through `f0ee27fe`, but it is **not an
experimental read, not frozen, not pushed, and not A/B-ready**.

The metric this leg ultimately moves remains host-accuracy contribution via the
Stage-2 MAJOR-1 gate; no paired fresh-init HRA ON/OFF result has been read yet.

## In flight
- **Stage-2 acceptance (esper-lite-2a4b56e719, in_progress):** pure scorer (PDR-0037),
  wrapper pure layer (PDR-0039), S5 reader + review fixes (PDR-0040), S6 packet/env-count
  assembler (PDR-0041), S7 emission/provenance/readers/G4/CLI (PDR-0042), and the
  follow-on review-remediation commits `7c6ec9da`, `0a2f6b9c`, `f0ee27fe` (PDR-0043).
  Latest hardening covers telemetry-derived G2 `host_params`, terminal per-env evidence
  coverage, duplicate `runs` row rejection, fail-loud required-field completeness, S7
  diagnostic scalar plumbing, G3 fossilize churn, and clearer return-variance diagnostics.
- **Branch survivor pick (esper-lite-1f1e55f58f, blocked by Stage-2):** unchanged
  (PDR-0038). Do not merge/delete/push while Stage-2 remains open.

## Facts the next session must not relitigate
- **Wrapper architecture settled — PDR-0039.** Caller-supplied pairing, telemetry-verified
  ON/OFF signature, type-enforced freeze order, and training-touch-last ordering stand.
- **S5/S6 reader and assembler seams settled — PDR-0040/PDR-0041.** G1/G2 are
  per-env-terminal; env coverage is assembly-layer validity; `build_report` never
  calibrates; `BuiltReport.validity` must be threaded into `render_packet`.
- **S7 built, then review-remediated — PDR-0042/PDR-0043.** The harness is no longer
  blocked on the known S7 review findings, but code-complete is not value-landed. No A/B
  before gate freeze.
- **Verification baseline remains mixed:** focused Stage-2 + adjacent regression suites pass;
  touched-file ruff, leyline lint, diff check, and wardline pass. Full mypy, full ruff,
  defensive-pattern lint, and GPU-sync lint still report existing non-Stage-2 failures.
- **Branch:** stay on `feat/ev-stab-stage2-hra`. Do not push without an explicit owner ask.

## Open questions / blocked-on-owner
- **Gate freeze:** after one final net-diff review, freeze the gate doc before any ON/OFF
  paired A/B. This is a product/owner gate, not an automatic consequence of code passing.
- **Launch:** paired fresh-init HRA ON/OFF A/B remains owner-launched only.
- **Threshold placeholders:** `Δparam_max`, burn-in `W`, north-star target/date, rent ceiling,
  and host-accuracy floor still need owner-set falsifiable values where the gate requires them.
- **Standing owner gates:** no push, tag, release, branch deletion, telemetry deletion, or remote
  action without explicit approval.

## Last checkpoint did (checkpoint #28)
- **PDR-0043** recorded that the S7 review gate moved from "findings outstanding" to
  "known findings remediated and committed." The latest commit is `f0ee27fe`.
- Updated `metrics.md` and `roadmap.md` so the product workspace no longer says only S7 is pending.
- No roadmap horizon changed: EV-stabilization remains Now; reward-efficiency statistics remain Next.
- No new experimental metric reading landed; Stage-2 MAJOR-1 is still pending a frozen paired run.

## Next session, start here
Run one final review of the net Stage-2 acceptance diff through `f0ee27fe`, focused on whether
the remediated evidence contracts are now freeze-ready. If clean, freeze the gate doc and then
prepare the owner-launched paired fresh-init HRA ON/OFF A/B. Use targeted verification first:
Stage-2 telemetry/packet/CLI tests, Karn/leyline adjacent tests, touched-file ruff, leyline lint,
and wardline. Keep full-repo red gates reported as baseline unless they move into the Stage-2 diff.
