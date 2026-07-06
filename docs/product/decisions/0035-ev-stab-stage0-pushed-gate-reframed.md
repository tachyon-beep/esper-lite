# PDR-0035 — EV-stab Stage-0 drain PUSHED to origin/main; pre-push gate reframed to a drained-module importer gate

Date: 2026-07-06   Status: accepted (owner-authorized in-session: "ok lets merge to main").
Executes the OWNER-GATED push reserved in PDR-0034. Author: Claude (agent).
Related: PDR-0034 (drain landed on local main, push reserved with a pre-push-regression reversal
trigger), PDR-0031 (byte-identity golden = named OFF-leg detector), PDR-0028 (value-free gate).
Tracker: esper-lite-f25b71c165.

## Context

PDR-0034 landed the 5 value-free Stage-0 commits on LOCAL main (`50af6643`) and reserved the push
to `origin` as OWNER-GATED, with an explicit reversal trigger: *"if a pre-push full-suite run
surfaces a regression the targeted sweep missed → fix on main before push."* The owner authorized
the push. The promised belt-and-braces full-suite run was launched and **never returned a verdict**:
it wedged for 3.5h wall / ~2m CPU (state `Sl`, stuck at 52%) on the pre-existing GPU-pipeline tests
`test_data_opt.py` and `test_dual_ab.py`, which hang deterministically under the machine's current
three-way pytest contention. "Wait for full-suite green" had stopped being a *reachable* gate.

## Options

- **(a) Reframe the gate to the drained-module IMPORTER set** — every test importing
  `partition` / `reward_variance` / `action_execution` / `vectorized_trainer` — and push if green.
  A change's regression surface *is* its importer set; tests outside it are provably immune.
- **(b) Keep waiting / re-run the full suite.** Rejected: under current contention it re-wedges on
  the next GPU test — the gate is unreachable, not slow. `pytest-timeout` is not installed.
- **(c) Push on the already-completed verification alone** (golden 2/2 + 410 tests + pytorch GO).
  Rejected as insufficiently explicit: the drain's own test subdirs sat in the unreached 48%; the
  importer gate closes exactly that hole with a stated finding.

## Call

**(a).** Killed the wedged run (PID 19210). Confirmed the hang is environmental — `test_data_opt.py`
imports **no** drained module (it's a CUDA/CIFAR gather-iterator test). Enumerated importers
(grep + `merge-base` containment) and ran the **complete importer set** on the drained `main` HEAD:
**862 passed / 0 failed / 0 error** in 13.5s. Pushed `2f95aa04..50af6643` to `origin/main` under the
**tachyon-beep** identity (no `git config` / `gh auth` touched); origin advanced, ahead 0 / behind 0.
`current-state.md` and `metrics.md` had already been refreshed in-session and pushed (`007de8c6`).

## Rationale

A change cannot regress a test that does not import it, so the importer set is the **complete,
bounded** regression surface — and an environmental hang *outside* that set hides nothing. This
converts an unreachable "whole-tree-green" gate into a specific, defensible finding ("862 importer
tests pass") that actually discriminates push vs. no-push. The push itself was pre-decided
(PDR-0034) and owner-authorized; only the *verification method* changed, and it was reframed toward
the same losslessness-proof discipline used for the branch retirements (PDR-0036), not relaxed.

## Reversal triggers

- If a later full-suite run (once contention clears, or the GPU hangs are quarantined) surfaces a
  failure whose test **imports a drained module** → that is a real regression: fix on `main`
  immediately (origin already carries the drain).
- If `test_data_opt.py` / `test_dual_ab.py` ever begin importing a drained module (they do not
  today) → the "environmental, not a regression" premise breaks; re-gate to include them.
