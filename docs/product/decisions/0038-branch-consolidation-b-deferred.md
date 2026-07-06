# PDR-0038 — Branch consolidation: continue on ev-stab; defer a winner-takes-all survivor pick to Stage-2 completion (B-deferred)

Date: 2026-07-06   Status: accepted   Author: Claude (Opus 4.8)   Owner sign-off: yes (owner chose B-deferred + winner-takes-all end-state this session)
Related: current-state.md, roadmap.md (EV-stab = Now), PDR-0030 (experiment-disposition discipline), PDR-0034 (Stage-0 drained to main), PDR-0036 (prior branch-topology consolidation)

## Context
`/own-product` kept re-bootstrapping the product workspace on every fresh session because
`docs/product/` lived **only on `main`**, and the owner boots on the working branch
`feat/ev-stab-stage2-hra`, where the existence check read ABSENT. Investigating that surfaced
a larger question — "which branch is the branch" — so we measured the `main` ↔
`feat/ev-stab-stage2-hra` divergence:

- **Merge-base 2026-06-21**; the two branches then developed **fully in parallel for 15 days**.
- **main: 101 unique commits** (Shapley/PIN-E A/B era + CI hardening + Stage-0 *drain* [PDR-0034]
  + all product checkpoints #1–#22); **ev-stab: 61 unique commits** (Stage-2 HRA + finished
  sanctum/karn UI + entropy Phase-1 + current Stage-0 originals).
- A trial merge conflicts on **13 files, all in core `simic`/`tamiyo`/`leyline` training code**
  (`vectorized_trainer.py` ±529, `ppo_coordinator.py` ±188, `partition.py`, `reward_variance.py`, …).
- **Triage of main's 20 core-training commits:** ~3 are drain-duplicates of ev-stab's own Stage-0;
  ~7 are **parked-Shapley/PIN-E** (`shapley_synergy_scale=0.0`, owner-gated — dormant); ~4 are
  general/CI-hygiene fixes; the rest are superseded/early (incl. an old Stage-2 HRA commit
  `575d2560` via `0.3.0`, and an entropy fix ev-stab's Phase-1 supersedes). **ev-stab holds all
  the *live* work; main's unique code is mostly dormant or duplicated.**
- **The verification gate is broken:** the full `pytest` suite is unreachable here — it wedges on
  the pre-existing GPU `test_data_opt`/`test_dual_ab` hangs (why the pre-push gate was reframed to
  a module-importer gate). So a "merge, gate on tests-green" plan is false comfort for exactly the
  RL code most at risk.

## Options considered
1. **A — Retire-at-completion (do nothing now).** Keep Stage-2 on ev-stab; merge at completion.
   Pro: trunk stays accepted-only. Con: the branch-juggling + workspace fork the owner is
   frustrated by persist through all of Stage-2.
2. **B-now — full merge onto main today.** Hand-resolve all 13 conflicts, specialist-review the
   merged loop, gate on a real check + smoke. Pro: single trunk immediately. Con: integrates mostly
   **dormant** work (drain-dup + parked-Shapley) through delicate RL code on a **broken test gate**,
   then forces building the telemetry wrapper on a freshly-stitched base.
3. **B-deferred — decide canonical now, integrate at Stage-2 completion.** Declare main canonical;
   keep localized Stage-2 work on ev-stab; do the one full merge when the code is finished + stable.
   Pro: same destination, reconciliation happens once on stable code with a real gate + specialist
   review; nothing blocks the wrapper. Con: two branches persist through the (bounded) defer.
4. **C — Flip trunk to ev-stab.** Rejected: inverts the origin/published-trunk convention for no gain.

## The call
**B-deferred — winner-takes-all, survivor decided later.**
- **Long term, exactly ONE of {`feat/ev-stab-stage2-hra`, current `main`} survives as the trunk;
  the other is jettisoned (archived).** This is *not* a commitment to reconcile the two histories —
  it is a commitment to pick a winner and discard the loser (salvaging only must-keep scraps via
  cherry-pick).
- **The survivor is NOT decided now.** It is picked at **Stage-2 completion / the acceptance
  verdict**: if ev-stab's live Stage-2 work validates, **ev-stab wins** (main archived); if it does
  not, **main may win** and ev-stab's *finished* sub-parts (e.g. the sanctum UI) are salvaged.
- **`main` is the *current* trunk of record** (origin, published, all provenance/checkpoint
  history) — but *current* ≠ *guaranteed survivor*.
- **We continue on `feat/ev-stab-stage2-hra` now** — all live work is there.
- **Informed jettison — mandatory.** The loser is NOT discarded blind. Before jettison, run a
  **salvage audit**: enumerate the losing branch's *unique* code (`merge-base..loser`, filtered to
  code), classify each piece **KEEP** (cherry-pick into the survivor) vs **DROP** (with a stated
  reason), and get **owner sign-off on the KEEP/DROP manifest**. Only then are the KEEPs ported and
  the loser archived. Jettisoning code is fine; jettisoning it *uninformed* is not. (The archive ref
  is the reversible backstop, but "we can dig it out of the archive later" is not a substitute for
  the audit — the point is to decide deliberately while the context is fresh.)
- **Re-init fixed** by vendoring `docs/product/` onto ev-stab (`6db68192`) as a bounded snapshot of
  main @ #22. During the defer the **living workspace runs on ev-stab** (where the owner boots and
  `/own-product` resumes). At the survivor pick the workspace simply *is* the surviving branch's; if
  main wins, ev-stab's newer PDRs/current-state are ported back. Bounded fork, defined resolution
  point — not two-sources-of-truth-forever.
- **Archives are the jettison/recovery net**: `archive-main-2026-07-06` @`2d296b6c`,
  `archive-stage2-hra-2026-07-06` @`6db68192`, plus `backup/ev-stab-stage2-hra-2026-07-06` @`2fad1fc4`.
  Whichever branch is jettisoned survives immutably as an archive ref.
- **Absorbed branches** `feat/sanctum-layout` and `feat/entropy-thermometer-fix` (0 unique commits —
  fully contained in ev-stab) + their worktrees are redundant; **prune at the survivor pick.**

## Rationale
Merging now integrates mostly **dormant** code (drained Stage-0 + parked-Shapley) through the most
delicate RL training files, on a **test gate that does not run**, and would have me build new
Stage-2 telemetry on a base stitched together the same hour. The one honest argument for "now" —
"the conflict only grows" — is weak here: main's remaining trajectory is dominated by
**non-conflicting** `product:`/docs checkpoints, and ev-stab's remaining Stage-2 work is
**localized** (`ppo_agent.py` + `simic/telemetry`). Deferring lets the **survivor pick** happen when
the acceptance verdict actually tells us which line is worth keeping — rather than laboriously
reconciling two live histories now and only later discovering we kept the losing one. If the endgame
needs a reconciling merge (not a clean jettison), it still gets the safeguards it requires
(a runnable gate + specialist review) against finished, stable code.

## Reversal trigger
Reopen this defer decision if **any** of:
- **(a)** `main` accretes *new* conflicting work in core `simic/training`, `simic/rewards`, or
  `tamiyo` before Stage-2 completes — the "smaller now" premise breaks; reassess B-now.
- **(b)** ev-stab is found to *need* one of main's CI-hygiene/robustness fixes to keep passing its
  own gate — then **cherry-pick those specific commits now**, do not wait for the full merge.
- **(c)** Stage-2 is abandoned or parked — re-decide disposition per PDR-0030 (merge finished
  sub-parts, e.g. the sanctum UI, or fork/abandon).

The eventual survivor pick is itself **gated** on: (i) a verification gate that *actually runs* —
the specific `simic`/`tamiyo` training+reward tests that complete in this environment **plus** a
short `train ppo` functional smoke proving the surviving tree still trains; (ii) **drl-expert /
pytorch-expert review** of the surviving training loop (CLAUDE.md mandate for `simic`); and
(iii) a **completed salvage audit with an owner-signed KEEP/DROP manifest** for the loser's unique
code (see "The call") — jettison does not happen before this. Jettisoning the loser, any reconciling
merge, git history rewrite / branch deletion, and any push to origin are **owner-gated** (grant:
never push without an explicit ask).
