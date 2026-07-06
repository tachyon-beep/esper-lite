# PDR-0036 — Branch topology consolidated 12→7; codex fork-base archived (not abandoned); EV research corpus salvaged to main

Date: 2026-07-06   Status: accepted (owner-driven in-session: "Sort branch topology first",
"flag that branch as a long term archive", "lets merge them to main", "great, lets do it").
Applies the PDR-0030 experiment-disposition discipline. Author: Claude (agent).
Related: PDR-0030 (force explicit merge/fork/abandon at every experiment end; a fork must stand
alone), PDR-0031 (prior 4→2 consolidation). Tracker: esper-lite-f25b71c165.

## Context

Post-drain the repo carried **12 local branches / 8 worktrees** — mostly dead `codex/*` and
`post-p01-*` / `wf_*` branches fully absorbed into `main`, two detached A/B experiment worktrees
(`ab-frozen`, `ab-on-relaunch`), plus the live EV stack. A concurrent owner session was actively
landing TUI work on `feat/ev-stab-stage2-hra` (which had been ff'd onto the `feat/sanctum-layout`
tip). PDR-0030 requires an explicit disposition per experiment line; the tangle was overdue.

## Options (per branch class)

- **Dead absorbed branches** (`codex/*`, `post-p01-*`, `wf_*`): abandon — nothing unique.
- **`codex/sanctum-pre-ready-crash`**: holds a 35-file oracle-sandbox WIP **and** a unique EV
  research corpus absent from `main`. Abandon-whole loses both. → abandon / merge-to-main / archive.
- **`ab-*` detached worktrees**: abandon — but a detached HEAD is the *only* ref pinning its commits.
- **Live stack** (`ev-stab` + `sanctum` + `entropy`): leave — actively worked.

## Call

- **Retired 6 fully-absorbed branches + 5 stale/experiment worktrees**, EACH proven lossless before
  deletion — `merge-base --is-ancestor <b> main` for the branches, and **patch-id equivalence** for
  the detached `ab-on-relaunch` commits (`d6514416`/`f98b6b3e` are byte-identical diffs to `main`'s
  `c2e500d3`/`bbc8c6bc`). `ab-frozen`'s only "uncommitted" content was a `data/` symlink.
- **`codex/sanctum-pre-ready-crash` → ARCHIVE-FORK** (owner call): renamed
  **`archive/oracle-sandbox-wip-2026-06`** and pushed to `origin` as a durable long-term archive
  (preserves the oracle-sandbox WIP + fork base). Its unique EV research corpus — the 4
  `docs/research/` variance-reduction reports + `reward-redesign-methodology` +
  `ppo-learning-gate-reward-efficiency` plan + 2 `docs/analysis/` runsheets, **all absent on main** —
  was **SALVAGED to `main`** (`127f25cb`), a clean additive commit (files absent ⇒ no conflict).
- **Live stack left intact** as ONE unit; retires together when it merges to `main` after Stage-2.
- **Kept** `0.3.0` (release) and `backup/0.1.1-pre-p01`. Result: **7 branches / 3 worktrees**.

## Rationale

Archive-fork (not abandon) honors PDR-0030's "a fork must stand alone and be justified": the
oracle-sandbox WIP is unfinished but not worthless, and a pushed `archive/` ref preserves it
durably without cluttering the active branch set. Salvaging the research corpus fixes a real
provenance gap — the `ev-variance-research-verdict` memory cited `docs/research/` reports that were
not actually on `main`. Every deletion was gated on a losslessness proof (ancestor / patch-id),
never a hunch — the same discipline as PDR-0035's importer gate.

## Reversal triggers

- If the oracle-sandbox WIP is resumed → branch off `origin/archive/oracle-sandbox-wip-2026-06`
  (it stands alone); do NOT resurrect the deleted worktrees.
- If Stage-2 acceptance bounces and the live stack must NOT merge to `main` → the "retires as one
  unit" plan reopens; re-disposition `sanctum-layout` + `entropy-thermometer-fix` independently.
- If any retired branch is later found to hold unique work (it should not — all verified) → recover
  from reflog / origin (`codex/post-p01-plan-closeout`'s origin copy is intact).
