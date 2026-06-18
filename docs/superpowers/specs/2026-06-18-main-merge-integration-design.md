# 0.1.1 → main Merge Integration — Design

```yaml
date: 2026-06-18
status: ready
domain: release-engineering / integration
reviewers:
  - axiom-python-engineering (Python patterns, architecture)
  - drl-expert (P0-1 value-head break, recurrent PPO surface)
origin: >
  Sprint item (2). 0.1.1 is ~46 commits ahead of main and carries the
  checkpoint-breaking P0-1 op-independent V(s) baseline, the multi-epoch
  recurrent PPO refactor (PR1/PR2), the vectorized-trainer transaction-phase
  refactor, Tier-0 profiler, and the leyline contracts move. This spec defines
  how that body of work lands on main safely.
```

## Problem

`main` has fallen ~46 commits behind the active `0.1.1` branch. The accumulated
work spans the highest-risk area of the system (`simic` training + agent), the
shared-contracts module (`leyline`), the policy/value networks (`tamiyo`), and
the telemetry surface (`karn`). It also carries **one breaking change**: the
P0-1 value-head split bumps `VALUE_HEAD_SCHEMA_VERSION` to `2` and rejects all
pre-v2 PPO checkpoints with no remap.

The naive framing — "main is 42 commits behind, this is a big risky three-way
merge" — is **wrong**, and the entire integration plan hinges on correcting it.
We must define the actual merge topology, the checkpoint-break migration story,
the CI/test gating posture, the single-PR-vs-staged decision, conflict/rollback
handling, and how the dependency-bump pass (Sprint item 3) rides the same window.

## Key reframe

**This is a fast-forward, not a three-way merge. There is zero conflict surface.**

Verified facts (all reproducible at HEAD of `0.1.1`):

- `git merge-base main 0.1.1` = `f7f1aecea90cef5b6c557da94e91b8889e5efacb`.
- `git rev-parse main` = `f7f1aecea90cef5b6c557da94e91b8889e5efacb` — **identical**.
- `git merge-base --is-ancestor main 0.1.1` → exit 0 (main is an ancestor).
- `git rev-list --count main..0.1.1` = **46** (commits ahead).
- `git rev-list --count 0.1.1..main` = **0** (no commits on main absent from 0.1.1).
- `git diff --stat main..0.1.1` = **206 files changed, +21869 / −3078**.

Because `main`'s HEAD *is* the merge-base, there is no divergent history. No file
on `main` has changed since the branch point. **Conflict probability is zero by
construction.** The "42 / 46 commits behind" number is real, but it describes a
linear lead, not a fork. This collapses the risk model: there is no merge
*algorithm* risk (no conflicts to resolve, no semantic interleaving of two
edited copies of a file). The residual risk is entirely about (a) the
correctness of the 46 commits as a unit when CI runs against them on `main`'s
triggers, and (b) the operational consequences of the checkpoint break and the
new estimator's telemetry on downstream consumers.

## Recommended design

### 1. Land as a single fast-forward, preserving curated history

Use `--ff-only` so the 46 curated commit messages survive on `main` for
bisection (P0-1, PR1/PR2, the refactor sequence, P0-2/P0-3 are all valuable
bisect anchors):

```bash
git checkout main
git merge --ff-only 0.1.1
git push origin main
```

A `--no-ff` merge commit is **rejected** (see Rejected alternatives): it adds a
synthetic node with no value here and obscures the linear lineage.

Open the integration as a PR from `0.1.1` → `main` so the `pull_request`
CI jobs gate it before the push.

**The landing mechanic is a single locked path — the CLI fast-forward:**

```bash
git checkout main && git merge --ff-only 0.1.1 && git push
```

This preserves the 46 commit SHAs *byte-for-byte*, so the
`git rev-parse origin/main == git rev-parse 0.1.1` SHA-equality check
(acceptance criteria 1/2) holds exactly. **The CLI push bypasses the
`pull_request` trigger entirely** — the 75% coverage gate, lint, typecheck, and
overwatch-web jobs never run against the push itself. That is why the §6 hard
precondition requires `gh pr checks` to be confirmed all-green on the open PR
*before* the CLI push runs.

The GitHub "Rebase and merge" / "Squash" / "Create a merge commit" buttons are
**rejected**: the rebase button replays the 46 commits with new SHAs (breaking
SHA-equality verification), squash discards the 46-commit history this design
explicitly preserves, and a merge commit adds a synthetic node with no value.
The CLI fast-forward is the only acceptable mechanic.

**Locked mechanic choice:** mandate the **CLI fast-forward path** (`git checkout
main && git merge --ff-only 0.1.1 && git push`) with **SHA-equality
verification**, gated by the §6 hard precondition that `gh pr checks` is green
first. No GitHub merge-button path is permitted.

### 2. Checkpoint-break migration story (VALUE_HEAD_SCHEMA_VERSION = 2)

This is the only breaking change in the merge and must be called out
prominently in the PR body. The migration story is "**retrain, no remap**", and
that is intentional under the No-Legacy policy.

**The single break:**

- `VALUE_HEAD_SCHEMA_VERSION = 2` is defined at `src/esper/leyline/__init__.py:125`
  and exported at `:877`. It **does not exist on `main` at all**
  (`git show main:src/esper/leyline/__init__.py | grep VALUE_HEAD` → empty).
- The break is carried by this orthogonal `value_head_schema_version` field — it
  is **not** a `CHECKPOINT_VERSION` bump. `CHECKPOINT_VERSION = 2` is identical
  on both branches (`src/esper/simic/agent/ppo_agent.py:64` on 0.1.1; `:63` on
  main — both `= 2`). The schema-version field is validated *before* the
  `CHECKPOINT_VERSION` check.

**Enforcement anchors in `src/esper/simic/agent/ppo_agent.py`:**

- Import: `:51` (`VALUE_HEAD_SCHEMA_VERSION` from leyline).
- Written at save: `:1540` (`'value_head_schema_version': VALUE_HEAD_SCHEMA_VERSION`).
- Required-field read at load inside a `try` block: `:1656`.
- **KeyError path (the real-world path for genuine old checkpoints):** the
  `except KeyError` at `:1663` raises `RuntimeError("Incompatible checkpoint
  format: missing required field … Please retrain …")` at `:1664-1668`.
- **Schema-version check:** `:1675`, raising `RuntimeError` with the message at
  `:1676-1683` ("the value head was split into an op-INDEPENDENT
  state_value_head … there is no remap. Please retrain.").

**Which error path actually fires (precision correction):** A genuinely old
pre-P0-1 checkpoint lacks the `value_head_schema_version` *key entirely*, so it
hits the **KeyError path at `:1663-1668`**, NOT the schema-version check at
`:1675`. The schema-version check only fires for a checkpoint that was saved
*with* the field present but at an older numeric value — an effectively-impossible
window (the field and the version-2 value were introduced together in P0-1).
**The migration story still holds** — both paths raise a fast, clear,
retrain-actionable `RuntimeError` — but acceptance criterion 3's cited line
(`:1675`) is the wrong path for real checkpoints; the operative line is `:1664`.

**What the migration story must state explicitly:**

1. All pre-v2 PPO checkpoints are **rejected with a fast, clear error** — this is
   a deliberate consequence of the value-head architecture change (op-conditioned
   `Q(s,op)` baseline → op-independent `state_value_head` V(s)), not a regression.
2. **No migration script will be written and none should be.** A pre-v2
   checkpoint has `value_head.*` weights with no `state_value_head.*` /
   `q_head.*` counterpart; there is no structural remap (per the No-Legacy
   policy — a shim would be exactly the prohibited pattern). "Retrain" is the
   only path.
3. **Confirm before merge** that no in-flight or long-running run on `main`
   depends on a pre-v2 op-conditioned value-head checkpoint that cannot be
   restarted. (The 200-ep EV-liftoff experiment is already on the 0.1.1
   estimator, so this is expected to be a no-op confirmation — but it must be an
   explicit checklist item, not an assumption.) **Make this confirmation
   greppable rather than human-only** (see §2-check below).
4. The error message is already actionable and fails fast; no additional UX work
   is required.

**§2-check (concrete automated step, converts the human confirmation):** run

```bash
grep -rn 'load_checkpoint\|load_state_dict\|ppo_agent.*checkpoint' tests/ \
  --include='*.py' | grep -v __pycache__
```

and inspect any checkpoint-loading test fixture for pre-v2 schema-version
assumptions or a bundled `.pt` artifact saved before P0-1. There is currently
**no pre-v2 checkpoint fixture** in the tree (a fixture loading a V(s)=1
checkpoint would error loudly at load), but this must be confirmed by the grep at
execution time, not assumed — CI fixture generation has no visibility into this
spec and could restore a stale artifact without anyone noticing during a
human-only confirmation step.

Adjacent items that look like breaks but are **not** (call out to avoid false
alarm during review):

- `kasmina` slot `_SCHEMA_VERSION = 2` (`src/esper/kasmina/slot.py:305`) is
  byte-identical on both branches; `slot.py`'s diff only *adds* a
  `blending_delta` field — not a break.
- `src/esper/simic/training/normalizer_checkpoint.py` is **new** on 0.1.1 and
  uses `OBS_V3_FEATURE_SCHEMA_VERSION = 1`; it introduces no break on existing
  artifacts because it did not exist on main.

### 3. Divergence inventory by domain and risk

| Domain | Files (src/tests) | Notable commits | Risk |
|---|---|---|---|
| simic training + agent | ~23 src / 36 tests | `vectorized_trainer.py` transaction-phase refactor (`def707ee`..`19cb0e9a`, +2614/−1550); recurrent PPO PR1/PR2 (`356008c4`, `b9218273`); P0-1 V(s) (`6a27b8e3`, `177a53aa`, `31bf8cb7`); P0-2 (`cf2cf549`); P0-3 rollback (`cf57d94a`); Tier-0 profiler (`0ee51801`, `1af4e22b`) | **Highest** |
| leyline (contracts) | 7 src / 7 tests | `telemetry_contracts.py` move P1-7 (`6ab43965`); dead-field deletes P1-2/P1-8 (`035ba449`, `d812f058`); `VALUE_HEAD_SCHEMA_VERSION` added (P0-1) | **High** (cross-domain imports + the break) |
| tamiyo / networks | 3 src / 9 tests | `factored_lstm.py` +237 (`state_value_head` split, P0-1); `lstm_bundle.py` | Medium |
| karn telemetry + sanctum | 11 src / 8 tests | `mcp/views.py` +100; sanctum schema/aggregator; `health.py`; store/triggers — **intersects the EV-consumer audit (Sprint item 1); see the explicit file:line list in §4 below** | Medium |
| nissa | 2 src / 3 tests | `output.py` silent-swallow fix P1-1 (`38744ed6`) | Low |
| docs / scripts / dotfiles | 27 / 2 / 14 | specs, plans, `proof_packet.py` +1736, `ev_liftoff_experiment.py`, skills, `weft.toml` | Low |

The risk concentration is exactly where it should be expected (the RL training
core) — but because there is no conflict surface, "risk" here means **"does this
unit pass CI and behave correctly on main"**, not "will the merge mangle a file".

The karn/sanctum divergence row maps directly onto the EV-consumer audit
gate-fix targets enumerated in §4 (`narrative_panel.py`, `ppo_losses_panel.py`,
`critic_calibration_panel.py`, `aggregator.py`, and `karn/constants.py`). Those
are the specific files Sprint item 1 must touch before this merge can safely
land — this makes Locked Decision 5 concrete and auditable.

### 4. CI / test gating strategy

Single workflow: `.github/workflows/test-suite.yml`. Triggers: PR,
push-to-main, nightly cron 02:00 UTC. Job chain:

1. **lint** (~5m): leyline-type-boundary check, defensive-pattern whitelist,
   gpu-sync whitelist, ruff.
2. **typecheck** (~10m): `MYPYPATH=src uv run mypy -p esper`.
3. **property-tests** + **unit-and-integration-tests** (~15m): **75% coverage
   gate** (`test-suite.yml:128-129`; the threshold is 75% not 80% precisely
   because this job excludes integration/slow/stress). Unit marker
   `-m "not integration and not stress and not property and not slow"`, then a
   second pass `-m integration`.
4. **overwatch-web-tests** (~20m): npm build + vitest + Playwright chromium.
5. **nightly-full-suite** (cron only): thorough property, slow, stress.

Suite size: **385 test files** under `tests/` (verified: `find tests -name
'test_*.py' | wc -l` = 385; the sprint doc's "~384" figure is consistent and
authoritative). At this scale the full local suite (Step 3) is feasible in a
single pre-PR run — there is no CI-timeout or full-suite-feasibility concern.

Gating posture for this merge:

- **The push-to-main trigger means the fast-forward will itself run the full
  PR-equivalent suite on main.** That is a feature: it re-validates the 46-commit
  unit in main's context. **Caveat:** this only fires for the CLI push *if* the
  PR was opened and its `pull_request` jobs ran first — the CLI push itself does
  not trigger `pull_request`. See §6 precondition.
- **The nightly-only jobs (slow/stress/thorough-property) do NOT gate the PR.**
  A bare `uv run pytest` is **NOT** the full suite: `pytest.ini` (`:18-24`)
  injects `addopts = … -m "not integration and not stress and not property and
  not slow"`, so the default invocation silently excludes four of the five
  lanes. Before opening the PR, run each CI lane explicitly, mirroring
  `.github/workflows/test-suite.yml` exactly, with a `--collect-only`
  non-empty assertion per lane so a config/env/GPU-gating skip cannot make a
  lane hollow:

  | Lane | Command (mirrors CI) | CI anchor |
  |---|---|---|
  | default/unit | `uv run pytest -m "not integration and not stress and not property and not slow" --cov=src --cov-report=json:.coverage-reports/coverage.json --cov-report=term-missing` | `test-suite.yml:116` |
  | integration | `uv run pytest -m integration -v` | `test-suite.yml:137` |
  | property | `HYPOTHESIS_PROFILE=thorough uv run pytest -m property -v --hypothesis-show-statistics` | `test-suite.yml:193-197` |
  | slow | `uv run pytest -m "slow and not integration and not stress and not property" -v` | `test-suite.yml:203` |
  | stress | `uv run pytest -m stress -v` | `test-suite.yml:209` |

  **Per-lane collect-only assertion (run before each lane, must list >0 tests):**
  ```bash
  uv run pytest --collect-only -m integration -q | tail -3
  uv run pytest --collect-only -m property -q | tail -3
  uv run pytest --collect-only -m "slow and not integration and not stress and not property" -q | tail -3
  uv run pytest --collect-only -m stress -q | tail -3
  ```
  Running every lane locally is the single most important pre-merge gate: it
  covers the recurrent-PPO and transaction-phase refactor paths that the nightly
  cron would otherwise be the first to exercise.
- **Coverage uses `--cov=src`, NOT `--cov=esper`.** CI measures the 75% gate on
  the default/unit lane only, with `--cov=src` (`test-suite.yml:116`). Use the
  identical marker set and `--cov=src` target for the local gate check — a full
  multi-lane `--cov` run reports *higher* coverage and can mask a sub-75%
  unit-lane result.
- **Domain-scoped coverage check (drifting-goals guard).** The 75% project-wide
  threshold was set against a smaller codebase and is **not recalibrated** here.
  The 46-commit body adds the high-complexity `vectorized_trainer.py`
  transaction-phase refactor (+2614/−1550) and the P0-1 value-head split; a fixed
  project-wide gate lets the *effective* coverage of the riskiest new code drift
  down while the overall percentage holds. Add a domain-scoped check to Step 3
  (not a threshold change — a visibility guard):
  ```bash
  uv run pytest tests/simic/ --cov=src/esper/simic --cov-report=term-missing
  ```
  Assert simic coverage does not drop below its tracked baseline.

- **EV-consumer audit dependency (Sprint item 1) — first-class gate-fix targets.**
  Under the new op-marginal V(s), `explained_variance` (EV) std blows out on K>1
  runs with low return-variance batches; a denominator-cratered EV (e.g. −8)
  is an **artifact**, not a value-function failure. Every EV consumer keyed on
  EV mean/std will false-alarm unless guarded. The audit must treat the following
  as first-class gate-fix targets — **not just the anomaly_detector backend and
  the Karn DuckDB views**, but the Sanctum display widgets that drive
  operator-visible health labels:

  | File | Lines | What it gates |
  |---|---|---|
  | `src/esper/karn/sanctum/widgets/tamiyo/narrative_panel.py` | 320, 457, 501, 503, 512, 514, 601, 624 | raw EV vs `EXPLAINED_VAR_WARNING`/`EXPLAINED_VAR_CRITICAL`; drives the "FAIL:Value" / "WARNING:Value" status labels, no denominator guard |
  | `src/esper/karn/sanctum/aggregator.py` | ~980 | substitutes `0.0` for `None` EV — a denominator-cratered EV of −8 passes through as a *real* value and trips `EXPLAINED_VAR_CRITICAL ≤ 0.0` downstream |
  | `src/esper/karn/sanctum/widgets/tamiyo/ppo_losses_panel.py` | 356, 358 | EV status color |
  | `src/esper/karn/sanctum/widgets/tamiyo/critic_calibration_panel.py` | 142, 144 | EV style/status |
  | `src/esper/karn/constants.py` | 120-121 | `EXPLAINED_VAR_WARNING = 0.3`, `EXPLAINED_VAR_CRITICAL = 0.0` (the TUI thresholds these widgets compare against) |

  The anomaly_detector already has a `value_collapse_applicable` guard at
  `ppo_coordinator.py:523` that partially mitigates the `VALUE_COLLAPSE_DETECTED`
  backend event — but the Sanctum display gates above are **unguarded** and will
  false-alarm on every K>1 batch with low return variance. **Required fix shape:**
  `aggregator.py` must propagate a **low-variance flag** alongside the EV value so
  downstream widgets can differentiate a denominator-cratered artifact from a
  genuine value-function failure, rather than treating `None→0.0` as a real
  `≤ 0.0` reading.

  The EV-telemetry consumer-audit gate-fix **must land before or with this merge**
  so that K>1 runs on `main` do not immediately false-alarm. This is a sequencing
  constraint, recorded here and owned by Sprint item 1's plan.

### 5. Single-PR vs staged decision

**Decision: single fast-forward PR. Do not stage.**

Rationale:

- It is *mechanically* a fast-forward. Staging would require artificially
  splitting a linear history into sub-ranges — manufacturing intermediate states
  that never existed and that no commit was ever tested at, which only
  *introduces* risk (broken bisection, untested interleavings).
- There are no conflicts to stage around (zero divergence).
- The work is internally coherent and order-dependent: refactor → recurrent PPO
  → P0 fixes build on one another (e.g. P0-1's V(s) split depends on the
  value-head split landing first; `31bf8cb7` is a bugfix *on top of* `6a27b8e3`).
  Splitting the range would risk landing a fix ahead of the change it fixes.
- A single window lets the merge, the EV gate-fix, and the dependency bump pass
  share one CI validation pass.

### 6. Conflict and rollback handling

- **Conflicts:** none possible (FF). If `git merge --ff-only` *fails*, that means
  someone pushed to `main` after this spec was written and the topology is no
  longer a pure FF — **stop and re-run the merge-base verification** (§Key
  reframe) before proceeding. Do not force; do not `--no-ff` around it without
  re-assessing.
- **HARD PRECONDITION on the CLI push (closes the PR-CI-bypass gap):** the CLI
  fast-forward sequence (`git checkout main && git merge --ff-only 0.1.1 && git
  push`) **bypasses the `pull_request` trigger**, so the 75% coverage gate,
  lint, typecheck, and overwatch-web jobs never run against the push. The CLI
  push is therefore permitted **only after** confirming the PR's CI is already
  green:
  ```bash
  gh pr checks --repo tachyon-beep/esper-lite <pr-number>   # MUST be all-green
  ```
  This is a mandatory precondition of the push step, not advisory. If
  `gh pr checks` is not green, do not push. No GitHub merge-button path is an
  acceptable substitute — the CLI fast-forward is the single locked mechanic
  (§1).
- **Rollback:** because history is linear, rollback is a single
  `git revert`-of-a-range or a branch reset coordinated with the team. Keep the
  dependency-bump commit (§7) as the **last** commit so a dependency regression
  can be reverted independently of the functional merge without touching the
  46-commit body. **Per CLAUDE.md Git Safety, `git reset --hard` on `main`
  requires explicit user permission** — prefer a forward `revert` commit over
  history rewrite.
- **Failure isolation:** sequence the schema/checkpoint confirmation first, then
  the functional FF, then the dependency bumps last — so a flaky transitive
  dependency cannot block or be confused with the functional merge.

### 7. How the dependency bump pass rides this window

Sprint item 3 (Dependabot triage: 1 critical / 34 high / 32 moderate / 13 low)
folds into this merge window as the **last step**, after the 46-commit FF lands
and CI is green:

- **Python:** `uv lock --upgrade && uv sync`. The fixes are overwhelmingly
  resolvable by lockfile delta — only 4 of 71 Python alerts touch packages
  declared directly in `pyproject.toml`. This is a **dependency
  manifest/lockfile commit**, not strictly "lockfile-only": a `[tool.uv]`
  constraint added to clear a held transitive mutates `pyproject.toml`, and the
  npm step (below) mutates `package.json` + lockfile. **Runtime-dependency
  reclassification (correction):** `datasets>=4.4.1` is a **core
  `[project].dependencies` entry** (`pyproject.toml:11`), not dev/optional. Its
  transitive `pyarrow` alert (high, use-after-free, GHSA-rgxp-2hwp-jwgg) is
  therefore a **required-runtime transitive reachable in standard CIFAR
  installs** — escalate its priority; do not treat it as dev-only. By contrast,
  `fastapi` / `starlette` / `python-multipart` are runtime-**optional** (declared
  under extras), not dev-only either. Do **not** bump `transformers` to
  `5.0.0rc3` (pre-release). `torch` #80 has no fix yet (accept/snooze with a
  note).
- **`transformers` minor-bump guard (hard assertion):** `transformers>=4.57.3`
  is declared in `[project].dependencies` (`pyproject.toml:12`, a **runtime**
  dep; `uv.lock` currently resolves `4.57.3`), and a `uv lock --upgrade` could
  resolve a new *minor* (4.58+) whose tokenizer-behavior change subtly alters
  `datasets`-backed CIFAR data loading. Before committing the bump, **assert the
  resolution satisfies `transformers >=4.57.3,<4.58.0`** (a patch bump, not a
  minor). If it would advance to `4.58+`, do **not** land it in this window: add
  a `[tool.uv] constraint-dependencies` cap (`transformers<4.58`) and defer the
  minor to a separate reviewed validation window. Rationale: a behavioral (not
  build) failure from a runtime bump could red-gate the post-FF CI re-run
  **after** the functional FF is already on `main` — exactly the hard-to-rollback
  scenario.
- **Per-package high/critical verification (not just pyarrow + urllib3):** add a
  pre-commit version/Dependabot-alert check asserting the patched floor for
  **every** high/critical Python package expected to clear, not only `pyarrow`
  and `urllib3`. The cluster includes `cryptography`, `GitPython`, `pillow`,
  `starlette`, `python-multipart`, `tornado`, `pyjwt`, and `mistune`. Each must
  be asserted at its patched version (or its alert confirmed cleared) before the
  bump commit lands; do not infer clearance from a green test run.
- **`override-dependencies` discipline (core-runtime transitives):** an inline
  `[tool.uv] override-dependencies` floor is **too permissive** for core-runtime
  transitives like `pyarrow` (reached via the core `datasets` dependency,
  `pyproject.toml:11`), because it can bypass upstream dependency metadata and
  produce a resolver solution outside what the upstream package declares
  compatible. **Allow only `[tool.uv] constraint-dependencies` inline, and only
  if the resolver still solves normally.** If `override-dependencies` is truly
  required, **defer the bump to a separate reviewed window** with targeted
  `datasets` / CIFAR / TinyStories smoke tests — do not land an override in this
  trailing commit.
- **npm (Overwatch web):** in `src/esper/karn/overwatch/web`, `npm update` +
  `npm audit fix` (or targeted `vite@^7.3.5 vitest@^4.1.0`). The current 1
  critical and 3 of the highs are JS **dev/build tooling**; `node_modules` are
  not shipped. (Note: the Overwatch web project does have runtime dependencies —
  Vue/ECharts — and `web/dist/**` is package data (`pyproject.toml:45`), so this
  is scoped to "the current high/critical npm alerts are dev/build tooling",
  not "the entire Overwatch npm surface is dev-only.")
- **Why last, why this window:** (a) the manifest/lockfile diff rides cleanly
  alongside the FF; (b) the merge already re-establishes CI posture, so
  re-running the suite after the bumps is "free"; (c) keeping the bump as a single
  trailing commit makes a dependency regression independently revertable; (d) the
  checkpoint-break migration (§2) and the bumps are orthogonal — sequence schema
  first so a flaky transitive cannot block the functional merge.

The full Dependabot triage table lives in the **parent sprint doc**, not here;
this spec only fixes the *sequencing contract* (bumps = last commit, same
window).

## Rejected alternatives

- **`--no-ff` merge commit.** Adds a synthetic merge node with no value (there is
  nothing to merge) and obscures the linear lineage that makes bisection across
  P0-1 / PR1 / PR2 cheap. Rejected.
- **Squash merge.** Discards the 46 curated commit messages this design
  explicitly preserves for bisection; collapses a coherent, order-dependent
  sequence into an unbisectable blob. Rejected.
- **Staged / multi-PR landing.** Manufactures intermediate states that never
  existed and were never tested, breaks bisection, and risks landing a fix ahead
  of the change it fixes — all for zero conflict-avoidance benefit (there are no
  conflicts). Rejected *under the FF topology*; **if the topology changes to a
  genuine three-way merge** (a push lands on `main` before this plan executes),
  this decision must be revisited — staging becomes the correct approach for a
  genuine fork. See §6 Conflict and rollback handling.
- **Writing a checkpoint migration / remap shim for pre-v2 checkpoints.**
  Directly prohibited by the No-Legacy policy and structurally impossible without
  fabricating `state_value_head` weights. Rejected; "retrain" is the contract.
- **Bumping dependencies in a separate pre- or post-merge window.** Creates two
  destabilizing events on `main` and wastes a second full CI pass; the
  manifest/lockfile bump rides the merge window cleanly. Rejected — **except**
  where a core-runtime transitive needs an `override-dependencies` entry or a
  `transformers` minor, in which case that specific bump **is** deferred to a
  separate reviewed window (see §7).
- **Bumping `transformers` to 5.0.0rc3 to clear its alert now.** Pulling a
  pre-release into the runtime to satisfy a *medium* alert is a worse trade than
  leaving the alert open until 5.0 GA. Rejected.

## Locked decisions

1. Land via the **CLI `git checkout main && git merge --ff-only 0.1.1 && git push`** path (SHA-preserving) with **SHA-equality verification**, gated by the §6 hard precondition that `gh pr checks` is green first — **never** squash, **never** `--no-ff`, **never** a GitHub merge button.
2. Single PR, not staged (revisit only if the topology changes to a three-way merge — §6).
3. Checkpoint break = retrain-only; no migration script, no remap shim, ever. Real pre-v2 checkpoints lack the `value_head_schema_version` key entirely and fire the **KeyError path** (read at `ppo_agent.py:1656`, raised at `:1664`), not the schema-version check (`:1675`, which is only for a field-present-but-wrong-numeric value).
4. Run **all five CI lanes locally** before opening the PR (default/unit, integration, property with `HYPOTHESIS_PROFILE=thorough`, slow, stress) with a per-lane collect-only non-empty assertion — a bare `uv run pytest` excludes four of five lanes (`pytest.ini:24`). Reproduce the 75% gate with the CI marker set and **`--cov=src`** (not `--cov=esper`) and run the simic domain-scoped coverage check (`--cov=src/esper/simic`).
5. The EV-consumer audit gate-fix (Sprint item 1) lands **before or with** this merge. Targets are the §4 file:line list (`narrative_panel.py`, `aggregator.py`, `ppo_losses_panel.py`, `critic_calibration_panel.py`, `karn/constants.py`), and `aggregator.py` must propagate a low-variance flag. This is a **hard co-land sequencing constraint** (classify as `depends_on`, not soft, in the plan metadata).
6. Dependency bumps are the **last commit** (a manifest/lockfile commit, not strictly lockfile-only), independently revertable, after the FF is green. Escalate the `pyarrow`/`datasets` (core-runtime) high; assert `transformers >=4.57.3,<4.58.0`; assert the patched floor for **every** expected high/critical package (cryptography, GitPython, pillow, starlette, python-multipart, tornado, pyjwt, mistune — not just pyarrow/urllib3); allow only `constraint-dependencies` inline (defer any `override-dependencies` or transformers-minor to a separate reviewed window).
7. No `git reset --hard` / `git push --force` on `main` without explicit user permission (CLAUDE.md Git Safety).

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| EV false-alarms on K>1 runs once estimator hits main | High if item 1 not landed | Medium (noisy gates/alerts, eroded trust) | Locked decision 5 — gate-fix lands with the merge; §4 enumerates the widget targets + low-variance flag |
| CLI push bypasses PR CI; 75% coverage/lint/typecheck/web never gate the push | Medium if precondition skipped | High (unvalidated code on main) | §6 hard precondition — `gh pr checks` must be green before the CLI push |
| Nightly-only paths (slow/stress) first exercised post-push regress the refactor | Medium | High (main goes red after merge) | Locked decision 4 — full local suite before PR |
| Effective coverage of high-complexity new simic paths drifts below the fixed 75% gate | Medium | Medium (latent untested risk) | §4 domain-scoped `tests/simic/` coverage check against a tracked baseline |
| A pre-v2 checkpoint someone relies on is silently lost | Low | High (lost training state) | §2 item 3 + §2-check greppable confirmation |
| `main` receives a push between spec and merge, breaking the FF assumption | Low | Low (detected immediately) | §6 — `--ff-only` fails loudly; re-verify merge-base; staging contingency in Rejected alternatives |
| `transformers` minor bump changes tokenizer behavior, red-gating post-FF CI | Low | High (main red after FF, hard to roll back) | §7 — verify patch-not-minor resolution; cap `transformers<4.58` if needed |
| `pyarrow` UAF reachable via core-runtime `datasets` in CIFAR installs | Medium | High (runtime security exposure) | §7 — escalated as required-runtime transitive, bumped in the trailing commit |
| Dependency bump introduces a transitive regression | Low | Medium | §7 — bumps are the last, independently revertable commit |

## Acceptance criteria

1. **Topology verified.** `git merge-base main 0.1.1` equals `git rev-parse main`
   and `git merge-base --is-ancestor main 0.1.1` exits 0 immediately before the
   merge command runs.
2. **Fast-forward preserved.** After the merge, `git log --oneline main` shows the
   46 individual commits (P0-1, PR1/PR2, refactor sequence) — no squash blob, no
   synthetic merge commit. **Verification:** `git rev-parse origin/main ==
   git rev-parse 0.1.1` (SHA-identical), the only mechanic that holds under the
   locked CLI fast-forward path.
3. **Checkpoint break documented and confirmed.** The PR body calls out
   `VALUE_HEAD_SCHEMA_VERSION = 2` (anchor `leyline/__init__.py:125`), states
   "retrain, no remap", notes that real pre-v2 checkpoints lack the
   `value_head_schema_version` key entirely and fire the KeyError path at
   `ppo_agent.py:1664` (NOT the schema check at `:1675`, which only fires for a
   field-present older numeric value), and the §2-check grep confirms no
   checkpoint-loading test fixture assumes a pre-v2 artifact.
4. **All five CI lanes green pre-PR.** Each lane (default/unit, integration,
   property with `HYPOTHESIS_PROFILE=thorough`, slow, stress) runs explicitly and
   passes before the PR is opened, each preceded by a `--collect-only` non-empty
   assertion. The CI-mirroring coverage pass (`-m 'not integration and not stress
   and not property and not slow' --cov=src`) reports ≥ 75%, and the simic
   domain-scoped coverage check (`--cov=src/esper/simic`) holds against its
   baseline.
5. **PR CI green.** All gating jobs (lint, typecheck, property, unit+integration
   with the 75% coverage gate, overwatch-web) pass on the PR. `gh pr checks` is
   confirmed all-green **before** any CLI push to `main`.
6. **EV gate-fix co-landed.** Sprint item 1's EV-consumer gate-fix is merged
   before or in the same window; the §4 widget targets are guarded and
   `aggregator.py` propagates a low-variance flag; no EV-keyed health gate,
   anomaly rule, or Sanctum panel status false-alarms on a K>1 smoke run on
   post-merge `main`.
7. **Dependency bumps land last and clean.** `uv lock --upgrade && uv sync` and
   the npm `audit fix` land as the final manifest/lockfile commit(s); CI re-runs
   green; the bump commit is a single revertable unit; `transformers` is **not**
   on a pre-release and resolves to `>=4.57.3,<4.58.0`; the patched floor of
   every expected high/critical package is asserted (pyarrow, urllib3,
   cryptography, GitPython, pillow, starlette, python-multipart, tornado, pyjwt,
   mistune); the `pyarrow`/`datasets` core-runtime high is addressed via
   `constraint-dependencies` only (any `override-dependencies` need defers the
   bump); `torch` #80 is annotated as no-fix.
8. **Rollback path stated.** The PR body names the forward-`revert` rollback path
   and confirms no `--hard`/`--force` is used on `main`.

## Files touched (by this integration, not by the spec)

- `src/esper/leyline/__init__.py` (`:125`, `:877`) — `VALUE_HEAD_SCHEMA_VERSION = 2` (lands via FF).
- `src/esper/simic/agent/ppo_agent.py` (`:51`, `:1540`, `:1656`, `:1663-1668` KeyError path, `:1675-1683` schema check) — checkpoint save/load enforcement (lands via FF).
- EV-consumer audit gate-fix targets (Sprint item 1, co-land): `src/esper/karn/sanctum/widgets/tamiyo/narrative_panel.py`, `ppo_losses_panel.py`, `critic_calibration_panel.py`, `src/esper/karn/sanctum/aggregator.py` (~`:980`), `src/esper/karn/constants.py` (`:120-121`).
- `uv.lock` — dependency bump pass (new trailing commit).
- `src/esper/karn/overwatch/web/package-lock.json` — npm dev-tooling bumps (new trailing commit).
- `docs/coord/PLAN_TRACKER.md` — updated by the orchestrator after merge (out of scope for this spec).

No code is authored *by this spec*; the merge transports already-committed work.

## Empirical context

- Branch lead verified at spec time: 46 commits, 206 files, +21869/−3078, FF topology (merge-base = main HEAD = `f7f1aece`).
- Suite size verified at spec time: 385 test files (`find tests -name 'test_*.py' | wc -l`).
- P0-1 A/B (motivating the EV co-landing constraint): op-head starvation fixed (head_op_grad_norm late-alive fraction K=4 control 0.05 → P0-1 0.89); K=4−K1 EV gap widened to +0.201 (control +0.137); best median EV +0.239 and lowest value_loss 0.099 of all arms — but raw EV std blew out (K=4 EV mean −0.337, std 2.264 vs control 0.572), a denominator artifact of the op-marginal V(s) under low return-variance batches. That artifact is exactly what Sprint item 1's gate-fix addresses (the §4 widget targets + low-variance flag) and why criterion 6 gates this merge.
- Dependabot at spec time: 80 open alerts (1 critical / 34 high / 32 moderate / 13 low); 71 Python (mostly lockfile-delta resolvable; 4 direct) + 9 npm (incl. the lone critical) whose current high/critical alerts are dev/build tooling (`node_modules` not shipped — though Overwatch web does carry runtime Vue/ECharts and `web/dist/**` is package data at `pyproject.toml:45`). `pyarrow` (high, UAF, GHSA-rgxp-2hwp-jwgg) is a required-runtime transitive via the core `datasets>=4.4.1` dependency.
```
