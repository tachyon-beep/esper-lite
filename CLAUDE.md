# Claude Code Guidelines

This file contains mandatory rules for Claude Code when working on this codebase.

## First Steps

**On session start, read these files for project context:**

1. **`README.md`** - Project overview, architecture, CLI reference with all flags
2. **`ROADMAP.md`** - Architecture principles (the "Nine Commandments"), execution phases, and biological component roles

**If more detailed context is needed at any time, read:**
3. **`docs/arch-analysis-*/*.md`** - Select the most recent in-depth architecture analysis and design rationale. Includes architectural diagrams and component descriptions.

These documents MUST be kept current and provide essential context for understanding the codebase.

## No Legacy Code Policy

**STRICT REQUIREMENT:** Legacy code, backwards compatibility, and compatibility shims are strictly forbidden.

### Anti-Patterns - Never Do This

The following are **strictly prohibited** under all circumstances:

1. **Backwards Compatibility Code**
   - No version checks (e.g., `if version < 2.0: old_code() else: new_code()`)
   - No feature flags for old behavior
   - No "compatibility mode" switches

2. **Legacy Shims and Adapters**
   - No adapter classes to support old interfaces
   - No wrapper functions that translate old APIs to new ones
   - No proxy objects for deprecated functionality

3. **Deprecated Code Retention**
   - No `@deprecated` decorators with code kept around
   - No commented-out old implementations "for reference"
   - No `_legacy` or `_old` suffixed functions

4. **Migration Helpers**
   - No code that supports "both old and new" simultaneously
   - No gradual migration paths in the codebase
   - No transition periods with dual implementations

### The Rule

**When something is removed or changed, DELETE THE OLD CODE COMPLETELY.**

- Don't rename unused variables to `_var` - delete the variable
- Don't keep old code in comments - delete it (git history exists)
- Don't add compatibility layers - change all call sites
- Don't create abstractions to hide breaking changes - make the breaking change

### Rationale

Legacy code and backwards compatibility create:

- **Complexity:** Multiple code paths doing the same thing
- **Confusion:** Unclear which version is "correct"
- **Technical Debt:** Old code that never gets removed
- **Testing Burden:** Must test all combinations
- **Maintenance Cost:** Changes must update both paths

**Default stance:** If old code needs to be removed, delete it completely. If call sites need updating, update them all in the same commit.

### Enforcement

- Claude Code MUST NOT introduce backwards compatibility code
- Claude Code MUST NOT create legacy shims or adapters
- Claude Code MUST delete old code completely when making changes
- Any legacy code patterns MUST be flagged and removed immediately

## Git Safety

**STRICT REQUIREMENT:** Never run destructive git commands without explicit user permission.

### Destructive Commands (REQUIRE PERMISSION)

The following commands can destroy uncommitted work or rewrite history. **ALWAYS ask before running:**

- `git reset --hard` - Discards uncommitted changes
- `git clean -f` - Deletes untracked files permanently
- `git checkout -- <file>` - Discards uncommitted changes to file
- `git stash drop` - Permanently deletes stashed changes
- `git push --force` - Rewrites remote history
- `git rebase` (on pushed branches) - Rewrites shared history

### When You Think You Need a Destructive Command

**Don't.** Go back and get clarification from the user.

### The Rule

**A messy commit or wrong files in a commit is a minor, fixable problem. Lost uncommitted work is permanent.**

If you make a mistake (wrong files staged, bad commit message, etc.):

1. Make another commit to fix it
2. Or ask the user if they want to do an interactive rebase
3. NEVER unilaterally run destructive commands to "clean up"

### Enforcement

- Claude Code MUST ask permission before any destructive git command
- Claude Code MUST prefer safe alternatives (stash, backup branch)
- Claude Code MUST NOT prioritize "clean history" over "don't lose work"

## Plans Organization

Plans are grouped into:

- `docs/plans/concepts/` — draft specs, strategies, and early ideas (not implementation-ready)
- `docs/plans/ready/` — approved or actively staged implementation plans
- `docs/plans/completed/` — executed plans and historical decision records
- `docs/plans/abandoned/` — superseded, deferred, or cancelled plans

### How to Treat Completed/Abandoned

- **Reference for understanding past decisions** and why certain approaches were taken
- **DO NOT** implement tasks from completed/abandoned plans without checking if they're still relevant
- **DO NOT** assume completed/abandoned plans reflect current architecture

### The Rule

When a plan is completed, move it to `docs/plans/completed/`. When a plan is superseded, deferred, or cancelled, move it to `docs/plans/abandoned/`. Keep `docs/plans/ready/` lean with only active/future work.

### Plan Tracking

**When creating, modifying, or obsoleting any plan, update `docs/coord/PLAN_TRACKER.md`.** This is the master coordination document for prioritization and dependency tracking. Use `docs/plans/PLAN_TEMPLATE.md` for metadata schemas.

### Plan Review Requirements

**All plans involving code changes MUST be reviewed by relevant specialist agents before approval.** Use the appropriate experts based on the plan's domain:

| Domain | Required Reviewer |
|--------|-------------------|
| RL training, rewards, policies | `drl-expert` agent + `yzmir-deep-rl` skills |
| PyTorch, tensors, torch.compile | `pytorch-expert` agent + `yzmir-pytorch-engineering` skills |
| Python patterns, architecture | `axiom-python-engineering` skills |
| Neural architectures | `yzmir-neural-architectures` skills |
| Training stability, optimization | `yzmir-training-optimization` skills |

For cross-domain plans (most non-trivial work), invoke **multiple specialists**. Document their sign-off in the plan's metadata under `reviewed_by`.

## Deferred Functionality

Whenever we make an active design decision to defer functionality due to complexity, add a TODO comment in the most logical place:

```python
# TODO: [FUTURE FUNCTIONALITY] - Brief description of what was deferred and why
```

## Development Conventions

### Specialist Subagents and Skill Packs

**Use specialist subagents and skill packs liberally and at your own discretion.**

This project is a deep reinforcement learning system built on PyTorch. Given the domain complexity, Claude Code should proactively leverage specialists when working on relevant areas.

#### Subagents (spawn for complex tasks)

| Agent | Use For |
|-------|---------|
| **pytorch-expert** | torch.compile optimization, FSDP/distributed training, TorchInductor, custom kernels, performance profiling, memory analysis, tensor operations, GPU debugging |
| **drl-expert** | PPO/SAC/TD3 implementation, reward engineering, training stability, policy/value network architecture, exploration strategies, hyperparameter tuning, RL debugging |

#### Skill Packs (invoke via Skill tool for guidance)

| Skill Pack | Use For |
|------------|---------|
| **yzmir-training-optimization** | NaN losses, gradient explosion, learning rate scheduling, optimizer selection, overfitting prevention, experiment tracking, convergence issues |
| **yzmir-neural-architectures** | Policy/value network design, CNN vs Transformer decisions, attention mechanisms, normalization techniques, architecture stability |
| **yzmir-pytorch-engineering** | PyTorch implementation patterns, memory profiling, distributed training, tensor operations, CUDA debugging |
| **yzmir-deep-rl** | Algorithm selection (PPO/SAC/TD3), reward shaping, exploration-exploitation, on-policy vs off-policy, sample efficiency |
| **ordis-quality-engineering** | E2E testing, flaky test remediation, test automation, property-based testing, chaos engineering |
| **axiom-python-engineering** | Python patterns, type systems, async code, package structure, code review |
| **elspeth-ux-specialist** | TUI design for Karn/Overwatch, dashboard layouts, keyboard navigation, status indicators |

**Default stance:** When in doubt, spawn the specialist or invoke the skill. The overhead is trivial compared to the cost of subtle bugs in tensor operations, RL training loops, or test flakiness.

**Examples of when to use:**

- Implementing or modifying anything in `simic/` (RL training) → `drl-expert` agent + `yzmir-deep-rl` skills
- Debugging NaN losses, exploding gradients, or training instability → `yzmir-training-optimization` skills
- Optimizing tensor operations or memory usage → `pytorch-expert` agent
- Writing custom loss functions or network architectures → `drl-expert` + `yzmir-neural-architectures`
- Reviewing RL or PyTorch code → spawn the relevant expert as a reviewer + `axiom-python-engineering`
- Working on Karn TUI or Overwatch dashboards → `elspeth-ux-specialist`
- Fixing flaky tests or improving test coverage → `ordis-quality-engineering`

### Karn MCP Server

The `esper-karn` MCP server provides SQL access (DuckDB) to training telemetry. Start with `mcp__esper-karn__list_runs` or `mcp__esper-karn__run_overview`, then use `mcp__esper-karn__query_sql` for custom queries (structured JSON). Use `mcp__esper-karn__describe_view` to inspect schemas, and `mcp__esper-karn__query_sql_markdown` for copy/paste tables.

Core views include `runs`, `epochs`, `ppo_updates`, `batch_epochs`, `batch_stats`, `seed_lifecycle`, `decisions`, `rewards`, `trends`, `anomalies`, `episode_outcomes`, and `raw_events`. Prefer filtering by `run_dir` to avoid mixing multiple runs in a single telemetry directory.

### Package Manager: UV

This project uses **UV** as the preferred package manager and Python executor.

```bash
# Install dependencies
uv sync

# Run Python code
uv run python -m esper.scripts.train ppo --episodes 100

# Run tests
uv run pytest
```

### Playwright Tests (Overwatch Web)

The Overwatch web dashboard has Playwright tests located in `src/esper/karn/overwatch/web/`.

```bash
# Install npm dependencies (from the web directory)
cd src/esper/karn/overwatch/web
npm install

# Install Playwright browsers (run once after npm install)
npx playwright install

# Run Playwright tests
npx playwright test
```

### Leyline: The Shared Contracts Module

**All new constants, enums, and shared types MUST be placed in `leyline`.**

The `leyline` module is the project's "DNA" — it defines the contracts that all other domains depend on:

- **Enums:** `SeedStage`, `ActionType`, `SlotPosition`, etc.
- **Constants:** Thresholds, defaults, magic numbers with semantic names
- **Type definitions:** Shared dataclasses, TypedDicts, Protocols
- **Tensor schemas:** Observation/action space definitions

**Rationale:** Leyline prevents circular imports and ensures all domains share a single source of truth for fundamental types. If you're defining something that multiple domains will import, it belongs in leyline.

### Documentation Metaphors

**IMPORTANT:** Esper uses two complementary metaphors. Do NOT mix them or invent alternatives.

#### 1. Body/Organism Metaphor — For System Architecture

The seven domains are organs/systems within a single organism. e.g.:

| Domain | Biological Role | Description |
|--------|-----------------|-------------|
| **Kasmina** | Stem Cells | Pluripotent slots that differentiate into modules |
| **Tamiyo** | Brain/Cortex | Strategic decision-making, high-level control |
| **Tolaria** | Metabolism | Energy conversion, training execution |
| **Simic** | Evolution | Adaptation through reinforcement learning |

#### 2. Plant/Botanical Metaphor — For Seed Lifecycle Only

The lifecycle of individual neural modules uses botanical terms:

| Stage | Meaning |
|-------|---------|
| **Dormant** | Inactive, waiting to be activated |
| **Germinated** | Module created and beginning to develop |
| **Training** | Growing, learning from host errors |
| **Blending** | Being integrated into the host |
| **Grafted/Fossilized** | Permanently fused with host |
| **Pruned** | Removed due to poor performance |

#### The Framing

Think of it as: **"The organism's stem cells undergo a botanical development process."** The body metaphor describes *what the system is*; the plant metaphor describes *how individual modules mature*.

#### Documentation Quality

- Claude Code MUST NOT introduce new metaphors (no "factory", "pipeline", "machine" language)
- Claude Code MUST NOT use body terms for seed states (no "seed is metabolizing")
- Claude Code MUST NOT use plant terms for system architecture (no "Tamiyo is the gardener")
- When in doubt, refer to the tables above
- If you are being asked to deliver a telemetry component, do not defer or put it off; if you are working on a half finished telemetry component, do not remove it as 'incomplete' or 'deferred functionality'. This pattern of behaviour is why we are several months in and have no telemetry.

## PROHIBITION ON "DEFENSIVE PROGRAMMING" PATTERNS

No Bug-Hiding Patterns: This codebase prohibits defensive patterns that mask bugs instead of fixing them. Do not use .get(), getattr(), hasattr(), isinstance(), or silent exception handling to suppress errors from nonexistent attributes, malformed data, or incorrect types. A common anti-pattern is when an LLM hallucinates a variable or field name, the code fails, and the "fix" is wrapping it in getattr(obj, "hallucinated_field", None) to silence the error—this hides the real bug. When code fails, fix the actual cause: correct the field name, migrate the data source to emit proper types, or fix the broken integration. Typed dataclasses with discriminator fields serve as contracts; access fields directly (obj.field) not defensively (obj.get("field")). If code would fail without a defensive pattern, that failure is a bug to fix, not a symptom to suppress.

### Legitimate Uses

This prohibition does not extend to genuine uses of type checking or error handling where appropriate, such as:

- **PyTorch tensor operations** (9): Converting tensors to scalars, device moves
- **Device type normalization** (6): `str` → `torch.device` conversion
- **Typed payload discrimination** (40): Distinguishing between typed dataclass payloads (correct migration pattern)
- **Enum/bool/int validation** (10): Rejecting bool as int in coercion
- **Numeric field type guards** (20): Conditional rendering of optional numeric fields
- **NN module initialization** (2): Layer type detection for weight init
- **Serialization polymorphism** (3): Enum, datetime, Path handling

For absence of doubt, when using these ask yourself "is this defensive programming to hide a bug that should not be possible in a well designed system, or is this legitimate type handling?' If the former, remove it and fix the underlying issue.

<!-- filigree:instructions:v3.0.0:65e6fb25 -->
<!-- filigree:last-writer:filigree install -->
## Filigree Issue Tracker

`filigree` tracks tasks for this project. Data lives in `.filigree/`. Prefer
the MCP tools (`mcp__filigree__*`) when available; fall back to the `filigree`
CLI otherwise.

### Workflow

```bash
# At session start
filigree session-context                            # ready / in-progress / critical path

# Pick up the next startable issue (atomic claim + transition into its working status)
filigree start-next-work --assignee <name>
# ...or claim a specific issue
filigree start-work <id> --assignee <name>

# Do the work, commit, then
filigree close <id>
```

Use the atomic claim+transition verbs — `work_start` / `work_start_next`
(MCP) or `start-work` / `start-next-work` (CLI). Do **not** chain
`work_claim` (MCP) or `filigree claim` (CLI) with a subsequent status
update — the two-step form races against other agents; the combined verb is
atomic.

**Ready ≠ startable.** The working status is type-specific (tasks →
`in_progress`, features → `building`). Bugs start at `triage`, which has no
single-hop transition into work (`triage → confirmed → fixing`), so a triage
bug is *ready* but not directly *startable*: `work_start` on one returns
`INVALID_TRANSITION` naming the next status, and `work_start_next` skips it.
`work_ready` items carry a `startable` flag (plus a `next_action` hint when
false). Pass `advance=true` (MCP) / `--advance` (CLI) to walk the soft
transitions to the nearest working status automatically.

### Observations: when (and when not) to use them

`observation_create` is a fire-and-forget scratchpad for *incidental* defects — things
you notice *outside the scope of your current task* (a code smell in a
neighbouring file, a stale TODO, a missing test for an edge case you happened
to spot). Notes expire after 14 days unless promoted. Include `file_path` and
`line` when relevant. At session end, skim `observation_list` and either
`observation_dismiss` or `observation_promote` for what has accumulated.

**You fix bugs in your currently defined scope. You do NOT use observations
to finish work prematurely.** If a defect, gap, or follow-up belongs to your
current task, you own it — handle it as part of that task: fix it now, expand
the task's scope, file a proper issue with a dependency, or surface it to the
user. Filing it as an observation and closing the task is *not* completing
the task; it is shipping known-broken work and hiding the debt in a 14-day
expiring scratchpad. The test is "would I have noticed this even if I weren't
working on this task?" If no, it's task scope, not an observation.

### Priority scale

- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog

### Reaching for tools

MCP tool schemas describe each tool; `filigree --help` and `filigree <verb>
--help` are the authoritative CLI reference. You do not need to memorise
either catalogue. The verbs you will reach for most:

- **Find work:** `work_ready`, `work_blocked`, `issue_list`, `issue_search`
- **Claim work:** `work_start`, `work_start_next`
- **Update:** `comment_add`, `label_add`, `issue_update`, `issue_close`
- **Admin (irreversible):** `issue_delete` (MCP) / `delete-issue` (CLI) —
  hard-deletes a terminal issue and its rows; `admin_undo_last` cannot reverse it.
- **Scratchpad:** `observation_create`, `observation_list`, `observation_promote`, `observation_dismiss`
- **Cross-product entity bindings (ADR-029):** `entity_association_add`,
  `entity_association_remove`, `entity_association_list`,
  `entity_association_list_by_entity`. Used when a sibling tool (e.g.
  Loomweave) needs to bind a Filigree issue to a function, class, or
  module identifier it owns. The `entity_id` is an opaque external string
  from Filigree's perspective and may be a `loomweave:eid:...` SEI or a legacy
  locator; callers may also supply `entity_kind` explicitly. The consumer (the sibling tool's read
  path) does drift detection against the stored
  `content_hash_at_attach`. `entity_association_list_by_entity` is the
  reverse-lookup surface — given an opaque external entity ID, return every
  Filigree issue bound to it (project isolation is by DB file). Also
  reachable over HTTP as
  `GET/POST /api/issue/{issue_id}/entity-associations`,
  `DELETE /api/issue/{issue_id}/entity-associations?entity_id=…`,
  and `GET /api/entity-associations?entity_id=…`.
- **Health:** `stats_get`, `metrics_get`, `mcp_status_get`

Pass `--actor <name>` (CLI) so events attribute to your agent identity. It
works in either position — before the verb (`filigree --actor X update …`) or
after it (`filigree update … --actor X`); the post-verb value overrides the
group-level one.

### Error handling

Errors return `{error: str, code: ErrorCode, details?: dict}`. Switch on
`code`, not on message text. Codes: `VALIDATION`, `NOT_FOUND`, `CONFLICT`,
`INVALID_TRANSITION`, `PERMISSION`, `NOT_INITIALIZED`, `IO`,
`INVALID_API_URL`, `FILE_REGISTRY_DISPLACED`, `REGISTRY_UNAVAILABLE`,
`LOOMWEAVE_REGISTRY_VERSION_MISMATCH`, `LOOMWEAVE_OUT_OF_SYNC`,
`BRIEFING_BLOCKED`, `STOP_FAILED`, `SCHEMA_MISMATCH`, `INTERNAL`.

On `INVALID_TRANSITION`, call `workflow_transition_list` (MCP) or
`filigree transitions <id>` to see what the workflow allows from here.

Two failure modes deserve a specific response:

- **`SCHEMA_MISMATCH`** — the installed `filigree` is older than the project
  database. The error message contains upgrade guidance. Surface it to the
  user; do not retry.
- **`ForeignDatabaseError`** — filigree found a parent project's database
  but no local `.filigree.conf`. Run `filigree init` in the current
  directory. Do **not** `cd` upward to a different project unless that was
  the actual intent.
<!-- /filigree:instructions -->

<!-- loomweave:instructions:v1.1.0-rc10:ca999d34 -->
<!-- loomweave:last-writer:loomweave install -->
## Loomweave (code archaeology)

This repo is indexed by Loomweave: it has pre-extracted the tree into a
queryable map of entities (functions, classes, modules, files), the call /
reference / import edges plus relation edges (inherits_from / decorates /
implements / derives), and subsystem clusters. Before grepping the tree to
answer "what calls X", "what subclasses X", "where is X defined", "what
subsystem owns X", or "find the thing that does Y" — ask Loomweave's MCP tools
(`mcp__loomweave__*`): `entity_find`, `entity_at`, `entity_callers_list`,
`entity_relation_list`, `entity_neighborhood_get`, `project_status_get`.

`entity_find` is the grep replacement for "find the thing that does Y": it
matches a concept word by substring over name, summary, and docstring content
(e.g. `library` finds `LibraryService`), with no embeddings required — reach for
it before grepping. Semantic *ranking* is the separate, opt-in
`entity_semantic_search_list`.

Entity IDs are `{plugin}:{kind}:{qualified_name}`; subsystems are
`core:subsystem:{hash}`. Never hand-construct one: get it from `entity_find` /
`entity_at`, or — for a pasted qualname, Rust `::` path, or SEI token — from
`entity_resolve`, then copy it verbatim into the next tool.

Index freshness and counts: `project_status_get` (or the `loomweave://context`
resource). If the index is stale, run `loomweave analyze <path>`.

LLM summaries (`entity_summary_get`) are off by default and need a live
provider; `project_status_get` reports the posture, `loomweave config check`
explains enabling.

Full workflow: the `loomweave-workflow` skill.
<!-- /loomweave:instructions -->

<!-- wardline:instructions:v1:bcd19330 -->
<!-- wardline:last-writer:wardline install -->
This project uses **wardline** as its trust-boundary gate. Before handing back code that touches external input, run `wardline scan . --fail-on ERROR` (exit 0 = clean, 1 = gate tripped, 2 = wardline error) and fix findings at the boundary, not the sink. The full scan -> explain -> fix -> rescan loop and the baseline-vs-waiver discipline live in the `wardline-gate` skill and in `docs/agents.md`.
<!-- /wardline:instructions -->

<!-- legis:instructions:v1.0.0:6604fe0c -->
## Legis (git/CI + governance)

Legis is the git/CI and governance layer of the Weft suite. Reach for it when a policy fires at the CI/git boundary and a change needs a *recordable* override or human sign-off, when you need governance attestations keyed to stable code identity (SEI), or when you need git/CI context — branches, commits, pull requests, check outcomes, and the Loomweave-bound rename feed — around the work. Enforcement is graded: agent-programmable policy cells decide whether a violation self-clears with an audit trail, is judged inline, or escalates to a human; every decision lands in an append-only, SEI-keyed audit trail that survives rename/move.

Prefer the `mcp__legis__*` MCP tools when available; fall back to the `legis` CLI.

CLI subcommands:

- `serve` — run the Legis API server.
- `mcp` — run the Legis MCP stdio server (launch-bound `--agent-id`).
- `check-override-rate` — exit 1 if the override-rate gate is FAIL (for CI).
- `governance-gate` — run governance CI gates (currently the override-rate gate).
- `sei-backfill` — resolve legacy locator-keyed governance records through Loomweave batch resolve.
- `policy-boundary-check` — fail when `@policy_boundary` metadata lacks current behavioural evidence.

Full command + MCP-tool reference: see the `legis-workflow` skill.
<!-- /legis:instructions -->

<!-- warpline:instructions:v1.0.0 -->
## Warpline (temporal change-impact)

`warpline` is the Weft federation's temporal / change-impact authority — "if I
touch X, what breaks, and what must I re-verify?". Prefer the MCP tools
(`mcp__warpline__*`); fall back to the `warpline` CLI. Endorsed names and short
shims return identical schema+data.

- `warpline_change_list` / `changed` — changed entities for a rev range; call first.
- `warpline_impact_radius_get` / `blast_radius` — downstream affected set.
- `warpline_reverify_worklist_get` / `reverify` — worklist to recheck before done.
- `warpline_entity_timeline_get` / `timeline`, `warpline_entity_churn_count_get` /
  `churn`, `warpline_edge_snapshot_capture` / `capture_snapshot` (only mutating
  tool; writes `.weft/warpline/` only).

Enrich-only and local-only: every response is `meta.local_only: true`,
`peer_side_effects: []`. `enrichment` is a CLOSED vocab
(`present|absent|unavailable`); sibling absence is explicit, never an implied
clean/allowed state. warpline facts are advisory and never gate. See the
`warpline-workflow` skill for the full loop.
<!-- /warpline:instructions -->
