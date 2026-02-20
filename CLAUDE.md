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

<!-- filigree:instructions -->
## Filigree Issue Tracker

Use `filigree` for all task tracking in this project. Data lives in `.filigree/`.

### Quick Reference

```bash
# Finding work
filigree ready                              # Show issues ready to work (no blockers)
filigree list --status=open                 # All open issues
filigree list --status=in_progress          # Active work
filigree show <id>                          # Detailed issue view

# Creating & updating
filigree create "Title" --type=task --priority=2          # New issue
filigree update <id> --status=in_progress                # Claim work
filigree close <id>                                      # Mark complete
filigree close <id> --reason="explanation"               # Close with reason

# Dependencies
filigree add-dep <issue> <depends-on>       # Add dependency
filigree remove-dep <issue> <depends-on>    # Remove dependency
filigree blocked                            # Show blocked issues

# Comments & labels
filigree add-comment <id> "text"            # Add comment
filigree get-comments <id>                  # List comments
filigree add-label <id> <label>             # Add label
filigree remove-label <id> <label>          # Remove label

# Workflow templates
filigree types                              # List registered types with state flows
filigree type-info <type>                   # Full workflow definition for a type
filigree transitions <id>                   # Valid next states for an issue
filigree packs                              # List enabled workflow packs
filigree validate <id>                      # Validate issue against template
filigree guide <pack>                       # Display workflow guide for a pack

# Atomic claiming
filigree claim <id> --assignee <name>            # Claim issue (optimistic lock)
filigree claim-next --assignee <name>            # Claim highest-priority ready issue

# Batch operations
filigree batch-update <ids...> --priority=0      # Update multiple issues
filigree batch-close <ids...>                    # Close multiple with error reporting

# Planning
filigree create-plan --file plan.json            # Create milestone/phase/step hierarchy

# Event history
filigree changes --since 2026-01-01T00:00:00    # Events since timestamp
filigree events <id>                             # Event history for issue
filigree explain-state <type> <state>            # Explain a workflow state

# All commands support --json and --actor flags
filigree --actor bot-1 create "Title"            # Specify actor identity
filigree list --json                             # Machine-readable output

# Project health
filigree stats                              # Project statistics
filigree search "query"                     # Search issues
filigree doctor                             # Health check
```

### Workflow
1. `filigree ready` to find available work
2. `filigree show <id>` to review details
3. `filigree transitions <id>` to see valid state changes
4. `filigree update <id> --status=in_progress` to claim it
5. Do the work, commit code
6. `filigree close <id>` when done

### Priority Scale
- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog
<!-- /filigree:instructions -->
