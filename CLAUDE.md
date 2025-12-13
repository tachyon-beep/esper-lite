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

## hasattr Usage Policy

**STRICT REQUIREMENT:** The use of `hasattr()` must be explicitly authorized by the operator for every case without exception.

### Authorization Requirements

Every `hasattr()` call in the codebase MUST be accompanied by an inline comment containing:

1. Explicit authorization from the operator
2. Date and time of authorization (ISO 8601 format)
3. Justification for why hasattr is necessary

### Format

```python
# hasattr AUTHORIZED by [operator name] on [YYYY-MM-DD HH:MM:SS UTC]
# Justification: [Specific reason why hasattr is required]
if hasattr(obj, 'attribute'):
    ...
```

### Example (Legitimate Use)

```python
# hasattr AUTHORIZED by John on 2025-11-30 14:23:00 UTC
# Justification: Serialization - handle both enum and string event_type values from external sources
event_type = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
```

### Rationale

The `hasattr()` function is often used to mask integration flaws:

- Checking for attributes that should always exist (poor type contracts)
- Checking for attributes that never exist (actual bugs)
- Duck typing that should be formalized with Protocols or ABCs

**Default stance:** If you need `hasattr()`, you probably have a design problem. Fix the design instead.

### Enforcement

- Any `hasattr()` without proper authorization MUST be flagged during code review
- Claude Code MUST NOT introduce new `hasattr()` calls without explicit operator approval
- Existing unauthorized `hasattr()` calls should be treated as technical debt to be removed

### Exceptions

The only legitimate uses of `hasattr()` are:

1. **Serialization/Deserialization:** Handling polymorphic data from external sources
2. **Cleanup Guards:** Defensive programming in `__del__`/`close()` methods
3. **Feature Detection:** When integrating with external libraries where feature availability varies

Even these cases require authorization and documentation.

## Archive Policy

**The `docs/plans/archive/` directory contains completed or superseded implementation plans.**

### What It Contains

- Implementation plans that have been executed
- Plans superseded by architectural changes
- Historical decision records

### How to Treat It

- **Reference for understanding past decisions** and why certain approaches were taken
- **DO NOT** implement tasks from archived plans without checking if they're still relevant
- **DO NOT** assume archived plans reflect current architecture

### The Rule

When a plan is completed or superseded, move it to the archive. Keep `docs/plans/` lean with only active/future work.

## Deferred Functionality

Whenever we make an active design decision to defer functionality due to complexity, add a TODO comment in the most logical place:

```python
# TODO: [FUTURE FUNCTIONALITY] - Brief description of what was deferred and why
```
- remember that we use UV in this project and use it as the preferred way to execute pythonc ode