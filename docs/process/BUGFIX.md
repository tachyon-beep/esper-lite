# Bug Ticket Remediation Process

This document codifies the process for **fixing** bug tickets that have been triaged.

**Prerequisite:** Tickets should be triaged first using the [TRIAGE.md](./TRIAGE.md) process. This document applies to tickets in `docs/bugs/triaged/`.

---

## Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Phase 1:   │───▶│  Phase 2:    │───▶│  Phase 3:   │───▶│  Phase 4:    │───▶│   Commit    │
│  Root Cause │    │   Pattern    │    │  Hypothesis │    │  Resolution  │    │  & Close    │
│Investigation│    │   Analysis   │    │   Testing   │    │              │    │             │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
```

**Core Principle:** ALWAYS verify claims before attempting fixes. Many bug tickets contain:
- Incorrect line numbers (code shifts over time)
- Exaggerated impact claims
- Fabricated evidence (hallucinated TODOs, comments that don't exist)
- Misunderstanding of intentional design

---

## Phase 1: Root Cause Investigation

**Goal:** Verify the bug actually exists as described.

### 1.1 Read the Actual Code

```bash
# DON'T trust line numbers in tickets - they shift!
# ALWAYS search for the actual code:
grep -n "function_name\|pattern_mentioned" src/esper/path/to/file.py
```

| Check | Action |
|-------|--------|
| **Line numbers** | Search for the code - don't assume ticket's line numbers are current |
| **TODO comments** | Verify claimed TODOs exist: `grep -n "TODO.*keyword" file.py` |
| **Function existence** | Confirm the function/class exists and has the claimed behavior |

### 1.2 Check Git History

```bash
# Did the bug ever exist?
git log --all -p -S "claimed_pattern" -- path/to/file.py

# Was it already fixed?
git log --oneline --grep="TICKET_ID"

# When was this code last changed?
git log --oneline -5 -- path/to/file.py
```

### 1.3 Verify Impact Claims

Most bug tickets exaggerate impact. Trace the actual data flow:

```bash
# Is this value actually USED anywhere?
grep -rn "variable_name" src/esper --include="*.py" | grep -v "def \|#"

# Does it affect training/rewards/decisions?
grep -rn "function_name" src/esper/simic/rewards src/esper/simic/training
```

**Red flags for exaggerated impact:**
- "Could cause X" without evidence X has ever happened
- Impact claims for values that are SET but never READ
- Theoretical edge cases that are structurally impossible

---

## Phase 2: Pattern Analysis

**Goal:** Understand the full context before proposing fixes.

### 2.1 Find Related Code

```bash
# Find similar patterns
grep -rn "similar_function\|similar_pattern" src/esper

# Check if there are multiple reward/bonus components
grep -rn "bonus\|reward\|penalty" src/esper/simic/rewards/rewards.py
```

### 2.2 Check Design Documentation

- Read comments near the code - they often explain WHY
- Check if the pattern is explicitly documented as intentional
- Look for "STRICT MODE", "ASYMMETRIC", "INTENTIONAL" comments

### 2.3 Understand the Full System

Many "bugs" are misunderstandings of composite systems:

| Pattern | Question to Ask |
|---------|-----------------|
| "Flat bonus" | Is there ANOTHER bonus that scales? |
| "Missing validation" | Is validation done at a different layer? |
| "Unused field" | Is it used in telemetry/UI only? |
| "Edge case possible" | Is it structurally prevented? |

---

## Phase 3: Hypothesis Testing

**Goal:** Form and test a hypothesis about whether this is a real bug.

### 3.1 Create Evidence Table

| Claim | Status | Evidence |
|-------|--------|----------|
| "X happens at line Y" | ✅ TRUE / ❌ FALSE | Actual finding |
| "This causes Z" | ✅ TRUE / ❌ FALSE | Trace of actual impact |

### 3.2 Test Minimally

If the bug appears real:
1. Make the SMALLEST possible change
2. Run relevant tests
3. Verify the fix addresses root cause

If the bug appears NOT real:
1. Document WHY it's not a bug
2. Show the evidence trail
3. Explain the intentional design

---

## Phase 4: Resolution

### Resolution Types

| Type | Folder | When to Use |
|------|--------|-------------|
| **FIXED** | `docs/bugs/fixed/` | Bug was real, code was changed |
| **NOT-A-BUG** | `docs/bugs/not-a-bug/` | Claims were incorrect, exaggerated, or misunderstood |
| **WONTFIX** | `docs/bugs/wontfix/` | Bug is real but intentional design or marginal value |
| **ALREADY FIXED** | `docs/bugs/fixed/` | Bug was fixed by a previous commit |

### Resolution Section Template

Add this to the ticket before moving:

```markdown
## Resolution

### Status: {FIXED|NOT-A-BUG|WONTFIX|ALREADY FIXED}

**{Closed via Systematic Debugging investigation.|Fixed in commit X.}**

#### {Why This Is Not A Bug | The Fix | Evidence}

| Claim | Status | Evidence |
|-------|--------|----------|
| "Claim from ticket" | ✅ TRUE / ❌ FALSE | Actual finding |

#### {Additional sections as needed}

- Why impact claims were wrong
- What the intentional design is
- Why the proposed fix would be harmful

#### Severity {Downgrade|Confirmation}

- Original: P{X} (based on {reason})
- Revised: P{Y} ({new assessment})
- Resolution: {OUTCOME}
```

### Commit Conventions

```bash
# For FIXED bugs:
fix({domain}): {TICKET_ID} - brief description

# For NOT-A-BUG/WONTFIX (triage only, no code change):
triage({TICKET_ID}): {NOT-A-BUG|WONTFIX} - brief reason
```

---

## Common Patterns

### Pattern: "Unused/Unwired Code"

**Verify:** Is it actually unused, or was it wired up since the ticket was filed?

```bash
# Check for callers
grep -rn "function_name" src/esper --include="*.py" | grep -v "def function_name"

# Check git history for recent wiring
git log --oneline --grep="function_name\|TICKET_ID" -- src/
```

### Pattern: "Wrong Formula/Calculation"

**Verify:** Is this the ONLY calculation, or part of a composite system?

```bash
# Find ALL related calculations
grep -rn "bonus\|reward\|scale" src/esper/simic/rewards/

# Trace the full reward flow
```

### Pattern: "Edge Case Could Cause X"

**Verify:** Is the edge case structurally possible?

```bash
# Find ALL code paths that could trigger the edge case
# Trace backwards from the symptom to the cause
# Check if invariants prevent the edge case
```

### Pattern: "Exaggerated Severity"

**Common downgrades:**

| Original | Actual | Reason |
|----------|--------|--------|
| P1 "correctness" | P4 cosmetic | Only affects UI display |
| P2 "training impact" | P4 not-a-bug | Value is SET but never READ |
| P1 "could explode" | P4 unreachable | Edge case is structurally impossible |

---

## Specialist Consultation

Invoke specialists when the fix touches their domain:

| Domain | Specialist |
|--------|------------|
| RL algorithms, rewards, policy | `drl-expert` agent or `yzmir-deep-rl` skills |
| Tensor ops, GPU, torch.compile | `pytorch-expert` agent or `yzmir-pytorch-engineering` skills |
| Python patterns, type system | `axiom-python-engineering` skills |
| Test coverage, flaky tests | `ordis-quality-engineering` skills |

---

## Quick Reference Checklist

### Before Proposing Any Fix

- [ ] Read the actual code (not just trust ticket's line numbers)
- [ ] Verify TODO comments exist (if ticket claims they do)
- [ ] Check git history for recent changes
- [ ] Trace data flow to verify impact claims
- [ ] Understand the full system context

### For NOT-A-BUG Resolution

- [ ] Document which claims were FALSE
- [ ] Explain the intentional design
- [ ] Show evidence trail (grep commands, git history)
- [ ] Downgrade severity with rationale

### For FIXED Resolution

- [ ] Fix addresses root cause (not symptoms)
- [ ] Tests pass
- [ ] mypy passes on modified files
- [ ] No defensive programming patterns added

### For ALREADY FIXED Resolution

- [ ] Find the commit that fixed it
- [ ] Verify the fix is complete
- [ ] Document the commit hash

---

## Folder Structure

```
docs/bugs/
├── triaged/                    # Validated bugs awaiting fix (start here)
├── fixed/                      # Closed tickets (fix verified or already fixed)
├── wontfix/                    # Valid but intentional behavior
├── not-a-bug/                  # Invalid findings (false positive, exaggerated)
└── ticket-template.md          # Template for new tickets

docs/process/
├── TRIAGE.md                   # Triage process (run first)
└── BUGFIX.md                   # This file (fix triaged tickets)
```

---

## Related Documentation

- **[TRIAGE.md](./TRIAGE.md)** - Triage process (run before this)
- **[ticket-template.md](../bugs/ticket-template.md)** - Template for new tickets
- **superpowers:systematic-debugging** - Full systematic debugging skill
