# Bug Triage Process

This document defines the process for triaging bug tickets from the 2712 quality audit.

---

## Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Gather    │───▶│   Validate   │───▶│  Classify   │───▶│    File      │
│   Tickets   │    │   Finding    │    │  Disposition│    │   Movement   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

Triage is **read-only analysis**. No bug fixing occurs during triage. The goal is to determine which bugs are still valid and need fixing vs. which are already resolved, invalid, or won't be fixed.

---

## Target Folder Structure

After triage, bugs are organized into four destination folders:

```
docs/bugs/
├── triaged/           # Valid bugs that still need fixing
├── fixed/             # Already fixed (code changed, bug resolved)
├── wontfix/           # Valid finding but intentional behavior / by design
├── not-a-bug/         # Invalid finding (false positive, outdated, duplicate)
└── archive-pre-2712/  # Legacy tickets (separate audit, untouched)
```

### Folder Definitions

| Folder | Criteria | Action After Triage |
|--------|----------|---------------------|
| `triaged/` | Bug confirmed present in current code | Follow BUGFIX.md process |
| `fixed/` | Code has changed; bug no longer exists | No action needed |
| `wontfix/` | Finding is valid but behavior is intentional | Document rationale |
| `not-a-bug/` | False positive, outdated reference, or duplicate | No action needed |

---

## Triage Decision Tree

For each bug ticket, follow this decision tree:

```
1. CODE EXISTS?
   │
   ├─ NO: File/function removed or renamed?
   │      └─ Search for relocated code
   │         ├─ Found elsewhere → Continue to step 2 with new location
   │         └─ Not found → NOT-A-BUG (outdated reference)
   │
   └─ YES: Continue to step 2

2. BUG PATTERN EXISTS?
   │
   ├─ NO: The specific issue no longer exists
   │      └─ FIXED (cite commit/change if determinable)
   │
   └─ YES: Continue to step 3

3. IS IT ACTUALLY A BUG?
   │
   ├─ NO: The code is correct as written
   │      └─ NOT-A-BUG (false positive)
   │
   └─ YES: Continue to step 4

4. IS IT INTENTIONAL?
   │
   ├─ YES: Documented design decision or tradeoff
   │       └─ WONTFIX (cite rationale)
   │
   └─ NO: Continue to step 5

5. IS IT A DUPLICATE?
   │
   ├─ YES: Another ticket covers same issue
   │       └─ NOT-A-BUG (duplicate of TICKET_ID)
   │
   └─ NO: TRIAGED (valid bug, needs fix)
```

---

## Validation Checklist

When validating a bug ticket, verify:

| Check | Method |
|-------|--------|
| File exists | `ls` / `Read` the cited file |
| Line numbers match | Read file at specified lines |
| Code pattern matches | Compare ticket's "Code Evidence" to actual code |
| Function/class exists | Grep for the cited function/class name |
| Issue is reproducible | Check if the described behavior can occur |

### Common Reasons for Disposition Changes

**FIXED:**
- Code refactored and bug pattern removed
- Function renamed/relocated and issue addressed
- Related ticket already fixed this
- Defensive code removed per project policy

**NOT-A-BUG:**
- False positive (reviewer misunderstood code)
- File/function deleted entirely
- Duplicate of another ticket
- Outdated (code has changed significantly)

**WONTFIX:**
- Intentional design tradeoff (documented)
- Low priority vs. complexity cost
- Conflicts with architectural principles

---

## Triage Manifest Format

Each triage session produces a manifest documenting dispositions:

```markdown
# Triage Manifest: Batch X

**Triaged by:** [agent/person]
**Date:** YYYY-MM-DD
**Tickets reviewed:** N

## FIXED (N tickets)

| Ticket | Original Finding | Resolution |
|--------|------------------|------------|
| B2-DRL-08 | get_alpha() naming | Renamed in commit abc123 |
| B2-DRL-09 | Missing validation | Pattern no longer present |

## NOT-A-BUG (N tickets)

| Ticket | Original Finding | Reason |
|--------|------------------|--------|
| B2-DRL-10 | Defensive .get() | Legitimate PyTorch API usage |
| B2-DRL-11 | Buffer leak | Duplicate of B3-DRL-05 |

## WONTFIX (N tickets)

| Ticket | Original Finding | Rationale |
|--------|------------------|-----------|
| B2-DRL-12 | No death penalty | Design decision per ROADMAP.md |

## TRIAGED (N tickets)

| Ticket | Severity | Summary | Notes |
|--------|----------|---------|-------|
| B2-DRL-13 | P2 | Missing reward signal | Confirmed in current code |
```

---

## Parallel Triage with Explore Agents

For large-scale triage (100+ tickets), use parallel Explore agents:

### Agent Configuration

| Parameter | Value |
|-----------|-------|
| Agent type | `Explore` (read-only) |
| Thoroughness | `medium` |
| Scope | 1-2 batch folders per agent |
| Output | Triage manifest markdown |

### Agent Prompt Template

```
Triage bug tickets in docs/bugs/batchN/.

For each B*.md ticket:
1. Read the ticket to understand the finding
2. Check if the cited file/line/function exists
3. Verify if the bug pattern is still present
4. Classify as: FIXED, NOT-A-BUG, WONTFIX, or TRIAGED

Produce a triage manifest following the format in docs/process/TRIAGE.md.

DO NOT modify any files. This is read-only analysis.
```

### Consolidation

After parallel agents complete:
1. Collect all manifests
2. Check for conflicts (same ticket classified differently)
3. Resolve conflicts with deeper analysis
4. Present unified report for approval
5. Execute file moves

---

## Post-Triage Actions

### Moving Files

After triage approval, move tickets to destination folders:

```bash
# Fixed bugs
mv docs/bugs/batchN/TICKET.md docs/bugs/fixed/

# Invalid/duplicate/outdated
mv docs/bugs/batchN/TICKET.md docs/bugs/not-a-bug/

# Intentional behavior
mv docs/bugs/batchN/TICKET.md docs/bugs/wontfix/

# Valid bugs needing fix
mv docs/bugs/batchN/TICKET.md docs/bugs/triaged/
```

### Updating Ticket Status

Before moving, update the ticket's Status field:

| Disposition | Status Value |
|-------------|--------------|
| FIXED | `closed` |
| NOT-A-BUG | `invalid` |
| WONTFIX | `wont-fix` |
| TRIAGED | `triaged` |

### Cleanup

After all tickets are moved from a batch folder:
1. Keep batch summary files (`BATCHN-SUMMARY.md`) in an archive
2. Delete empty batch folders
3. Update `docs/bugs/SUMMARY.md` with final counts

---

## Metrics

Track triage progress and outcomes:

| Metric | Description |
|--------|-------------|
| Total tickets | Starting count of tickets to triage |
| Triage rate | Tickets triaged per session |
| Fixed % | Percentage already fixed |
| Invalid % | Percentage false positives |
| Valid % | Percentage still need fixing |

---

## Related Documentation

- **[BUGFIX.md](./BUGFIX.md)** - Process for fixing triaged bugs
- **[ticket-template.md](../bugs/ticket-template.md)** - Template for new tickets
- **[SUMMARY.md](../bugs/SUMMARY.md)** - Bug statistics overview
