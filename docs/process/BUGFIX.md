# Bug Ticket Remediation Process

This document codifies the process for **fixing** bug tickets that have been triaged.

**Prerequisite:** Tickets should be triaged first using the [TRIAGE.md](./TRIAGE.md) process. This document applies to tickets in `docs/bugs/triaged/`.

---

## Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Review    │───▶│   Verify     │───▶│  Implement  │───▶│   Sign-Off   │───▶│    Close    │
│   Ticket    │    │   Finding    │    │     Fix     │    │  (Required)  │    │   Ticket    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
```

---

## Step 1: Review Ticket

Read the ticket and assess:

| Field | Action |
|-------|--------|
| **Severity** | P0-P1 = mandatory systematic debugging; P2-P4 = judgment based on complexity |
| **Complexity** | Assess if bug warrants systematic debugging regardless of priority |
| **Category** | Understand the type of issue (correctness, API design, performance, etc.) |
| **Cross-Review** | Check what other agents said (DRL, PyTorch, CodeReview verdicts) |
| **Recommended Fix** | Review proposed solution(s) |
| **Status** | Update to `investigating` when starting work (see `ticket-template.md` for all valid status values) |

---

## Step 2: Verify Finding

Before implementing, confirm the issue exists:

1. **Read the actual code** at the specified location
2. **Reproduce the symptom** if applicable (grep for patterns, run tests)
3. **Assess accuracy** — is the ticket correct? Partially correct? Outdated?

If the finding is **invalid or already fixed**, document why and close as "won't fix" or "already fixed".

---

## Step 3: Implement Fix

### For P0-P1 (Critical/High Priority)

**Systematic debugging is MANDATORY:**

- Use the `superpowers:systematic-debugging` skill
- Follow the full diagnostic process before implementing
- Document root cause analysis

### For P2-P4 (Medium/Low Priority)

Use judgment based on complexity:

| Complexity | Approach |
|------------|----------|
| Simple | Implement directly |
| Moderate | Plan approach, then implement |
| Complex | **Use systematic debugging** (same as P0-P1) |

**Complexity indicators that warrant systematic debugging:**

- Multiple interacting components
- Unclear root cause
- Risk of introducing regressions
- Previous fix attempts failed
- Touches critical paths (training loop, tensor ops, state management)

### Specialist Consultation

Invoke specialists when the fix touches their domain:

| Domain | Specialist |
|--------|------------|
| RL algorithms, rewards, policy | `drl-expert` agent or `yzmir-deep-rl` skills |
| Tensor ops, GPU, torch.compile | `pytorch-expert` agent or `yzmir-pytorch-engineering` skills |
| Python patterns, type system | `axiom-python-engineering` skills |
| Test coverage, flaky tests | `ordis-quality-engineering` skills |

### Implementation Checklist

- [ ] Fix addresses root cause (not just symptoms)
- [ ] No new `type: ignore` comments introduced
- [ ] No defensive programming patterns added
- [ ] Tests enhanced, extended or updated if behavior added, removed or changed
- [ ] mypy passes on modified files
- [ ] Relevant tests pass

### Commit Conventions

- **One commit per ticket** (atomic changes preferred)
- **Format:** `fix(domain): {TICKET_ID} - brief description`
- **Record in ticket:** Add commit hash to the Resolution → Commits section
- **Multi-file fixes:** Keep as single commit if logically atomic

### Partial Fixes

If a fix only addresses part of a ticket:

1. **Complete what you can** — implement the portions that are ready
2. **Split the ticket** — create new ticket(s) for remaining work with cross-references
3. **Document in Resolution** — explain what was fixed and what was deferred
4. **Get sign-off for completed portion** — the originator confirms partial fix is acceptable
5. **Close original, leave new tickets open** — maintain clear audit trail

---

## Step 4: Sign-Off (REQUIRED)

**Every ticket must be signed off by its originator before closure.**

| Ticket Agent | Sign-Off Via Task Tool |
|--------------|------------------------|
| `codereview` | `subagent_type="feature-dev:code-reviewer"` |
| `drl` | `subagent_type="drl-expert"` |
| `pytorch` | `subagent_type="pytorch-expert"` |

### Sign-Off Request Template

```
Review the fix for ticket {TICKET_ID}: "{TICKET_TITLE}"

## Problem
{Brief description of the original issue}

## Fix Applied
{Description of what was changed}

## Changes Made
{List of files and changes}

## Verification
- mypy passes
- {N} tests pass
- {Any other verification}

Please confirm this fix fully addresses {TICKET_ID} and sign off.
```

### Sign-Off Outcomes

| Outcome | Action |
|---------|--------|
| **Approved** | Move ticket to `docs/bugs/fixed/` |
| **Approved with suggestions** | Apply suggestions, then move to fixed |
| **Rejected** | Address feedback, re-request sign-off |

---

## Step 5: Close Ticket

Update the ticket's **Status** field and move to the appropriate folder:

### Fixed Tickets

```bash
# Update status to 'closed' in the ticket, then:
mv docs/bugs/triaged/{TICKET_ID}.md docs/bugs/fixed/
```

### Won't Fix (Intentional Behavior)

```bash
# Update status to 'wont-fix' in the ticket, add Resolution section explaining why, then:
mv docs/bugs/triaged/{TICKET_ID}.md docs/bugs/wontfix/
```

### Invalid / Duplicate

```bash
# Update status to 'invalid' in the ticket, add Resolution section explaining why, then:
mv docs/bugs/triaged/{TICKET_ID}.md docs/bugs/not-a-bug/
```

### Already Fixed by Another Ticket

```bash
# Add to Resolution: "Fixed by {OTHER_TICKET_ID}", then:
mv docs/bugs/triaged/{TICKET_ID}.md docs/bugs/fixed/
```

---

## Folder Structure

```
docs/bugs/
├── triaged/                    # Validated bugs awaiting fix (start here)
├── fixed/                      # Closed tickets (fix verified)
├── wontfix/                    # Valid but intentional behavior
├── not-a-bug/                  # Invalid findings (false positive, duplicate)
├── archive-pre-2712/           # Legacy tickets from before audit
├── SUMMARY.md                  # Statistics overview
└── ticket-template.md          # Template for new tickets

docs/process/
├── TRIAGE.md                   # Triage process (run first)
└── BUGFIX.md                   # This file (fix triaged tickets)
```

---

## Quick Reference

### Severity Guide

| Priority | Response | Debugging |
|----------|----------|-----------|
| **P0** | Immediate | Systematic (mandatory) |
| **P1** | High | Systematic (mandatory) |
| **P2** | Medium | Systematic if complex, otherwise judgment |
| **P3** | Low | Systematic if complex, otherwise judgment |
| **P4** | Trivial | Direct fix (escalate if unexpectedly complex) |

### Sign-Off is Non-Negotiable

- No ticket is closed without originator sign-off
- This ensures domain expertise validates the fix
- Prevents "fixes" that introduce new issues

### Batch Processing

Work through tickets in batch order (batch1, batch2, ...) to maintain:

- Consistent progress tracking
- Related tickets addressed together
- Clear audit trail

---

## Exceptions

### Already Fixed

If a ticket was fixed by a previous ticket:

1. Document which ticket fixed it
2. Get sign-off confirming it's resolved
3. Move to `fixed/` with note

### Won't Fix

If a ticket describes intentional behavior:

1. Document rationale
2. Get sign-off confirming decision
3. Move to `wontfix/` with note

### Invalid

If a ticket is a false positive or duplicate:

1. Document rationale
2. Move to `not-a-bug/` with note

### Deferred

If a ticket is valid but out of scope for current push:

1. Document why it's deferred
2. Leave in `triaged/` folder
3. Update ticket status to "deferred"

---

## Related Documentation

- **[TRIAGE.md](./TRIAGE.md)** - Triage process (run before this)
- **[ticket-template.md](../bugs/ticket-template.md)** - Template for new tickets
- **[SUMMARY.md](../bugs/SUMMARY.md)** - Bug statistics overview
