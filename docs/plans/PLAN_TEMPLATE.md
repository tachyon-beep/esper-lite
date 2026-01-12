# Plan Metadata Template

Use this template at the top of any plan or concept document to enable tracking and prioritization.

---

## Metadata Block (copy this to your plan)

```yaml
# Plan Metadata
id: <short-kebab-case-id>           # e.g., "simic2-vectorized-refactor"
title: <human readable title>
type: concept | planning | ready | in-progress | completed | abandoned
created: YYYY-MM-DD
updated: YYYY-MM-DD
owner: <who is driving this>        # Optional for concepts

# Prioritization
urgency: critical | high | medium | low | backlog
  # critical = blocking other work or causing active pain
  # high = needed for next milestone
  # medium = valuable but not blocking
  # low = nice to have
  # backlog = someday/maybe

value: <1-2 sentence description of what we get>

# Constraints
complexity: S | M | L | XL
  # S = < 1 day, single file/module
  # M = 1-3 days, few files, well-understood
  # L = 1-2 weeks, cross-domain, needs design
  # XL = 2+ weeks, architectural, high uncertainty

risk: low | medium | high | critical
  # low = well-understood, easily reversible
  # medium = some unknowns, reversible with effort
  # high = significant unknowns, hard to reverse
  # critical = could destabilize system if wrong

risk_notes: <what specifically could go wrong>

# Dependencies
depends_on:
  - <plan-id>                       # Hard dependency - must complete first
  - <plan-id>

soft_depends:
  - <plan-id>                       # Benefits from but doesn't require

blocks:
  - <plan-id>                       # What is waiting on this

# Status
status_notes: <current state, next action, blockers>
percent_complete: 0-100             # Optional, for in-progress items

# Expert Review (REQUIRED before promotion to ready)
reviewed_by:
  - reviewer: drl-expert | pytorch-expert | python-engineering | neural-arch | training-opt
    date: YYYY-MM-DD
    verdict: approved | approved-with-changes | needs-revision
    notes: <key findings, concerns, or required changes>
```

---

## Field Guidance

### Type Lifecycle

```
concept ──► planning ──► ready ──► in-progress ──► completed
                │           │            │
                └───────────┴────────────┴──► abandoned
```

- **concept**: Early idea, exploring feasibility. May never happen.
- **planning**: Actively designing. Has a workspace folder.
- **ready**: Approved, scoped, implementation-ready. Promoted to `docs/plans/ready/`.
- **in-progress**: Being executed.
- **completed**: Done. Move to `docs/plans/completed/`.
- **abandoned**: Superseded or cancelled. Move to `docs/plans/abandoned/`.

### Urgency vs Value vs Risk

These are orthogonal:
- A **high-value, low-urgency** item might wait for a quiet period
- A **low-value, high-urgency** item might be tech debt causing pain
- A **high-value, high-risk** item needs careful sequencing and validation

### Complexity Calibration

| Size | Typical Scope | Example |
|------|---------------|---------|
| S | Single function, config change, doc update | Add a new blueprint variant |
| M | One module, clear requirements | Refactor a reward component |
| L | Cross-module, needs coordination | Kasmina2 Phase 0 |
| XL | Architectural, research-adjacent | Emrakul immune system |

### Risk Categories

Consider these risk types:
- **Technical**: Will it work? Do we understand the problem?
- **Stability**: Could it break training or existing behavior?
- **Integration**: Does it touch many systems?
- **Reversibility**: Can we undo it if wrong?
- **Scope creep**: Is it well-bounded?

### Expert Review Requirements

**Plans MUST be reviewed by relevant specialists before promotion to `ready` status.**

| Plan Domain | Required Reviewer(s) |
|-------------|---------------------|
| RL training, rewards, policies | `drl-expert` agent + `yzmir-deep-rl` skills |
| PyTorch, tensors, torch.compile | `pytorch-expert` agent + `yzmir-pytorch-engineering` skills |
| Python patterns, architecture | `axiom-python-engineering` skills |
| Neural network design | `yzmir-neural-architectures` skills |
| Training stability, optimization | `yzmir-training-optimization` skills |

**Cross-domain plans require multiple reviewers.** Most non-trivial Esper work touches RL + PyTorch + Python, so expect 2-3 specialist reviews.

Review verdicts:
- **approved**: No blocking issues, ready to proceed
- **approved-with-changes**: Minor issues identified, can proceed after addressing
- **needs-revision**: Significant concerns, requires design changes before re-review

---

## Example

```yaml
# Plan Metadata
id: simic2-phase1-vectorized-split
title: Simic Vectorized Module Split
type: in-progress
created: 2025-12-20
updated: 2026-01-08
owner: Claude

urgency: high
value: Unblock all Simic changes by making vectorized.py maintainable

complexity: L
risk: medium
risk_notes: Behavioral regression possible; mitigated by baseline tests

depends_on: []
soft_depends: []
blocks:
  - simic2-phase2-typed-contracts
  - kasmina2-phase0-simic-training

status_notes: Phase 1 extraction complete, running baseline comparison
percent_complete: 75

reviewed_by:
  - reviewer: pytorch-expert
    date: 2025-12-22
    verdict: approved-with-changes
    notes: Ensure baseline parity tests cover edge cases in gradient accumulation
  - reviewer: python-engineering
    date: 2025-12-22
    verdict: approved
    notes: Module split follows standard patterns, no concerns
```
