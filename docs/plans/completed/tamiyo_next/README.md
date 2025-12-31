# Tamiyo Next Implementation Guide

## How to Use This Directory

This directory contains the **Obs V3 + Policy V2 implementation plan**, split into manageable phases.

### Authoritative Sources

**Execute from the individual phase documents.** They are authoritative over the parent 6000-line file:

| File | Phase | Status |
|------|-------|--------|
| `00-preamble.md` | Prerequisites & setup | Reference |
| `01-phase1-leyline-constants.md` | Leyline constants | ✅ COMPLETE |
| `02-phase2-obs-v3-features.md` | Obs V3 feature extraction | Execute |
| `03-phase3-blueprint-embedding.md` | Blueprint embedding module | Execute |
| `04-phase4-policy-network-v2.md` | Policy network V2 | Execute |
| `05-phase5-training-config.md` | Training configuration | Execute |
| `06-phase6-vectorized-training.md` | Vectorized training integration | Execute |
| `07-phase7-validation-testing.md` | Validation & testing | Execute |
| `08-phase8-cleanup-appendix.md` | Cleanup & reference | Execute |

### When to Consult the Parent Document

If there is **ambiguity** in a phase document, consult `tamiyo_next.md` for:

- Cross-phase dependencies and ordering
- Rationale behind design decisions
- Edge cases that span multiple phases
- The "why" when the phase doc only gives the "what"

### Supporting Design Documents

- `2025-12-30-obs-v3-design.md` - Observation space V3 design rationale
- `2025-12-30-policy-v2-design.md` - Policy network V2 architecture design

### Implementation Order

Follow the phase numbers in sequence (1 → 2 → 3 → ...). Phase 1 is already complete.

---

## Subagent-Driven Development

**Always use subagent-driven development.** Your job is to implement a single specified phase in full, using the following workflow:

### Phase 1: Pre-Implementation Review (Go/No-Go)

Before starting any phase, spawn a **three-person specialist panel** to review the phase document:

| Specialist | Focus Area |
|------------|------------|
| **DRL Specialist** | RL correctness, advantage estimation, policy gradients, reward handling |
| **PyTorch Specialist** | Tensor operations, memory, dtypes, device handling, torch.compile compatibility |
| **Code Reviewer** | Python patterns, API consistency, test coverage, CLAUDE.md compliance |

**Panel task:** Review the phase document and confirm:
1. The proposed changes are technically sound
2. No obvious gaps or contradictions with the codebase
3. Dependencies from prior phases are satisfied
4. Test strategy is adequate

**Output:** Each specialist provides `✅ GO` or `❌ NO-GO` with reasoning. Proceed only on unanimous approval.

### Phase 2: Implementation

If the panel approves, use subagents to implement the phase:

1. **Break the phase into discrete tasks** (use TodoWrite)
2. **Spawn implementation subagents** for each task (group by file/module when possible)
3. **Run tests incrementally** as each component is completed
4. **Track progress** via the todo list

### Phase 3: Post-Implementation Sign-Off

After implementation, reconvene the **same three-person panel** to verify:

| Specialist | Verification |
|------------|--------------|
| **DRL Specialist** | RL math is correct, no bias introduced, training loop integrity |
| **PyTorch Specialist** | No memory leaks, correct dtypes, gradients flow properly |
| **Code Reviewer** | Tests pass, code matches plan, no regressions |

**Output:** Each specialist provides `✅ APPROVED` or `❌ NEEDS REVISION` with specific findings.

### Example Workflow

```
User: "Implement Phase 3: Blueprint Embedding Module"

Claude:
1. Read 03-phase3-blueprint-embedding.md
2. Spawn 3 specialists in parallel for go/no-go review
3. If approved → create TodoWrite tasks → spawn implementation agents
4. After implementation → spawn 3 specialists for sign-off
5. If approved → commit with sign-off summary
```

### Why This Matters

- **Pre-review catches plan errors** before they become code bugs
- **Specialists stay focused** on their domain expertise
- **Post-review catches implementation errors** before they ship
- **Context stays manageable** by delegating to subagents
