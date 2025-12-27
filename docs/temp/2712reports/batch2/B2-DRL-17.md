# Finding Ticket: ConvBlock in Public API Is Questionable

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-17` |
| **Severity** | `P4` |
| **Status** | `open` |
| **Batch** | 2 |
| **Agent** | `drl` |
| **Domain** | `kasmina` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/__init__.py` |
| **Line(s)** | `29, 65` |
| **Function/Class** | Package exports |

---

## Summary

**One-line summary:** `ConvBlock` is re-exported from Kasmina but is a building block from `blueprints.cnn`, not a core Kasmina concept.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [x] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

`ConvBlock` is exported in `kasmina/__init__.py` but it's an implementation detail from the `blueprints` subpackage, not a core Kasmina abstraction.

### Why This Is Low Priority

The export is convenient for users who want to build custom CNNs. It's a minor API design question, not a bug.

---

## Recommended Fix

Consider whether `ConvBlock` belongs in public API or should be imported directly from `kasmina.blueprints.cnn`.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "Severity-Tagged Findings Summary" - B2-17

---

## Cross-Review

| Verdict | Reviewer | Domain |
|---------|----------|--------|
| **OBJECT** | DRL Specialist | Deep RL |
| **ENDORSE** | PyTorch Specialist | PyTorch Engineering |

**DRL Evaluation:** ConvBlock is intentionally re-exported for user convenience when building custom CNN seeds.
Users should not need to know internal package structure; `__all__` explicitly includes it as public API.

**PyTorch Evaluation:** Re-exporting `ConvBlock` at package level is intentional convenience - users building custom CNN architectures should not need deep imports. This follows PyTorch's own pattern (e.g., `torch.nn.Conv2d` vs `torch.nn.modules.conv.Conv2d`). The ticket correctly identifies this as a design question, not a bug - recommend close as won't-fix.
