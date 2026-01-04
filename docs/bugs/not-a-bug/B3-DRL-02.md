# Finding Ticket: AttentionSeed param_estimate Doesn't Scale

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-02` |
| **Severity** | `P4` |
| **Status** | `closed` |
| **Batch** | 3 |
| **Agent** | `drl` |
| **Domain** | `kasmina` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/blueprints/cnn.py` |
| **Line(s)** | `112-163` |
| **Function/Class** | `AttentionSeed` (SE attention) blueprint |

---

## Summary

**One-line summary:** AttentionSeed `param_estimate=2000` is 65x off at high dimensions (512 channels: actual=131,584).

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [x] Performance bottleneck
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

The SE attention seed has a fixed `param_estimate=2000`, but actual parameter count scales quadratically with channels:

For 64 channels with reduction=4:
- Linear(64,16) + Linear(16,64) + bias = 64*16 + 16*64 + 64 = 2112 ✓

For 512 channels with reduction=4:
- Linear(512,128) + Linear(128,512) + bias = 512*128 + 128*512 + 512 = 131,584
- Estimate 2000 is **65x off**

### Why This Matters

High-dim attention seeds appear much cheaper than they are, creating:
- Policy bias toward attention seeds at high dimensions
- Incorrect rent economy signals
- Potential reward hacking via dimension scaling

---

## Recommended Fix

Make param_estimate scale with dimension:

```python
# SE attention param count formula:
# fc1: dim * (dim // reduction) + (dim // reduction)  [weights + bias]
# fc2: (dim // reduction) * dim + dim  [weights + bias]
# Total ≈ 2 * dim * (dim // reduction) + dim + (dim // reduction)

param_estimate=lambda dim, reduction=4: 2 * dim * (dim // reduction) + dim + (dim // reduction)
```

---

## Verification

### How to Verify the Fix

- [ ] Test param_estimate vs actual_param_count across dimension range
- [ ] Verify rent economy stability after fix
- [ ] Check for unintended policy behavior changes

---

## Related Findings

- B3-DRL-01: NormSeed param_estimate incorrect
- B3-DRL-03: DepthwiseSeed param_estimate fixed

---

## Cross-Review

| Reviewer | Verdict | Evaluation |
|----------|---------|------------|
| `code-review` | **ENDORSE** | Correctly identifies quadratic scaling issue. For SE attention with reduction=4: `2 * dim * (dim/4) + dim + dim/4` yields 2112 at 64ch and 131,584 at 512ch. The 65x error at high dimensions could bias policy toward attention seeds inappropriately. P2 is warranted given rent economy implications. |
| `drl-specialist` | **ENDORSE** | Confirmed quadratic scaling: O(channels^2/reduction). At 512 channels the 65x underestimate creates severe reward hacking incentive - policy would rationally exploit cheap-appearing attention at high dims. P2 appropriate; consider P1 if multi-scale architectures are planned. |
| `pytorch` | **ENDORSE** | Verified. SE block param count: `fc1_weights + fc2_weights + fc2_bias = C*(C/r) + (C/r)*C + C`. For 512ch/r=4: `2*512*128 + 512 = 131,584` vs estimate 2000 = 65.8x error. Severity depends on whether multi-resolution channels are planned; if seeds remain fixed at 64ch, the current estimate (2112 actual) is acceptable. |

---

## Resolution

### Status: NOT-A-BUG

**Closed via Systematic Debugging investigation.**

#### Why This Is Not A Bug

The claimed impacts are **incorrect** based on codebase analysis:

| Claim | Status | Evidence |
|-------|--------|----------|
| "Policy bias toward attention seeds" | ❌ FALSE | PPO uses static `BlueprintAction` enum from `leyline/factored_actions.py`. Action ordering is hardcoded (NOOP=0, CONV_LIGHT=1, ATTENTION=2...), not sorted by param_estimate. |
| "Incorrect rent economy signals" | ❌ FALSE | Rent calculation uses `BLUEPRINT_COMPUTE_MULTIPLIERS` dict in `nissa/analytics.py` (attention=1.35), NOT param_estimate. |
| "Reward hacking via dimension scaling" | ❌ FALSE | grep confirms `param_estimate` is not used anywhere in `simic/` (RL training) or reward calculations. |

#### What param_estimate Actually Affects

`param_estimate` is ONLY used in `BlueprintRegistry.list_for_topology()` to sort blueprints for the dynamic action enum built by `leyline/actions.py:build_action_enum()`. This enum is ONLY used by `HeuristicTamiyo` baseline (for comparison purposes), NOT by PPO training.

#### Evidence Trail

1. `leyline/actions.py:89`: `blueprints = BlueprintRegistry.list_for_topology(topology)` - builds action enum sorted by param_estimate
2. `leyline/actions.py:82`: Comment says "PPO training uses factored actions instead"
3. `leyline/factored_actions.py:20-34`: `BlueprintAction` is a static enum with hardcoded ordering
4. `grep -r "param_estimate" src/esper/simic/` - No matches (PPO/rewards don't use it)
5. `grep -r "param_estimate" src/esper/tamiyo/` - No matches (policy doesn't use it)

#### Severity Downgrade

- Original: P2 (based on incorrect impact claims)
- Revised: P4 (cosmetic - only affects baseline heuristic ordering)
- Resolution: NOT-A-BUG / WONTFIX

The param_estimate values ARE inaccurate for high channel counts (verified empirically), but this has no meaningful impact on the system since neither PPO training nor rewards use this value.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P2 - Performance/Safety"
