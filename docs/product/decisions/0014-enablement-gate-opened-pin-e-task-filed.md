# PDR-0014 — Enablement gate OPENED via the PIN-E leg; PIN-E task filed (the recorded pointer was a mis-binding)

Date: 2026-07-03   Status: accepted (owner sign-off in-session: "great, lets open the enablement
gate via its first leg")   Author: Claude (agent)
Related: PDR-0013, esper-lite-f22a1d48a7 (enablement gate), esper-lite-94869250f1 (PIN-E harness, NEW),
docs/analysis/2026-06-25-phase0-objective-and-instrumentation.md §2 (PIN E method)

## Context
Checkpoint #6 left "start the enablement gate?" blocked-on-owner. The owner granted it 2026-07-03.
DISPATCH reconciliation then found the recorded PIN-E pointer (esper-lite-3d67b09687 — cited in the
gate description, roadmap, current-state, and agent memory) was a **mis-binding**: that ID is the
EV-variance epic's Stage-0 telemetry task (parent esper-lite-f25b71c165), and a tracker-wide search
confirmed **no PIN-E issue ever existed**.

## Options
1. Reuse esper-lite-3d67b09687 (rejected — different epic, unrelated scope; would tangle the
   EV-variance and reward-redesign tracks and misattribute the EV task's dependents).
2. File a new, properly scoped task and correct every stale pointer (chosen).

## The call
Filed **esper-lite-94869250f1** scoped to the phase-0 §2 PIN-E method: near-inert **SMALL-REAL**
placebo (`noop` rejected as degenerate — deterministic-zero delta gives bit-identical leave-out
logits and a vacuous floor), small real blueprint + zero-init final layer, forced through the
FIXED_SCHEDULE proof-baseline lifecycle to FOSSILIZED, residency telemetry on. Deliverables:
per-stage bias/std/CoV of the placebo's credited contribution (the GATE-0 report line), the terminal
placebo-φ spread (k=1 ⇒ terminal LOO ≡ φ = v({s})−v(∅), measurable with the term hard-OFF —
shapley_synergy_scale stays 0.0 throughout), and a tau recommendation for the gate. Wired as a
blocker of esper-lite-f22a1d48a7; gate description, workspace, and memory pointers corrected;
task claimed and started. Enabling scale>0 remains a separate owner decision (PDR-0013 stands).

## Rationale
tau is the gate's first criterion and everything else in the gate calibrates against it; the harness
is also GATE-0's last open leg, so it is needed regardless of the enablement outcome. The pointer
correction is reconciliation, not re-planning — the tracker is authoritative for tactical status and
the workspace must not carry a dangling ID into another epic.

## Reversal trigger
If the harness cannot satisfy the spec's **non-degeneracy requirement** (the placebo's credit spread
demonstrably NOT driven by the same noise sources that move a real seed's c_t), the tau method
reverts to methodology §5 option (a) — K resampled val minibatches per step — per the phase-0 doc's
stated alternative, and esper-lite-94869250f1 re-scopes accordingly.
