# Vision — Esper (Morphogenetic Neural Networks)

## Purpose
Esper is a framework for **morphogenetic AI**: neural networks that grow, prune, and
adapt their own topology *during* training, driven by a learned RL controller (Tamiyo)
under an explicit selection-pressure economy (Simic). The change it exists to make is to
move AI from **Architectural Engineering** (a human commits to a static architecture up
front) to **Architectural Ecology** (structure is germinated, trained safely in
isolation, and only committed when it earns its keep). The controller is a *persistent
product that accumulates "structural taste"* across projects — not a disposable per-run
script.

## Who it serves
- **Primary:** the project maintainer and ML researchers building self-modifying /
  grown neural architectures who need an instrumented, reproducible substrate for
  in-training topological growth.
- **Secondary:** the morphogenetic-AI / AutoML-adjacent research community (eventually,
  on publication). *(assumption — inferred from README framing "platform, not a model"; confirm.)*
- **Explicitly not:** a production AutoML/NAS service for end users; not a general
  drop-in training-framework competitor. The value is the *grown-during-training*
  mechanism and its instrumentation, not turnkey model delivery.

## Anti-goals (what it refuses to be)
- A **static NAS / hyperparameter search** that picks an architecture before training —
  the entire thesis is *in-training* topological growth.
- A reward design the controller can **Goodhart** — reward and analytics must reflect
  *accuracy minus rent* and genuine committed counterfactual value, never a dense signal
  the policy can farm without improving the host.
- A body that grows a capability **without a corresponding sensor** (Signal-to-Noise
  commandment: if the policy can't see it, it can't optimise it).
- A codebase carrying **legacy / backwards-compat / shim** code (No Legacy Code policy).

## Authority grant
Granted by: john (GitHub: tachyon-beep)     Last reviewed: 2026-06-28
Review cadence: on any vision change, or monthly — whichever first.

Autonomous within strategy — the agent MAY, without asking:
  prioritize the backlog, write specs/PRDs, dispatch delivery, **launch/kill GPU
  experiment runs within the active bet**, run analysis, accept against criteria,
  reprioritize, kill a failing bet per metrics.md, and **commit to the workspace at
  checkpoint**.

Escalate BEFORE acting — the agent MUST get owner sign-off for:
  changing this vision/strategy/grant; **pushing/tagging/releasing or any
  GitHub-remote/external action**; deprecating a subsystem or reward mode others rely on;
  deleting telemetry/run data; anything touching an external party.
  Standing rules (always): git identity stays **tachyon-beep** (never johnm-dta without
  explicit say-so); **never push without an explicit ask**; no destructive git without
  permission.
  (Taxonomy + rationale: product-ownership-operating-model.md.)
