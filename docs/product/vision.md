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
  *(Live instance, 2026-07-14, PDR-0074/0076: a fossil's contribution is structurally unmeasured — the counterfactual is
  undefined at permanence — and coerced `None→0` in reward, observation, and value; the policy learned that permanence is
  worthless because we told it so. The current Now bet, "make permanence visible," directly serves this commandment. The
  proposed settlement fix must also honour the Goodhart anti-goal above — a mis-priced settlement would farm fossils.)*
- A codebase carrying **legacy / backwards-compat / shim** code (No Legacy Code policy).

## Authority grant
Granted by: john (GitHub: tachyon-beep)     Last reviewed: 2026-07-10
Review cadence: on any vision change, or monthly — whichever first.

Autonomous within strategy — the agent MAY, without asking:
  prioritize the backlog, write specs/PRDs, dispatch delivery, **launch/kill GPU
  experiment runs within the active bet**, run analysis, accept against criteria,
  reprioritize, kill a failing bet per metrics.md, and **commit to the workspace at
  checkpoint**.
  **Run authorization (owner-stated 2026-07-10, restated and broadened):** the agent
  may CREATE NEW EXPERIMENTS, EXTEND runs, or ADD runs at its own discretion whenever
  it judges that previous runs did not give us everything we need — within the active
  research program, each recorded as a PDR with a pre-committed reading (the PDR-0053/
  0054 pattern). Pre-registered acceptance gates stay owner-gated as before.
  **Experiment-value principle (owner, 2026-07-10):** prefer tossing a week and
  restarting with an experiment that answers the question 100% over salvaging a
  near-done run that answers 20%. The test for any run, salvage, or redesign is:
  *"will this give us insights that inform future decisions — by ruling things in or
  out, or verifying a theory?"* Sunk cost and near-completeness are not reasons to
  keep a compromised instrument.

Escalate BEFORE acting — the agent MUST get owner sign-off for:
  changing this vision/strategy/grant; **pushing/tagging/releasing or any
  GitHub-remote/external action**; deprecating a subsystem or reward mode others rely on;
  deleting telemetry/run data; anything touching an external party.
  Standing rules (always): git identity stays **tachyon-beep** (never johnm-dta without
  explicit say-so); **never push without an explicit ask**; no destructive git without
  permission.
  (Taxonomy + rationale: product-ownership-operating-model.md.)
