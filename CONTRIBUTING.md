## Introduction

Thanks for taking an interest in Esper. This project is an experiment in “architectural ecology” for neural networks: we let models grow, stabilise, and slim down while training, under strict rules about determinism, safety, and observability. It’s fun, it’s weird, and it’s very easy to accidentally break if you add code that hides errors or lies about performance.

These guidelines exist to protect everyone’s time. Esper’s maintainers are happy to review and merge changes, but the bar is high because the system is designed to be *debuggable*. If a change makes the telemetry less trustworthy, introduces hidden CPU↔GPU synchronisation, or quietly bypasses contracts, it will be rejected even if it “works”.

### What we’re looking for

Contributions that help most:

* **Bug fixes** that improve correctness, determinism, or stability.
* **Performance work** that removes overhead without changing semantics (especially anything that reduces Python overhead in hot paths or avoids GPU sync points).
* **Telemetry improvements** that make the system easier to reason about (new metrics, better aggregation, clearer UI), provided they keep strict schemas and don’t introduce “optional key” drift.
* **Tests** (unit, integration, property, performance, mutation) that lock down invariants and prevent regressions.
* **Documentation** that explains contracts, invariants, and how to reproduce/diagnose behaviours.

If you’re unsure where to start, documentation and tests are always welcome. Esper is built to survive refactors; high quality tests and clear contracts are a direct contribution to that goal.

### What we’re not looking for

To keep the codebase legible and the signals trustworthy, we generally won’t accept:

* **Defensive programming that masks bugs**, such as `dict.get`, `getattr/hasattr` fallbacks, or broad exception swallowing, unless it’s explicitly justified and whitelisted.
* **Unreviewed synchronisation points** (`.cpu()`, `.item()`, `.numpy()`, `torch.cuda.synchronize`, etc.) in hot paths. If you need one, it must be explicitly whitelisted with a reason.
* **Hooks and hidden callbacks** for core model wiring. Esper prefers explicit routing and deterministic ordering over implicit interception.
* **Large feature additions without an issue/design note**. Anything that changes contracts (action space, slot ordering, telemetry payloads, reward interfaces) should start with a short proposal.
* **“Quick fixes” that change behaviour** without tests. If it changes semantics, it needs coverage.

### Ground rules

Esper is built around a few non-negotiables:

* **Determinism and replayability matter.** If we can’t reproduce behaviour, we can’t debug it.
* **Sensors must match capabilities.** Telemetry is treated as an API, with typed payloads and schema tests.
* **No silent fallbacks.** If a field is missing or a contract changes, we prefer failing fast to continuing with guessed defaults.
* **Performance hazards are explicit.** CPU↔GPU sync points and other slow paths must be visible and reviewed.
* **Small, scoped pull requests are preferred.** Changes are easiest to review and safest to merge when they are narrow and well tested.

If you’re new to open source, this process might feel strict. It’s strict because Esper is an evolving system where subtle mistakes can look like “the model is learning weirdly” for days before anyone realises it was an integration bug.

### Your first contribution

Good first contributions include:

* Improving docs for a specific subsystem (Tolaria, Simic, Kasmina, Tamiyo, Emrakul, Narset, Esika).
* Adding or strengthening tests around a contract boundary (slot ordering, mask invariants, telemetry schemas, buffer shapes).
* Fixing a small bug with a clear reproduction and a test.
* Tightening a performance invariant (removing a stray `.item()` or eliminating an accidental sync) and adding a regression test.
* Creating new experimental configurations that have revealed something unusual or remarkable.

If you’ve never made a pull request before, these guides are helpful:

* [https://makeapullrequest.com/](https://makeapullrequest.com/)
* [https://www.firsttimersonly.com/](https://www.firsttimersonly.com/)

### Getting started

At a high level:

1. Fork the repository and create a branch.
2. Make a small, scoped change.
3. Add or update tests that prove correctness.
4. Run the project’s test suite locally (or the relevant subset if the suite is large).
5. Open a pull request with:

   * what you changed
   * why it’s needed
   * what tests cover it
   * any performance implications (especially sync points or compile behaviour)

Small “obvious fixes” (typos, docs, comment corrections) are welcome and usually don’t need extensive discussion. Anything that touches contracts, action space, slot ordering, reward semantics, or telemetry schemas should start with an issue or design note so we can align before you invest a lot of work.
