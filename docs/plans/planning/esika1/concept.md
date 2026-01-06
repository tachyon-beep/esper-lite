# Esika Design Note

## Host superstructure as container and deconfliction ruleset

**Status:** One-page concept note (planning, no code)
**Purpose:** Define Esika as the host-level container that stitches multiple Kasmina regions into one coherent system, hosts Narset, and enforces deconfliction rules and safe boundaries. Esika is not a policy.

---

## 1) What Esika is

**Esika is the host superstructure**: a container and ruleset that coordinates a mesh of Kasmina “cells” (each containing seeds/phages and governed locally by a Tamiyo/Emrakul pair). Esika provides:

* a stable topology (what regions exist and how they connect)
* deconfliction rules (who is allowed to act where and when)
* safe-boundary scheduling (when irreversible operations may occur)
* an execution context for Narset (budget allocation across regions)

Esika is *not* learned. She is infrastructure.

---

## 2) Why Esika exists

At small scale, a single Kasmina is enough. At scale (dozens to thousands of slots), you need a layer that ensures:

* global coordination doesn’t leak into local decision logic
* regions remain stable and independently debuggable
* deterministic replay remains possible across many concurrent regions
* “maintenance” actions happen only at safe boundaries
* resource allocation is consistent and auditable

Without Esika, coordination pressure tends to collapse back into one of the existing modules (typically Simic or Tolaria), recreating the “god object” problem.

---

## 3) What Esika does

### A. Topology and identity

* Defines the region graph: a mesh/tree/stack of Kasmina cells.
* Assigns stable region IDs and ordering invariants.
* Owns the “layout manifest” so telemetry, ledgers, and budgets have unambiguous keys.

### B. Deconfliction rules (hard constraints)

Esika enforces **physics**, not strategy:

* Jurisdiction boundaries: which Tamiyo/Emrakul pair owns which region.
* Action timing rules: what is allowed mid-episode vs only at safe boundaries.
* Blast-radius constraints: irreversible operations are scheduled, not executed ad hoc.
* Coordination invariants: no cross-region mutation without explicit, logged migration/replication steps.

### C. Safe-boundary scheduling

Esika schedules operations that must happen at controlled boundaries, such as:

* compaction / physical rewrites
* physical lysis events
* region replication/migration (copying a design into another region)
* checkpoint orchestration for deterministic replay

### D. Host Narset (allocator)

Esika “hosts Narset” in the sense that:

* Narset runs in Esika’s namespace (global view of region summaries)
* Narset’s outputs are applied through Esika’s budget plumbing
* Esika ensures Narset only allocates budgets and cannot reach into local module/submodule decisions

---

## 4) What Esika does NOT do

Esika is not a policy and must not drift into one.

Esika does not:

* choose slots, blueprints, alpha targets, or subtargets
* interpret seed semantics or lifecycle beyond enforcement of allowed transitions
* implement reward shaping or pricing (Simic does that)
* run probes/sedation/lysis logic (local Emrakul/phage wrappers do that)
* create “smart” behaviour at the superstructure level (that’s Narset’s allocator job, and even Narset stays coarse)

---

## 5) Interfaces (conceptual)

### Inputs Esika consumes

* Region telemetry summaries (progress, stability, efficiency proxies)
* Safe-boundary signals (epoch/batch/checkpoint boundaries)
* Local capability declarations (what each region supports, schema versions)

### Outputs Esika produces

* Budget allocations and constraints per region (via Narset)
* Scheduling plans for safe-boundary operations
* Stable topology manifests and IDs for aggregation and replay

---

## 6) Design principles

1. **Infrastructure, not intelligence:** Esika is a container and ruleset.
2. **Physics-first:** enforce invariants and safe boundaries; never “pick actions”.
3. **Deterministic and auditable:** all topology and scheduling decisions are logged.
4. **Local autonomy:** Tamiyo/Emrakul pairs remain the only fine-grained actors.
5. **Prevent re-centralisation:** Esika must never become “Simic but bigger”.

---

## 7) One-sentence definition

**Esika is the superstructure: a deterministic container and deconfliction ruleset that hosts Narset and coordinates many Kasmina regions, enforcing stable IDs, safe boundaries, and budget plumbing without becoming a policy.**
