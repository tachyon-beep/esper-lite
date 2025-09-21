Perfect—here’s the consolidated **Scaffolds 2.0 + Ral** proposal updated to match your governance and plane boundaries:

> **Reading note:** Agents should read **Urza** (blueprint library), **Karn** (blueprint generator), and **Tezzeret** (blueprint compiler) as additional context when reviewing the scaffold component and the Ral micro-compiler.

---

# ADR-014-ter (Final Wiring): **Scaffolds 2.0** & **Ral** with Kasmina-Owned Swaps

**Status:** PROPOSED → READY FOR IMPLEMENTATION
**Owners:** Control Plane (Tamiyo/Emrakul/Narset/Elesh), Execution Plane (Kasmina/Ral), Slow-Time Toolchain (Urza/Karn/Tezzeret/Urabrask), Orchestration (Tolaria/Oona/Nissa)
**Feature Flags:** `scaffolds_enabled`, `protected_zones_enabled`, `inverse_throttle_enabled`, `ral_enabled`
**Date:** 2025-09-15

---

## 0) What changed vs prior draft (delta)

* **Kasmina owns all swaps.** Ral **produces** on-demand, hot-swappable artifacts; **Kasmina** loads/validates/commits them.
* **Emrakul is Control Plane** (under Tamiyo). **Narset & Elesh** are also Control, issuing instructions to **Kasmina seeds** (which perform the edits).
* **Multiple Kasmina seeds per Narset.** Seeds are the mutable execution handles that **germinate into blueprints** or **deploy as scaffolds** (profile).
* **Ral lives on the Execution Plane.** Delays in Ral affect the training loop; Urza/Karn/Tezzeret/Urabrask stay **slow-time / async** with no timing guarantees.
* **Slow-time gap:** Tamiyo/Narset can request a precompiled blueprint from Tezzeret/Urza, but to keep training moving, **Ral fills the gap** with fast, region-local micro-compiles.

---

## 1) Planes, roles, and chain of command

### Control Plane (authoritative)

* **Tamiyo (Strategy):** Sets `PolicyEnvelope`, `BudgetEnvelope`; issues `ProtectedZoneGrants`; runs **inverse throttle** for pruning; approves SOPs/TTL extensions.
* **Emrakul (Planner @ checkpoint, under Tamiyo):** Consumes Elesh analysis + budgets; emits **structured compaction plans** and **scaffold exit decisions** (FUSE/PRUNE/PERSIST), honoring Protected Zones.
* **Narset 1..N (Tactics per region):** Detects instability; requests scaffoldable blueprints; manages α schedules & dwell; commands **Kasmina seeds**; reports **StrategicHealth** up.
* **Elesh (Offline analysis workers):** At checkpoint: head/channel/layer vitality, attribution deltas, **foldability** (SVD/EVD), feeds Emrakul.

### Execution Plane (does the work, owns timing)

* **Kasmina (Executor):** Runs the compiled graph; maintains **seed lifecycle** (unchanged); enforces gradient isolation; **loads artifacts and commits swaps**; holds Region/Zone registries; applies fold/prune diffs at checkpoint.
* **Ral (Micro-compiler service):** Compiles small **region kernels** quickly on demand from blueprint IRs / templates; returns **signed artifacts** to Kasmina; no direct graph mutation.

### Slow-time Toolchain (asynchronous; no deadlines)

* **Karn:** Generates blueprint candidates (incl. scaffold profiles).
* **Tezzeret:** Full compiler; emits **paired manifests** (standard/scaffold) with fold descriptors; slow but global AOT quality.
* **Urza:** Artifact store & catalog (standard + Ral outputs).
* **Urabrask:** Safety review & **code signing** (attestation).

### Orchestration & Telemetry

* **Tolaria:** Trainer/loop orchestration (epochs/checkpoints); Kasmina’s “sparring partner”.
* **Oona:** Message bus (commands/events/telemetry).
* **Nissa:** Metrics, traces, dashboards, SLOs.

---

## 2) Seeds, blueprints, scaffolds (taxonomy)

* A **seed** is an execution handle hosted by **Kasmina**, owned by a Narset region, with the standard lifecycle (DORMANT, TRAIN\_ISOLATED, GRAFTING, STABILISATION, ACTIVE/FOSSILISED).
* A **blueprint** is a compiled kernel (from Tezzeret or Ral) the seed can **germinate** into.
* A **scaffold** is **not a new seed**—it’s a **deployment profile** on a blueprint (`supports_scaffold: true`) that adds **TTL**, **success gates**, **weaning**, and **sunset hooks** (Mary-Poppins pattern).

**Invariant:** No lifecycle state is added or changed; scaffolds are **policy overlays** on the existing stages. Gradient isolation invariant holds (`∇L_host ∩ ∇L_seed = ∅`), with controlled joint training only during explicit **weaning** inside the RegionSpec.

---

## 3) End-to-end control & data flows

### 3.1 Nominal “stabilize then compact” loop

1. **Narset** detects local instability (entropy collapse, gradient oscillation) in Region R → **ScaffoldRequest** (blueprint id, TTL, radius, metrics).
2. **Tamiyo** approves; issues **ProtectedZoneGrant** (binding at next checkpoint). **Budgets:** publishes `B_seed`, `B_scaffold` per region; computes `B_prune` via **inverse throttle**.
3. **Kasmina** (for the target seed) asks **Ral** for a fast artifact if Tezzeret/Urza not ready; Ral compiles & signs; uploads to **Urza**.
4. **Kasmina** loads artifact from Urza (Tezzeret if present, else Ral), validates signature/KCC, then **commits the swap** (Kasmina-owned prepare/commit, DDP-barrier).
5. The seed runs **GRAFTING → STABILISATION → WEANING** under Narset’s α schedule and gates.
6. At **checkpoint**, **Elesh** computes vitality & foldability; **Emrakul** issues **ScaffoldExitDecision** and structured compaction plan outside Protected Zones; **Kasmina** applies fold/prune diffs; embargo lifts one checkpoint later if drift ≤ δ.
7. **Tolaria** resumes training; **Nissa** collects all telemetry; **Oona** broadcasts events.

### 3.2 Timing guarantees

* **Kasmina swaps** are micro-batch safe (commit between minibatches), but **Protected Zones** are binding for Emrakul only at **next checkpoint**.
* **Elesh/Emrakul** work **only** at checkpoint; if time budget exceeded, **skip** this cycle (non-blocking).
* **Ral** is best-effort execution service; compiles in parallel; **Kasmina never blocks the step** waiting—falls back to host/no-op and retries swap when artifact arrives.

---

## 4) Contracts (updated for Kasmina-owned swaps)

### 4.1 Strategy & budgets (Tamiyo → Narset/Emrakul)

```protobuf
message PolicyEnvelope { double risk_tolerance; double inverse_beta; double prune_quiet_cap; }
message BudgetEnvelope {
  message RegionBudget { string region_id; uint32 B_seed; uint32 B_scaffold; }
  repeated RegionBudget regions;
  uint32 B_prune_base; uint32 B_prune_cap; uint32 B_prune_min;
}
message ProtectedZoneGrant {
  string grant_id; RegionSpec region;
  string activates_at_ckpt; string lift_after_ckpt;
  bool pruning_embargo; string reason;
}
```

### 4.2 Narset → Kasmina (tactical)

```protobuf
message AdaptationCommand {
  string region_id; string seed_id; string blueprint_id; // Urza id
  bool deploy_as_scaffold; uint32 ttl_ckpts; bool merge_ok;
  map<string,double> alpha_schedule;  // warmup/hold/decay in ckpts
  map<string,double> success_gates;   // jitter, drift, diversity, parity
}
```

### 4.3 Kasmina ↔ Ral (artifact provisioning)

```protobuf
// Kasmina asks for a “fast path” artifact when Tezzeret's isn’t ready.
message RalCompileRequest { string region_id; string blueprint_ir_id; string seed_id; map<string,string> kcc; }
message RalCompileResult  { string artifact_id; bytes sha256; string urza_uri; uint64 workspace_bytes_max; }
```

### 4.4 Checkpoint decisions (Elesh/Emrakul → Kasmina)

```protobuf
message ScaffoldExitDecision {
  string scaffold_id;
  enum Outcome { FUSE=0; PRUNE=1; PERSIST=2; } Outcome outcome;
  double theta_fold; uint32 lora_rank; double delta_metric;
  string reason; // gates_passed | ttl_expired | not_foldable | resource_limit
}
message PruningPlan { repeated StructuredOp ops; /* channels/heads/layers */ }
```

---

## 5) State machines (authoritative)

### 5.1 Kasmina — **Swap & Region Registry** (executor, owns commit)

```
IDLE
 └─ load(artifact from Urza) → VALIDATING(sig,KCC)
VALIDATING
 ├─ fail → ABORT
 └─ ok   → PREPARED
PREPARED
 └─ on minibatch boundary + all ranks ready → COMMITTED (pointer flip)
COMMITTED
 └─ telemetry OK → RUNNING; else → REVERT → IDLE
```

**Notes:**

* Kasmina manages DDP barriers and rank consensus.
* If artifact arrives late, seed runs host/no-op until COMMITTED.
* **Opaque call boundary** is implemented by **Kasmina** (not Ral) to avoid Dynamo retraces.

### 5.2 Narset — regional tactics

```
IDLE → (instability & budget_ok) → REQUEST
REQUEST → (PZ grant ack) → ACTIVE
ACTIVE: GRAFTING→STABILISING→WEANING
  ├─ (gates pass) → SUNSET_READY (await checkpoint)
  ├─ (ttl expires)→ SUNSET_READY
  └─ (thrash)     → COOLDOWN → IDLE
SUNSET_READY (checkpoint):
  ├─ Emrakul=FUSE  → NOOP
  ├─ Emrakul=PRUNE → NOOP
  └─ Emrakul=PERSIST (+TTL ext) → ACTIVE
```

### 5.3 Ral — execution service

```
QUEUED → COMPILING → UPLOAD_URZA → READY
(READY artifact referenced by Kasmina; Ral never touches the model graph)
```

### 5.4 Emrakul/Elesh — checkpoint planner

* **Elesh:** vitality sketches, attribution deltas on small eval slices, foldability SVD with energy target η (default .90).
* **Emrakul:** honors Protected Zones; emits compaction plan; decides scaffold exits by θ (foldability) & Δmetric gates.

---

## 6) Algorithms & safety (grounded)

### 6.1 Isolation & blending

```
out = α·seed_out + (1−α)·host_out.detach()
```

Hooks use weakrefs; removed on phase exit; no `.item()` in hot path. Joint training only in **WEANING** and only inside RegionSpec.

### 6.2 Foldability (θ) & LoRA rank

* **Linear:** `θ_lin = 1 − ||ΔW − Π_linear(ΔW)||_F / ||ΔW||_F`
* **LoRA:** smallest `r` with spectral energy ≥ η; `θ_lora = 1 − √(Σ_{i>r}σ_i²)/√(Σ_iσ_i²)`
* **Final θ:** `max(θ_lin, θ_lora) * (1 − λ_stab·sigmoid(c·Δloss_offline))`
* **Gate:** `θ ≥ 0.85` and `Δmetric ≥ 0` → FUSE allowed; else PRUNE/PERSIST.

### 6.3 Budgets & inverse throttle

```
spend_hat = EMA(B_seed_spend + B_scaffold_forecast, τ=3)
B_prune = clamp(B_base − β·spend_hat, B_min, B_cap)
if any zone ∈ {STABILISING, WEANING}: B_prune = min(B_prune, quiet_cap)
```

---

## 7) PyTorch/Inductor compatibility (Kasmina-owned)

* **Opaque boundary is Kasmina’s op** (e.g., `kasmina.region_call(region_id, inputs, alpha)`).
* α is a **runtime tensor** (buffer/input), **not** a graph constant.
* Fixed **shape-class** guards; no dynamic dispatch on types/strides.
* Seeds swap artifacts by pointer—**no host graph recompilation**.
* Ral produces **AOT kernels** (Triton/CUDA/Inductor) with tight KCC (dtype/stride/shape-class, workspace bounds).

---

## 8) Performance & memory budgets (realistic)

* **Importance sketches (Elesh):** width 8–16k, depth 5–7, uint16 → **\~8–32 MB** @ 100 layers.
* **Swap overhead:** commit between minibatches; amortized < 0.5 ms per swap; failure → **NOOP** fallback.
* **Checkpoint:** per scaffold exit analysis ≤ **30–45 s**; skip cycle if over budget.
* **Expected gains:** **2–10×** regional convergence in the regimes scaffolds target; zero serving latency from temporary capacity after sunset.

---

## 9) Security, provenance, and audit

* All artifacts (Tezzeret & Ral) are **Urabrask-signed**; Kasmina verifies before load.
* Checkpoint metadata records **plan hashes**, **fold diffs** (reversible), **zone state**.
* Oona events: `scaffold.{requested|phase|sunset}`, `zones.{granted|lifted|merged}`, `swap.{prepared|committed|reverted}`, `emrakul.{plan|skipped}`.

---

## 10) SRE runbook (delta)

**Manual controls:**
`zones.freeze(region, epochs)` · `zones.extend(grant, epochs)` · `scaffold.kill(id)` · `embargo.manual(region)` · `swap.abort(region)`

**Dashboards:**
`B_prune_t vs base` · active zones & TTL · scaffolds by phase · swap success/latency · checkpoint time usage · exit outcomes.

**Breakers:**

* Swap breaker: N failures per M mins → auto-NOOP & backoff.
* Checkpoint pressure: if compute > 0.8·T\_ckpt → defer exits and compaction next cycle.

---

## 11) Implementation plan (ownership clarified)

**Sprint 1 — Control & schema**

* Tamiyo: Policy/Budget envelopes; inverse throttle; zone grants/lifts.
* Narset: ScaffoldRequest path, α/TTL policies, SOP bundling.
* Oona/Leyline: `scaffold.*`, `zones.*`, `swap.*` topics; codecs.
* Urza/Karn/Tezzeret: scaffold profile schema; paired manifests; fold hints.

**Sprint 2 — Execution & Ral**

* **Kasmina:** Region/Zone registries; **swap state machine** (prepare/validate/commit/revert); opaque op; NOOP artifact; hook refactors; allocator pool.
* **Ral:** Micro-compile service; KCC; upload to Urza; signing via Urabrask; perf telemetry.
* Wiring: Kasmina calls Ral, not vice-versa.

**Sprint 3 — Checkpoint logic**

* **Elesh:** vitality + foldability (randSVD), bounded eval; parallel dispatcher.
* **Emrakul:** structured compaction (zones-aware), exit decisions, post-sunset prioritization.
* **Kasmina:** fold/prune application at checkpoint; reversible diffs.

**Hardening (2–3 weeks)**
Long-haul fuzz, crisis drills, multi-zone pressure, compile latency chaos; SRE playbooks & dashboards.

---

## 12) CI / verification (must-pass)

1. **Isolation tests:** `∇L_host ∩ ∇L_seed = ∅` in all phases (except explicit weaning), enforced via backward-hook monitors (weakref).
2. **Opaque-op stability:** 1k α changes, 100 swaps → **no Dynamo recompiles**.
3. **Swap fuzz:** racey prepare/commit/revert across DDP ranks → no split-brain, deterministic fallback.
4. **KCC & signature:** invalid artifact rejected deterministically.
5. **Embargo:** proposed ops intersecting `RegionSpec ⊕ radius` → Emrakul emits **zero** entries.
6. **Checkpoint pressure:** ≥3 concurrent TTL expiries finish within budget or cleanly skip.
7. **Fold parity:** after FUSE, parity ε ≤ 0.01; drift ≤ δ in next checkpoint.

---

## 13) Open knobs for Conclave

* Default **zone radius**: 1 block (baseline) vs 2 for MoE/long-skip.
* LoRA energy η: 0.90 default vs 0.95 (conservative).
* **β** (inverse throttle): 0.6 default; tune by active-scaffold count.
* Exit-eval cap per scaffold: 30 s vs 45 s @ `T_ckpt=300 s`.
* Max concurrent scaffolds **per region**: 1 (default) vs 2 with auto-merge.

---

## 14) Worked example with your command chain

**Instability:** entropy collapse in encoder blocks 9–12.

1. **Narset** → `AdaptationCommand(deploy_as_scaffold, ttl=10, radius=1)` on the seed serving 9–12.
2. **Tamiyo** → `ProtectedZoneGrant` (binds next checkpoint) + budgets; **Emrakul**’s `B_prune` inversely throttled.
3. **Kasmina** requests artifact: Tezzeret not ready → **Ral** compiles fast, uploads to **Urza**; Kasmina validates & **commits swap**.
4. α ramp & dwell; **weaning** with KD; gates green.
5. **Checkpoint:** **Elesh** computes θ (0.88) with LoRA rank 8; **Emrakul** → **FUSE**, and compaction plan outside the zone.
6. **Kasmina** applies fold diff, converts to **NOOP**; after one clean checkpoint, **zone lifts**; next cycle compaction (FFN channel) proceeds.

---

### Appendix A — Safe code patterns (Kasmina-owned)

**Blend (always detach host):**

```python
out = alpha * seed_out + (1.0 - alpha) * host_out.detach()
```

**Leak-free backward hook:**

```python
import weakref
class GradLeakGuard:
    def __init__(self, monitor, layer): self._mref=weakref.ref(monitor); self._layer=layer
    def __call__(self, module, grad_input, grad_output):
        m = self._mref(); 
        if not m: return
        outs = grad_output if isinstance(grad_output, tuple) else (grad_output,)
        for g in outs:
            if g is not None and g.is_cuda: m.observe(self._layer, g)
hook = module.register_full_backward_hook(GradLeakGuard(self, name))
# on phase exit:
hook.remove()
```

**Kasmina opaque call (illustrative signature):**

```cpp
// kasmina.region_call: owned by Kasmina; takes alpha as runtime tensor.
// Prevents Dynamo retrace while allowing artifact pointer swap.
std::vector<Tensor> kasmina_region_call(
  uint64_t region_id,
  std::vector<Tensor> inputs,
  Tensor alpha_runtime,
  KasminaCtx* ctx   // non-traced, holds artifact pointers & KCC
);
```

---

If you want, I can split this into:

* `14-narset-unified-design.md` (tactical controller spec),
* `02.6-kasmina-region-registry-and-swaps.md` (executor swap ABI + state machine),
* `15-ral-micro-compiler.md` (service API & KCC),
* plus a tiny `00-leyline-scaffolds.proto`.

Say the word and I’ll draft them in your house style.
