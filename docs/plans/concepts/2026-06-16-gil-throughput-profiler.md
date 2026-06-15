# Plan Metadata

```yaml
# Plan Metadata
id: gil-throughput-profiler
title: Tiered GIL / Throughput Profiler for the Vectorized PPO Runtime
type: concept
created: 2026-06-16
updated: 2026-06-16
owner: john@pgpl.net

# Prioritization
urgency: high
value: >
  An instrument + methodology that PROVES whether the vectorized PPO runtime is
  bottlenecked by a single Python dispatch thread (and specifically by GIL
  contention), and ATTRIBUTES wall-clock to the 6 transaction phases so the team
  can target Python->Rust migrations with evidence instead of intuition.

# Constraints
complexity: L
risk: medium
risk_notes: >
  The hot path is determinism-critical (seeded make_env_seed ordering, replay
  verification gate) and sync-disciplined (gpu_sync_whitelist.yaml + lint). A
  naive event-readback or a GIL-sampler thread can (a) insert an un-whitelisted
  host<->device sync, (b) perturb seeded thread interleaving, or (c) measure a
  false GPU-busy window. Each is a hard determinism/correctness hazard and is
  explicitly fenced below. The instrument is opt-in and tiered so the always-on
  layer carries near-zero risk.

# Dependencies
depends_on: []
soft_depends:
  - vectorized-ppo-runtime-transaction-phases   # docs/plans/2026-06-15-...; phase names align
blocks: []

# Status
status_notes: >
  Concept. Tier 0 is buildable immediately against existing seams. Tier 1/2 are
  opt-in and gated. Characterization (Phase A) must run BEFORE any Rust migration
  decision. Awaiting specialist sign-off (see reviewed_by) before promotion to ready.
percent_complete: 0

# Expert Review
reviewed_by:
  - reviewer: pytorch-expert
    scope: CUDA-event occupancy windows, stream-context placement, torch.profiler/NVTX overhead
    status: design-input-folded   # adversarial verdict folded; sign-off pending on built artifact
  - reviewer: determinism-reviewer
    scope: no-new-sync guarantee, seeded-ordering safety, timing-as-external-effect, bit-identical-when-disabled
    status: design-input-folded   # consolidated determinism review folded as Rules 1-6
  - reviewer: drl-expert
    scope: phase boundaries match PPO rollout/update semantics; reward/governor as migration targets
    status: pending
```

---

## Goal

Prove or disprove, with evidence, the user's claim: *"we hit the GIL limit before
maxing out training hardware — one Python thread can't dispatch fast enough to keep
N independent host models (each on its own CUDA stream) saturated."*

The instrument must separate five quantities, attributed to the **6 transaction
phases** (rollout / action / reward / rollback / telemetry / ppo-update) defined in
`docs/plans/2026-06-15-vectorized-ppo-runtime-transaction-phases.md`:

- **(a) GPU genuinely busy** — device-clock kernel-active time per env stream.
- **(b) GPU idle, starved by slow Python dispatch** — phase wall-clock minus the
  union of stream-busy windows, while VRAM/SM have headroom.
- **(c) GIL HOLD time per phase** — Python CPU time the dispatch thread spent holding
  the GIL inside a phase (the work that starves dispatch).
- **(d) GIL WAIT / contention time per phase** — time a runnable thread was blocked
  waiting for the GIL held by another thread (dispatch vs nissa encode thread).
- **(e) per-phase Python CPU time** — the raw thread-CPU cost of each phase.

It must be **low-overhead, toggleable, and determinism-safe**: bit-identical to a
baseline run when disabled, and never adding a CPU<->GPU sync or perturbing seeded
ordering when enabled.

### What survived critique (kept)

- **Two-independent-clocks methodology** (device-clock CUDA-event busy vs host
  `perf_counter` wall over the same phase window, compared only after streams drain).
- **Attribution to the 6 transaction phases** via `record_function`/probe names so
  profiler output and refactor units align.
- **Report BOTH** sum-of-busy (per-stream saturation) **and** interval-union (true
  GPU-occupied wall); **union-vs-phase-wall is the single GPU-starvation metric**.
- **Karn DuckDB `phase_occupancy` view** keyed on `run_dir`/`epoch`, emitted via the
  existing nissa background queue so encoding stays off the hot path.
- **Tiered toggles** compiling to `None`/`nullcontext` when off; reuse the existing
  `training_profiler` gate at `vectorized_trainer.py:837-848`.
- **GIL HOLD/WAIT decomposition** via `time.thread_time_ns()` vs `perf_counter_ns()`,
  cross-validated against a direct GIL-state sampler.

### What was killed or corrected

- **"This proves the GIL ceiling" — corrected.** A `wall >> busy` gap proves *the
  single dispatch thread is too slow*, which has multiple causes (GIL, pure serial
  compute, allocator stalls). The GIL-specific claim requires GIL-state instrumentation
  (Tier 2). Tier 0/1 are honestly framed as a **single-thread dispatch-starvation
  profiler**; Tier 2 supplies the GIL-attribution leg.
- **Train-phase event bracket at `vectorized_trainer.py:1138` — corrected (was a
  correctness bug).** That `stream_ctx` wraps only the trivial `train_loss_accum.add_()`
  / `train_correct_accum.add_()` ops (verified lines 1138-1140); the heavy
  `process_train_batch` already enqueued earlier inside its OWN
  `torch.cuda.stream(env_state.stream)` context at `batch_ops.py:126-130`. The bracket
  MUST live inside `batch_ops.py:126-130` or the train "busy" is falsely near-zero.
- **`sys.monitoring` (PEP 669) — NOT available on the current runtime (corrected).**
  An earlier draft of this plan asserted the runtime was Python 3.12.3 and made
  `sys.monitoring` the preferred Tier-2 backend. That was wrong: `uv run python
  --version` (the authoritative project interpreter per CLAUDE.md) is **3.11.11**, and
  `pyproject.toml` requires only `>=3.11`. PEP 669 `sys.monitoring` is 3.12+, so it is
  **unavailable here**. Tier-2 GIL attribution therefore relies on **in-process
  HOLD/WAIT inference + out-of-band `py-spy --gil`** (see Tier 2). `sys.monitoring`
  is retained as a *conditional backend* that unlocks on 3.12+. NOTE (2026-06-15): an
  interpreter upgrade to 3.13/3.14 is now authorized — see "Interpreter upgrade &
  free-threading"; that both unlocks `sys.monitoring` AND puts the GIL-removing
  free-threaded build on the table as remediation (A).
- **`compiled_train_step` intra-graph time — kept only as a region boundary.** An
  Inductor graph fuses host fwd/bwd into few launches; a `record_function` cannot
  decompose intra-graph Python time. Do not infer Python attribution inside the graph.

---

## Architecture

A single new module, `src/esper/simic/telemetry/phase_profiler.py`, owns the whole
instrument. It is entered as a sibling context manager next to `training_profiler` at
`vectorized_trainer.py:837-848` and yields a `PhaseProfiler` handle (or a `NullProfiler`
no-op when all tiers are off, mirroring `profiler.py:69-71` which already yields `None`).

The handle exposes:

- `phase(name)` — a context manager wrapping a transaction-phase span. Records
  `perf_counter_ns` (wall), `thread_time_ns` (Python CPU = quantity e), and pushes a
  `record_function` row (Tier 1) and NVTX range (Tier 1, nsys-gated).
- `env_dispatch(env_idx, stream)` — a context manager (Tier 1) bracketing a per-env GPU
  dispatch with a pooled, paired `torch.cuda.Event(enable_timing=True)`. `try/finally`
  guarantees the stop event always records (Rule 3).
- `drain()` — called at exactly ONE place (the epoch-end synchronize loop,
  `vectorized_trainer.py:1144-1147`) where all env streams are already drained. Reads
  every event pair's `elapsed_time()` (safe only here — Rule 1), computes per-phase
  `busy_ms` (sum + union), `wall_ms`, `gap_ms`, `hold_ms`, `wait_ms`, and copies
  thread-local accumulators (Rule 6). Emits one typed payload to the nissa queue.

State that crosses threads (dispatch thread vs nissa encode thread) is **thread-local
and partitioned by thread id** (Rule 6); aggregation is copy-not-share at `drain()`.

A new leyline dataclass `PhaseProfileReport` (per the CLAUDE.md leyline rule) carries the
per-phase numbers and the process-level GIL-held fraction; it is **observation-only**
(Rule 4) and flows one-way to nissa.

```
training_profiler(...) ── existing torch.profiler gate (vectorized_trainer.py:837-848)
phase_profiler(tier0, tier1, tier2) ── NEW sibling ctx (same try/finally lifetime)
   │
   ├─ Tier 0 (always-on, cheap): perf_counter_ns + thread_time_ns per phase
   │       -> per-phase wall_ms, python_cpu_ms; GPU-starvation ratio at drain()
   ├─ Tier 1 (opt-in): + record_function rows, NVTX ranges, per-env CUDA-event
   │       brackets inside existing stream contexts; torch.profiler timeline
   └─ Tier 2 (opt-in, diagnostic-only): in-process GIL HOLD/WAIT inference +
           out-of-band py-spy --gil attach; gil_load fallback; sys.monitoring iff 3.12+
   │
   └─ drain() at vectorized_trainer.py:1144-1147 ── PhaseProfileReport -> nissa queue
                                                  ── Karn DuckDB view `phase_occupancy`
```

### The six transaction-phase seams (verified file:line)

| Phase | Span | Entry seam (verified) |
|-------|------|------------------------|
| rollout | obs/mask/policy `get_action` | `vectorized_trainer.py:1820-2057` |
| action | `execute_actions()` (serial per-env loop) | call site `vectorized_trainer.py:2059`; loop body `action_execution.py:545-1634` |
| reward | scalar `compute_reward` block | `action_execution.py:848-1082` (call at `:952`) |
| rollback | governor rollback in stream | `action_execution.py:625-654` |
| telemetry | episode finalize / Shapley | `action_execution.py:1554-1633` |
| ppo-update | handle_rollbacks + run_update | `vectorized_trainer.py:2185-2239` |
| (bootstrap 2.5) | bootstrap value compute | `vectorized_trainer.py:2117-2167` |

The existing throughput anchors reconcile against the profiler: `epoch_start` is the same
anchor used at `vectorized_trainer.py:2169-2172` (`throughput_step_time_ms_sum`,
`throughput_dataloader_wait_ms_sum`), verified.

---

## The single GPU-starvation metric (chosen)

> **`gpu_starvation_ratio[phase] = 1 - (busy_union_ms[phase] / wall_ms[phase])`**, reported
> only alongside concurrently-sampled SM-occupancy and allocator-reserved headroom.

- `busy_union_ms` is the **interval-union** of per-env-stream CUDA-event windows in the
  phase (NOT the sum), so concurrent streams are not double-counted (Stream-Truth risk
  #1, accepted as blocking-fix). `busy_sum_ms` is *also* reported as per-stream saturation,
  but the **union** is the authoritative number that drives migration decisions (resolves
  the gate's "two numbers that disagree" finding — union wins).
- A high ratio (GPU idle) is only evidence of starvation **if headroom exists**. The
  premise is therefore captured as data, not asserted: at `drain()` sample
  `torch.cuda.utilization()` (ASSUMPTION: NVML available; falls back to Nsight GPU-metrics
  in Tier 1) and `torch.cuda.memory_stats()` reserved/allocated. A phase with
  `gpu_starvation_ratio` near 1.0 **and** SM/VRAM headroom **and** (Tier 2) GIL-held
  fraction near 1.0 is category (b): GPU starved by Python dispatch.
- **Tiebreak rule:** when the union window for a stream is suspected inflated by
  intra-stream inter-kernel gaps (Stream-Truth risk #2, the metric is least trustworthy
  exactly in the starved regime), the Tier-1 `torch.profiler` per-kernel CUDA activity
  times are authoritative over the coarse event-pair window. The event window is the
  cheap always-available estimate; per-kernel torch.profiler is the validator.

Tier 0 cannot record CUDA events (it adds no GPU work). At Tier 0 the starvation signal is
the coarser **`python_cpu_ms[phase] / wall_ms[phase]`** ratio: a phase whose wall is almost
entirely Python CPU (e.g. reward, which is pure scalar Python in `contribution.py`, verified
0 torch refs) has no GPU work to overlap and is a single-thread hold. Tier 1 upgrades this to
the true `gpu_starvation_ratio`.

---

## Tier 0 — always-on cheap per-phase wall-clock + starvation ratio

**Toggle:** on by default; disabled with `--no-phase-profiler` (single bool; default-on
because cost is sub-0.1%/epoch). Compiles to `NullProfiler` when off.

**What it records (no GPU, no extra thread, no sync):**
- Per-phase `wall_ms` via `time.perf_counter_ns()` deltas at the 6 phase boundaries,
  anchored to the SAME `epoch_start` as the existing throughput counters
  (`vectorized_trainer.py:2169-2172`) so the numbers reconcile.
- Per-phase `python_cpu_ms` via `time.thread_time_ns()` deltas (= quantity (e),
  per-phase Python CPU time; `thread_time_ns` is 3.7+, safe on the 3.11.11 runtime, no extra thread).
- Per-phase `python_cpu_ratio = python_cpu_ms / wall_ms` — the Tier-0 starvation proxy.

**Integration points (verified file:line):**
- `vectorized_trainer.py:837-848` — enter `phase_profiler(...)` beside `profiler_cm`,
  same `try/finally` lifetime; bind handle.
- `vectorized_trainer.py:1820` — `with profiler.phase("rollout"):` around rollout span
  (through 2057).
- `vectorized_trainer.py:2059` — `with profiler.phase("action"):` around the
  `execute_actions()` call (the whole serial loop is one attributed span).
- `vectorized_trainer.py:2117-2167` — `with profiler.phase("bootstrap"):`.
- `vectorized_trainer.py:2185-2239` — `with profiler.phase("ppo_update"):`.
- `action_execution.py:848-1082` — `with profiler.phase("reward"):` around the scalar
  reward block (call at `:952`). Phase-granularity only; NOT per-env (Rule 2).
- `action_execution.py:625-654` — `with profiler.phase("rollback"):` around the governor
  rollback branch.
- `action_execution.py:1554-1633` — `with profiler.phase("telemetry"):` around episode
  finalize / Shapley.
- `vectorized_trainer.py:1144-1147` — `profiler.drain()` immediately AFTER the existing
  per-env `stream.synchronize()` loop (streams already drained; Rule 1).

**Overhead:** ~7-8 phase boundaries/epoch x 2 clock reads + 1 dict accumulate ≈
sub-microsecond each; <50 us/epoch total, i.e. well under 0.1% against ms-to-second epochs.
No allocation in the hot loop (accumulator dict pre-created per epoch). Determinism-safe by
construction: touches no tensors, issues no CUDA call, adds no sync. This is why it is
default-on.

---

## Tier 1 — opt-in torch.profiler + CUDA-event occupancy + NVTX

**Toggle:** two orthogonal flags, both default-off:
- `--phase-profiler-occupancy` — gates `record_function` rows + per-env CUDA-event
  brackets (the `gpu_starvation_ratio`). Independent of torch.profiler because event-pair
  accounting is far cheaper and can run for many epochs.
- reuse existing `--torch-profiler` (already plumbed into `training_profiler` at
  `vectorized_trainer.py:837-848`) for the full CPU/CUDA timeline.
- NVTX emission auto-gated by an `ESPER_NSYS` env (or detect no current stream capture);
  NVTX is near-free when nsys is not attached.

**What it adds:**
- `record_function("phase.<name>")` rows for the 6 phases (CPU-timeline; ~7 calls/epoch).
  **Phase granularity only** — NO `record_function` inside the per-env serial loop
  (`action_execution.py:545`); per-env detail is NVTX (Rule 2), itself nsys-gated.
- Per-env paired `torch.cuda.Event(enable_timing=True)`, pooled per env across epochs
  (mirroring the P2-STREAMPOOL stream-reuse discipline at `vectorized.py:1357-1362`), at the
  TWO real GPU dispatch sites:
  - **Train fwd/bwd:** inside `process_train_batch`'s OWN stream context at
    `batch_ops.py:126-130` (CORRECTED — NOT `vectorized_trainer.py:1138`, which wraps only
    the trivial `add_` accumulators; verified lines 1138-1140).
  - **Lifecycle dispatch:** inside `lifecycle_ctx = torch.cuda.stream(env_state.stream)` at
    `action_execution.py:1086-1089`.
- Concurrent SM-occupancy + allocator sampling at `drain()` (`torch.cuda.utilization()`,
  `torch.cuda.memory_stats()`) so the "headroom exists" premise is data.
- NVTX ranges via `with torch.cuda.nvtx.range(...)` context managers ONLY (never bare
  push/pop — Rule 3), per-env `env{idx}` ranges nested in `phase.action` for the Nsight
  timeline.

**The CUDA-event readback discipline (Rule 1 — load-bearing):**
- `event.record(stream)` is a pure async enqueue: it does NOT drain the stream, so kernel
  ordering and seeded `make_env_seed` ordering (`vectorized_trainer.py:885-890`) are
  untouched.
- `event.elapsed_time()` is **only** called at `drain()` (`vectorized_trainer.py:1144-1147`)
  where the per-env synchronize loop has already drained every stream. It is NEVER called
  after `execute_actions()` returns (`:2059`) — there is no synchronize between `:2059` and
  bootstrap `:2117`, so reading there would either raise (event not complete) or force a NEW
  synchronize that perturbs seeded ordering. Lifecycle event pairs recorded at
  `action_execution.py:1086-1089` are therefore drained-and-read at the epoch-end barrier,
  not inline.
- Event-pair parity is asserted at `drain()`; a start with no matching stop (e.g. the panic
  `continue` at `action_execution.py:711`) fails closed and discards that env's window
  rather than reading a mispaired pair across a drain boundary (Rule 3).

**Overhead:** `record_function` ~1-3 us/call x ~7/epoch = negligible. `event.record()` ~1-2
us host async enqueue; 2 events x N_envs x 2 sites ≈ sub-50 us host/epoch (N=8). NVTX near-free
when nsys off. **Caveat (the Heisenberg axis):** the per-env event records land in the very
serial dispatch loop whose throughput is the measurement target; under the dispatch-starved
hypothesis even small per-iteration host work compounds across N x steps. This is quantified,
not assumed — see Validation Phase A2 (occupancy-on vs occupancy-off A/B at fixed seed; report
the wall delta as part of the instrument's own self-validation).

---

## Tier 2 — opt-in GIL hold/wait attribution (diagnostic-only)

This is the leg that actually measures the **GIL** (quantities (c) and (d)), as opposed to
Tier 0/1 which prove single-thread dispatch-starvation. Tier 2 is **diagnostic-only** and
**mutually exclusive with replay-verification / `--proof-baseline` runs** (Rule 5).

**Toggle:** `--phase-profiler-gil` (default off; lazy-imports its backend only in the enabled
branch so there is zero import cost when off).

**Backends, in preference order (current runtime is Python 3.11.11):**
1. **In-process HOLD/WAIT inference — PREFERRED (always available, no extra thread).** per-phase
   `hold_ms = thread_cpu_delta` (Python CPU advancing while holding the GIL);
   `wait_ms = wall_delta - thread_cpu_delta - known_sync_ms`, where `known_sync_ms` is
   subtracted from the existing sync inventory (`gpu_sync_whitelist.yaml`) so a genuine CUDA
   kernel-wait is not miscounted as GIL-wait. Residual unexplained WAIT is flagged, never
   silently bucketed. This is inference (proves Python-CPU-boundness), not direct GIL state —
   so it is cross-validated against backend 2 (the out-of-band witness).
2. **`py-spy --gil` (out-of-band, zero in-process overhead):** external-process attach for a
   one-off confirmation pass / cross-check; not phase-aware. The independent witness that the
   in-process inference number is not a tool artifact. This is the trustworthy direct-GIL
   reading on 3.11.
3. **`gil_load` (fallback, build-risk):** whole-process and per-thread GIL-held fraction via a
   sampler thread. FLAGGED: unmaintained, pokes private CPython internals, not confirmed to
   install in the uv env (`gil-load` reported "Package(s) not found"). Its sampler thread takes
   the GIL at 1-10 ms intervals and *nudges* timing — so a run with it ON is NOT bit-identical
   and must never run on a replay/proof-baseline run (Rule 5). Use only if backend 1 is
   unavailable and an in-process whole-process number is required.
4. **`sys.monitoring` (PEP 669) — CONDITIONAL FUTURE (3.12+ only; UNAVAILABLE on the current
   3.11.11 runtime).** The lowest-overhead in-process per-thread option with no sampler thread,
   but it does not exist before 3.12. Adopt only if/when the training box is upgraded; until
   then backends 1+2 are the path.

**nissa encode-thread attribution:** add a `thread_time_ns()` delta around the existing
`_payload_to_dict` encode span at `output.py:259-268` (alongside the existing `time.time()`
span) to measure GIL HOLD on the background encode thread. The accumulator is **thread-local,
partitioned by thread id** — no shared mutable dict written by two threads (Rule 6); aggregation
is copy-not-share at the epoch-end drain. This is what proves dispatch-thread WAIT spikes
correlate with nissa-thread HOLD bursts — the contention witness.

**Overhead:** in-process inference is two extra clock reads per phase boundary (sub-us). A
`gil_load` sampler is ~0.1-1% wall (keep poll >=5 ms). `sys.monitoring` cost depends on the
event set subscribed; subscribe minimally. `py-spy` is external (zero in-process). All of Tier 2
is no-op when disabled.

---

## Determinism safety

Class reviewed against (inferred, since no `01-` artifact exists): **logical-equivalence under
fixed seed + bit-identical-when-disabled**, with a replay obligation
(`docs/plans/completed/03-verify-vectorized-determinism.md`). The instrument is determinism-safe
by construction when disabled, and the six rules below are MANDATORY for the enabled path.

- **Rule 1 — one readback site.** ALL CUDA-event `elapsed_time()` reads happen at exactly one
  place: the epoch-end synchronize loop at `vectorized_trainer.py:1144-1147`, where every env
  stream is already drained. No readback site may introduce a `synchronize()`. Reading after
  `execute_actions()` returns (`:2059`) is FORBIDDEN — no drain exists there (next sync is the
  epoch-end loop), so it would force a new sync and perturb seeded ordering. *(Resolves CRITICAL
  #1.)*
- **Rule 2 — no `record_function` in the per-env loop.** Phase-granularity `record_function`
  only (~7/epoch). Per-env detail uses NVTX, itself nsys-gated. Per-env `record_function` would
  add N allocations/epoch competing for the GIL — contaminating the GIL measurement. *(Resolves
  MEDIUM allocation-churn finding.)*
- **Rule 3 — exception-safe instrument lifecycle.** Every CUDA-event bracket and NVTX range is a
  `try/finally` context manager that always records its stop / pops its range, even on the panic
  `continue` (`action_execution.py:711`) or governor rollback (`:625-654`). Event-pair parity is
  asserted at `drain()`; on mismatch the env's window is discarded (fail-closed), never read as a
  mispaired pair across a drain boundary. *(Resolves HIGH event-leak + MEDIUM NVTX-LIFO findings.)*
- **Rule 4 — timing is observation-only.** All profiler timing floats (`*_ms`, `*_ratio`,
  GIL-held fraction) flow ONE-WAY to nissa. They MUST NOT be read by any deterministic-spine
  component (reward `rewards/`, `tolaria/governor.py`, scheduler, PPO update) and MUST NOT be
  serialized into snapshot bytes or any divergence hash. A lint/test asserts no import of the
  profiler accumulator types from `rewards/`, `tolaria/governor.py`, or the snapshot/encoding
  path. *(Resolves HIGH clock-as-divergence-channel finding.)*
- **Rule 5 — diagnostic observer threads are forbidden on replay runs.** The Tier-2 `gil_load`
  sampler (and any backend that nudges timing) is mutually exclusive with `--proof-baseline` and
  replay-verification modes — assert-fail if both are set. Prefer the no-extra-thread
  in-process inference layer (backend 1) as the always-eligible Tier-2 backend. Before
  adopting any sampler, audit the dispatch path (`contribution.py`, `governor.py`) for any
  process-global `random.`/`np.random.` use — if found, that is a pre-existing isolation bug the
  profiler would expose, not cause. *(Resolves CRITICAL #2.)*
- **Rule 6 — thread-local accumulators.** Cross-thread state (dispatch thread vs nissa encode
  thread) is thread-local and partitioned by thread id; cross-thread aggregation happens only at
  the single drain barrier, copy-not-share. No shared mutable dict written by two threads (also
  future-proofs against free-threaded builds). *(Resolves HIGH nissa-thread-contention finding.)*

**gpu_sync_whitelist.yaml / lint:** the instrument adds NO `.item()`/`.cpu()`/`.tolist()`/
`.synchronize()` calls. `event.elapsed_time()` reads a host-side float timer, NOT a tensor->CPU
transfer, so it is not in scope for the whitelist's keys and `scripts/lint_gpu_sync.py` needs no
new entries. ASSUMPTION: the lint does not itself flag `elapsed_time` or `Event` — verify during
build; if it does, add a documented whitelist entry rather than suppressing.

**Bit-identical-when-disabled:** with all tiers off the handle is `NullProfiler`, every call site
is a no-op context manager, no backend is imported, no event allocated, no NVTX symbol emitted.
The run is byte-identical to baseline; the replay-hash gate is unperturbed.

---

## Overhead budget

| Tier | State | Overhead/epoch | Notes |
|------|-------|----------------|-------|
| all | disabled | 0 | NullProfiler, no import, no alloc, no sync — bit-identical |
| 0 | on (default) | < 50 us (< 0.1%) | ~8 phase boundaries x 2 clock reads + dict accumulate; no GPU, no thread |
| 1 | occupancy on | < 100 us | + ~7 `record_function` + 2 events x N_env x 2 sites async enqueue + 1 NVML sample at drain |
| 1 | torch.profiler on | schedule-scoped | use existing wait/warmup/active/repeat schedule; trace only the active window |
| 1 | nsys attached | nsys-managed | scope with `cudaProfilerStart/Stop` (Tier-1 deliverable) to a few epochs or .nsys-rep is unmanageable |
| 2 | in-process GIL | < 20 us | 2 extra clock reads/boundary; thread-local |
| 2 | gil_load sampler | 0.1-1% wall | poll >=5 ms; diagnostic-only, never on replay (Rule 5) |
| 2 | py-spy --gil | 0 in-process | external attach |

The Heisenberg caveat (Tier-1 event records land on the suspect dispatch thread) is measured, not
assumed: Validation Phase A2 reports the occupancy-on vs occupancy-off wall delta at fixed seed as
part of the instrument's self-validation.

---

## Validation plan

- **V1 — bit-identical-when-disabled (gating).** Run the existing replay-verification flow
  (`docs/plans/completed/03-verify-vectorized-determinism.md`) with the profiler fully disabled;
  assert the replay hash is unchanged vs the pre-instrument baseline. This is the determinism
  gate and MUST pass before merge.
- **V2 — lint clean.** `scripts/lint_gpu_sync.py` passes with no new whitelist entries (the
  instrument adds no tensor-sync). If `elapsed_time`/`Event` trips the lint, add a documented
  entry; do not suppress.
- **V3 — Rule-4 isolation test.** A unit test asserts no import of the profiler accumulator
  types from `rewards/`, `tolaria/governor.py`, or the snapshot/encoding path (timing-as-external-
  effect fence).
- **V4 — Rule-3 parity test.** A unit test drives an `env_dispatch` bracket through a simulated
  panic `continue` and asserts the stop event still records and `drain()` discards the orphaned
  window rather than raising or reading a mispaired pair.
- **V5 — reconciliation.** Assert per-phase `wall_ms` summed over the 6 phases reconciles with
  the existing `throughput_step_time_ms_sum` (`vectorized_trainer.py:2169-2172`) within tolerance
  (same `epoch_start` anchor).
- **V6 — cross-validation (Tier 1/2).** Validate coarse CUDA-event union windows against
  torch.profiler per-kernel CUDA activity times (tiebreak rule); validate in-process GIL
  inference against a one-off `py-spy --gil` attach on a fixed-seed batch (and `sys.monitoring`
  once on 3.12+)
  (guards against tool-specific artifact).

---

## Interpreter upgrade & free-threading — a first-class remediation lever

> Added 2026-06-15: the interpreter is authorized to move to Python 3.13 or 3.14 if it
> yields performance wins. For a GIL-bound workload this reframes the profiler: it is the
> **decision instrument between three remediations**, not merely a Rust-targeting tool.

**(A) Free-threaded CPython (3.13t / 3.14t — PEP 703/779).** Removes the GIL outright —
the most direct attack on the measured ceiling, and the lowest code-churn *if it works*
(no language boundary). The runtime is already shaped for it: N independent host models,
each on its own CUDA stream (`env_state.stream`), so a thread-per-env (or worker-pool)
dispatch maps onto the existing stream-per-env design, letting the serial per-env loop
(`action_execution.py:545`) and the scalar reward/governor/telemetry work run truly
parallel across cores. 3.14's free-threaded build is officially supported (less
experimental than 3.13t) with materially lower single-thread overhead.
MUST validate in a spike: (1) a free-threaded torch wheel exists for torch 2.9.x + cu128
(cp313t/cp314t) — ASSUMPTION, confirm; (2) thread-safety of every shared object crossed by
multiple env threads — the policy, `simic/agent/rollout_buffer.py`, obs/reward normalizers,
the nissa `output.py` queue — each needs a lock or per-thread partition; (3) torch.compile
behaviour under free-threading; (4) that the FT single-thread tax is repaid by real
parallel speedup (this profiler's occupancy A/B measures it directly).

**(B) Standard 3.13/3.14 (GIL retained).** Faster adaptive interpreter, (3.14) JIT
maturity, and it unlocks `sys.monitoring` (PEP 669) as the lowest-overhead Tier-2 GIL
backend. Modest; does NOT remove the ceiling. Take it as a free side-effect of any
upgrade, not as a fix on its own.

**(C) Rust offload of scalar hotspots (reward/governor/telemetry).** A PyO3 crate that
releases the GIL during heavy scalar work — surgical, lower blast radius than an
interpreter swap, and complementary to (A) (cuts residual contention even on a
free-threaded build).

**Recommended order, once this profiler lands the evidence:** spike (A) first (it is
authorized and may eliminate the problem with no language boundary), measure it with this
instrument, then apply (C) to whatever residual scalar work does not parallelize cleanly.
(B) rides along with any upgrade. The profiler must be able to attribute the *same* per-phase
numbers on a GIL build, a free-threaded build, and a Rust-offloaded build so the three are
compared on one axis.

## Phased task sequence (characterization first)

**Phase A — Characterization (build Tier 0, then Tier 1; PROVE or DISPROVE before any Rust).**
- A1. Implement `src/esper/simic/telemetry/phase_profiler.py` (PhaseProfiler + NullProfiler),
  the `phase_profiler(...)` ctx in `profiler.py`, the leyline `PhaseProfileReport` dataclass, and
  the nissa payload + Karn `phase_occupancy` view. Wire Tier 0 at the 8 verified phase seams and
  `drain()` at `vectorized_trainer.py:1144-1147`. Land V1, V2, V3, V5 tests.
- A2. Add Tier 1 (occupancy + record_function + NVTX), CUDA-event brackets at the TWO corrected
  sites (`batch_ops.py:126-130`, `action_execution.py:1086-1089`), SM/VRAM headroom sampling.
  Land V4, V6, and the occupancy-on/off Heisenberg A/B.
- A3. Run a fixed-seed characterization batch with Tier 0+1 on. Produce the Karn
  `phase_occupancy` ranking: phases by `gpu_starvation_ratio` (idle GPU under headroom) and by
  `python_cpu_ratio` near 1.0 (pure single-thread hold). Expected top suspects per the grounding:
  reward (`contribution.py`, 0 torch refs) and governor (`governor.py`, scalar). This is the
  evidence artifact.

**Phase B — GIL attribution (Tier 2; only if Phase A shows a `wall >> busy` gap under headroom).**
- B1. Add the in-process HOLD/WAIT inference layer + nissa encode-thread thread-local
  attribution (Rule 6). (`sys.monitoring` backend is deferred until/unless the box moves to 3.12+.)
- B2. Confirm the GIL ceiling: pair `gpu_starvation_ratio` near 1.0 (Phase A) with GIL-held
  fraction near 1.0 (Tier 2) to land category (b). Cross-validate with a `py-spy --gil` one-off
  (V6) — the trustworthy direct-GIL witness on 3.11. Only after this is the claim "GIL ceiling"
  earned, not merely "single thread too slow".

**Phase C — Migration prioritization (the deliverable's purpose).**
- C1. From the Karn view, rank phases by GIL-attributable HOLD (Tier 2) where available, falling
  back to `python_cpu_ms` (Tier 0). Produce the Python->Rust migration priority list, cross-run
  via `run_dir`/`epoch`, so each migration can be measured for whether it moved the needle.
- C2. (Future, aligns with the transaction-phase refactor) re-key `gpu_sync_whitelist.yaml` if
  `run()` is decomposed; the profiler attributes to the same 6 phases the refactor moves.

**Phase D — Free-threading spike (only if Phase B confirms a GIL ceiling; remediation (A)).**
- D1. Feasibility gate (no code change): confirm a free-threaded torch wheel for 2.9.x + cu128
  (cp313t and/or cp314t) installs in a throwaway uv env; if absent, (A) is blocked and the
  decision falls to (C) Rust offload. Record the FT single-thread overhead vs the GIL build on a
  fixed-seed micro-batch.
- D2. Thread-safety audit of shared state crossed by multiple env threads: policy module,
  `simic/agent/rollout_buffer.py`, obs/reward normalizers, nissa `output.py` queue. Catalogue each
  as already-safe / needs-lock / needs-per-thread-partition. (No mutation; audit + plan only.)
- D3. Restructure the per-env dispatch (`action_execution.py:545`) from a serial loop into a
  thread-per-env / worker-pool on the free-threaded build, behind a flag. Re-run the profiler:
  the win is real only if `gpu_starvation_ratio` drops AND wall-clock improves past the
  single-thread tax. This profiler is the acceptance gate for the experiment.

---

## Open ASSUMPTIONS to confirm at build time

- Current interpreter is Python **3.11.11** (verified via `uv run python --version`;
  `pyproject.toml` requires `>=3.11`). This means `sys.monitoring` (PEP 669, 3.12+) is
  UNAVAILABLE; Tier-2 GIL attribution uses in-process inference + `py-spy --gil`. An upgrade
  to 3.13/3.14 is authorized (2026-06-15): on 3.12+ `sys.monitoring` unlocks, and the
  free-threaded 3.13t/3.14t build is remediation (A) — gated on a free-threaded torch wheel
  for 2.9.x+cu128 (cp313t/cp314t), confirm in the Phase D spike.
- torch is pinned `>=2.8.0` (installed **2.9.1+cu128**, CUDA 12.8). Free-threaded torch wheel
  availability for this version is the gating unknown for remediation (A).
- `torch.cuda.utilization()` / NVML is available on the training box for SM-occupancy headroom
  sampling; otherwise Tier-1 falls back to Nsight GPU-metrics.
- `scripts/lint_gpu_sync.py` does not flag `Event`/`elapsed_time` (it scans tensor-sync calls);
  verify, and add a documented entry if it does.
- `gil_load` does not install cleanly in the uv env (`gil-load` not found at design time) — Tier 2
  relies on `sys.monitoring` + in-process inference + `py-spy`, not `gil_load`.
- The exact `process_train_batch` enqueue line within `batch_ops.py:126-130` is the stream-context
  open at `:130`; the heavy fwd/bwd is enqueued inside it (the event bracket goes here, NOT at
  `vectorized_trainer.py:1138`). Verified that `:1138` wraps only `add_` accumulators.
