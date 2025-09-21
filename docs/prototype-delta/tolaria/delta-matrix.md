# Tolaria — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Epoch lifecycle & budget | `01.1-tolaria-epoch-lifecycle.md` | End‑of‑epoch work ≤18 ms; enforce guard and conservative mode on breach | `src/esper/tolaria/trainer.py` (budget knobs + breaker + conservative fallback) | Implemented | Must‑have | Budgets enforced with breaker + telemetry escalation; conservative mode entered automatically. |
| Tamiyo handshake | `01.1`, `01.4` | Build `SystemStatePacket`, invoke Tamiyo with timeout (≈2 s), process `AdaptationCommand` | `trainer.run()` wraps Tamiyo/Kasmina calls with timeouts and conservative fallback | Implemented | Should‑have | Timeouts convert to conservative commands; security/HMAC handled upstream. |
| Kasmina integration | `01.4` | Apply adaptation commands using Leyline contracts | `trainer.run()` calls `KasminaClient.apply_command` | Implemented | Should‑have | Minimal application path; no error handling/escalation. |
| Multi‑seed gradient aggregation | `01.1` | Aggregate host/seed losses with state‑aware weights; stabilise | Not present | Missing | Should‑have | Only standard loss; no per‑seed weighting or PCGrad‑like conflict handling. |
| Unified LR controller | `01.3` | Central LR authority; rebuild optimisers safely; monitor integrity | Not present | Missing | Must‑have | Trainer uses raw optimiser; no LR governance or rebuild path. |
| Dynamic optimiser manager | `01.3` | Rebuild preserves momentum; registers LR groups | Not present | Missing | Should‑have | No support in prototype. |
| Checkpoint + WAL semantics | `01.2` | Atomic persistence with O_DSYNC; metadata consistency | `trainer._checkpoint()` saves torch state + `wal.json` | Partially Implemented | Should‑have | WAL exists; no O_DSYNC/atomicity guarantees; single last‑checkpoint only. |
| Two‑tier rollback | `01.2` | 500 ms fast rollback (LRU cache); 12 s full | `trainer.rollback_to_last_checkpoint()` | Missing | Must‑have | Only restores last checkpoint; no fast/full coordinators, no shared‑memory signalling. |
| Emergency protocol | `01.2` | Four‑level escalation; shared memory broadcast | Not present | Missing | Must‑have | No severity handling or broadcast. |
| Circuit breakers & conservative mode | `01.1` | Breakers around timing/integrity; degrade features | `TolariaTrainer` breaker + conservative mode events | Implemented | Must‑have | Breaker tracks failures; conservative mode emits telemetry + priority hint. |
| Telemetry & Oona | `01.1`, `01.4` | Structured metrics/events; publish to Oona | `build_telemetry_packet`, `publish_history()` | Implemented | Should‑have | No emergency bypass/priorities; basic metrics present. |
| Leyline as canonical | `01.4` | Use Leyline data classes/enums without local mappings | `leyline_pb2.SystemStatePacket`, `AdaptationCommand` | Implemented | Must‑have | Contracts respected in prototype. |
| Performance profiling | `01.1` | Capture timings; optional Chrome trace | Not present in code | Missing | Nice‑to‑have | See `docs/project/profiling.md` for harness; not wired in Tolaria. |
