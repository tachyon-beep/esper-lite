# BUG-030: Kasmina gate decisions are not synchronized under DDP (rank divergence risk)

- **Title:** `SeedSlot.advance_stage()` applies local gate decisions without DDP synchronization; `_sync_gate_decision()` exists but is not used, so ranks can diverge on lifecycle transitions
- **Context:** Kasmina seed lifecycle staging (`advance_stage()` / gate checks) in distributed (DDP) training
- **Impact:** P2 – correctness risk under multi-GPU. Divergent lifecycle stages across ranks can lead to architecture divergence (different active seeds/params), followed by shape mismatches, crashes, or silent desync.
- **Environment:** HEAD @ workspace; any future/experimental DDP run that executes Kasmina lifecycle logic on each rank
- **Reproduction Steps:**
  1. Run any training loop under DDP where lifecycle transitions can differ across ranks (e.g., different per-rank metrics/batches or nondeterministic gate inputs).
  2. Trigger `slot.advance_stage()` for the same slot on all ranks.
  3. Observe that one rank can transition while another does not, leaving `SeedState.stage` inconsistent across ranks; subsequent forward/backward can crash due to mismatched parameters/graphs.
- **Expected Behavior:** Lifecycle transitions should be rank-consistent under DDP (e.g., rank 0 decides and broadcasts, or gate decisions are synchronized deterministically across ranks).
- **Observed Behavior:** `advance_stage()` uses `self.gates.check_gate(...)` and immediately applies the result locally; `_sync_gate_decision()` (which implements an all-reduce consensus) is never called.
- **Logs/Telemetry:** May present as DDP hangs/crashes later (parameter shape mismatch) rather than at the moment of divergence.
- **Hypotheses:** DDP support is partially sketched (`_sync_gate_decision` exists) but not wired into the lifecycle transition path.
- **Fix Plan:**
  - Prefer rank-0 decision + broadcast of the resulting stage transition (and any side-effect commands), or
  - Wire in `_sync_gate_decision()` but only after ensuring every rank calls gate checks in identical order for all slots (otherwise collective-call deadlocks are likely).
- **Validation Plan:**
  - Add a minimal DDP test harness that forces a divergent local gate outcome and verifies the synchronized path keeps stages consistent across ranks (or fails fast with a clear error if DDP unsupported).
- **Status:** Fixed
- **Fix Applied:**
  - Replaced `_sync_gate_decision()` implementation from `all_reduce` consensus to rank-0 broadcast
  - Rank 0 makes the authoritative gate decision and broadcasts to all ranks via `broadcast_object_list`
  - This avoids JANK-003 deadlocks when ranks have seeds at different stages
  - Wired `_sync_gate_decision()` into `advance_stage()` after local gate check
  - Added divergence tracking in message field when local result differs from synced result
  - Removed unused `_ddp_sync_buffer` attribute
  - Added 6 unit tests in `tests/kasmina/test_ddp_gate_sync.py`
- **Links:**
  - `advance_stage()`: `src/esper/kasmina/slot.py:1238-1243`
  - `_sync_gate_decision()`: `src/esper/kasmina/slot.py:1964-2028`
  - Test file: `tests/kasmina/test_ddp_gate_sync.py`
  - Related jank: `docs/bugs/JANK-003-P1-kasmina-ddp-gate-deadlock.md` (partially addressed by rank-0 broadcast design)

