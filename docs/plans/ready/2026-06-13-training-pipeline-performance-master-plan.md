# Master Plan — Simic (PPO) + Tolaria (Host) Training Performance

**Status:** Ready
**Branch target:** `confounder-drain`
**Owner:** Lead engineer (performance)
**Reviewed by:** pytorch-expert, drl-expert, memory-diagnostician (design); Reality / RL-Numerics / Systems (adversarial)
**Hardware:** 2× RTX 4060 Ti (Ada sm_89, 16 GB, BF16-native, ~288 GB/s, low SM count → launch-overhead-bound, PCIe-only, no NVLink). 12 envs → 6/card.
**Repro command (the user's real run):**
```
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --devices cuda:0 cuda:1 --telemetry-level debug --sanctum \
  --config-json configs/config-3slot-3seed-baseline-shaped.json
```

---

## 1. Executive Summary

Two symptoms, both root-caused and verified against live code:

- **Symptom A — ~92% GPU memory fragmentation (reserved ≫ allocated).** Root cause: `vectorized_trainer.py:417` rebuilds all 12 env states at the top of every batch; each `create_env_state()` mints a fresh `MorphogeneticModel` + host optimizer + **fresh `torch.cuda.Stream`** (`env_factory.py:190-196`). The caching allocator keeps **per-stream** free pools; ~1800 dead streams over a run strand their segments. No `PYTORCH_CUDA_ALLOC_CONF` anywhere in the tree, and a synchronous `empty_cache()` (`utils/data.py:401`) actively fights any growable-segment policy.
- **Symptom B — running EAGER (torch.compile effectively off).** Root cause: `--sanctum` → `quiet_analytics=True` (`train.py:925`) → `vectorized.py:973-976` forces `effective_compile_mode="off"` unless `--force-compile`. This is an intentional TUI-safety gate; compile under TUI is *currently* unsafe because of `@torch.compiler.disable` CPU-sync graph breaks in the mask/dist path.

**Headline strategy (5 phases, measured before/after at each step):**

- **Set the allocator policy + delete the hostile flush first** (`expandable_segments:True,garbage_collection_threshold:0.8`; remove `empty_cache()`). Highest memory leverage, near-zero risk, instantly testable as an env-var prepend.
- **Instrument fragmentation before changing anything** (`log_frag` probe + `on_allocator_stats` telemetry into Karn) so every later change is A/B-measured, not asserted.
- **Move the PPO update into BF16 with an explicit FP32 seam, and make rollout precision-symmetric** — the single most expensive kernel (512-hidden LSTM × 150 steps × 12 envs) currently runs full FP32. **Rollout MUST be made BF16 too**, or the importance ratio is biased (CRITICAL-1).
- **Eliminate per-epoch / per-head CPU syncs** (MaskedCategorical validation, finiteness fold, forced-step loop, Q-loop, learnable-fraction) that serialize the launch-bound card.
- **Cure fragmentation structurally** with a persistent per-env stream pool + fenced `del env_states`; gate the high-effort model-reuse (`reset_to_initial`) behind a measured go/no-go.
- **Enable compile last, in order**: `dynamic=False` → validate without `--sanctum` → `--force-compile` → narrow the gate → host `reduce-overhead` compile. **`reduce-overhead` (CUDA graphs) is incompatible with `expandable_segments`** — this is a hard, plan-level constraint (see §5 STOP-HOST).

**Project-rule compliance:** No legacy/back-compat paths — every behavior change updates all call sites and deletes old code. No defensive `.get()/getattr/hasattr` to hide bugs (the two authorized exceptions — device normalization, owned-cache presence — are flagged inline). Telemetry is sacred — no panel removed; two new telemetry surfaces added and fully wired (not stubbed). New shared constants/event types go in `leyline`.

---

## 1b. Execution Review Verdict (2026-06-14) — GO_WITH_FIXES

A second adversarial review (reality-anchors / rl-numerics / pytorch-memory / systems-telemetry, against live `torch 2.9.1+cu128`, sm_89, GPUs up) returned **GO_WITH_FIXES**: 38/40 anchors exact, zero missing, zero contradicted. **One BLOCKER and four HIGH fixes are MANDATORY and folded into the items below.** None corrupts persisted data; all are local.

- **[BLOCKER · P1-BF16/CRITICAL-1] Rollout path needs the SAME FP32 seam.** The plan's claim (orig. §P1-BF16 step 2) that `old_log_probs` already carry the FP32 seam is **false**. Rollout `get_action` computes log-probs in `_sample_head` (`factored_lstm.py:1005/1025`) and the OP-head block (`:1054/1085`) with **no `.float()` upcast**. Once BF16 autocast wraps `get_action` (`vectorized_trainer.py:1474`), rollout log-prob math runs BF16 (~7.8e-3 log-prob divergence) while the update leg is FP32 → biased importance ratio → can FAIL V0. **Fix:** apply an explicit *unconditional* FP32 seam (`logits.float()` before `masked_fill`/`log_softmax`) in BOTH rollout sampling helpers; do **not** rely on the probability-floor branch incidentally forcing FP32 (holds only because all 8 default heads are floored). **Promote P1-EVAL V2** (cross-path `get_action` log_prob == `evaluate_actions` log_prob < 1e-5 on the SAME BF16 weights) **to a P1-BF16 BLOCKER gate run in shipping BF16.** Document `dropout=0.0` as a required CRITICAL-1 precondition (rollout `inference_mode` disables dropout; update train-mode enables it — any feature-net `dropout>0` re-opens this).
- **[HIGH · P1-VALID] The finiteness gate does NOT catch empty masks.** An all-False mask → `masked_fill(~mask, -1e4)` over all entries → uniform softmax over *invalid* actions → **finite** log-probs (`isfinite=True`), so `ppo_agent.py:886-911` never fires. With `validate=False` a structurally invalid action is sampled and executed against live Kasmina state. **Fix:** keep a sync-free empty-mask guard in the **training** path — fold a single batched `mask.any(dim=-1)` assertion into the P1-SYNC single finiteness sync (one extra sync/update, not per head). Reword Risk Register row: finiteness gate does NOT catch empty masks.
- **[HIGH · P2-DEL] De-dent the `del`/fence out of `if hub:`.** `if hub:` at `vectorized_trainer.py:1765` (indent 16) guards `:1784-1832`; `batch_summary` at `:1834` is indent 16. Inserting "between :1832/:1834" literally nests the fence inside `if hub:`, so with telemetry OFF the memory cure + async fence are skipped. **Fix:** place the block at loop-body indent (16) immediately before `batch_summary = BatchSummary(` (`:1834`), AFTER the `if hub:` block closes. (`env_states` last-read `:1789` is well before — safe.)
- **[HIGH · P3-HOST] Hard dependency on P2-RESET (model reuse).** `create_env_state` (host compile site `env_factory.py:152-158`) runs **per-batch** (`vectorized_trainer.py:417-429`, 12 fresh models/batch) → a fresh dynamo trace recurs every batch and defeats the goal unless the model OBJECT is reused. **Fix:** P3-HOST is GATED on P2-RESET landing GO. If P2-RESET is NO-GO, do NOT compile the host inside `create_env_state` (skip P3-HOST or hoist model construction out of the per-batch loop first).
- **[HIGH · P2-FRAGMETRIC] Event-type casing + explicit enum + no raw_events wiring.** Serializer writes `event.event_type.name` **UPPERCASE** (`nissa/output.py:96`, `collector.py:247`); the plan's lowercase `event_type='allocator_stats'` validation queries return **zero rows** → false silent-drop signal. Emission REQUIRES a new `TelemetryEventType.ALLOCATOR_STATS` member (`leyline/telemetry.py`) — make it explicit, plus the typed `AllocatorStatsPayload` dataclass + `on_allocator_stats` on `VectorizedEmitter`. `raw_events` is a generic `read_json_auto` view (`views.py:30`) that auto-ingests any event — it needs **no** wiring; query `WHERE event_type='ALLOCATOR_STATS'`. (A first-class typed view, if wanted, needs an explicit `CREATE OR REPLACE VIEW` + `json_extract` block.)
- **[NOTE · P0-ALLOC] torch 2.9.1 renamed the key.** `PYTORCH_CUDA_ALLOC_CONF` is deprecated (warning on every run + spawn worker) → set **`PYTORCH_ALLOC_CONF`** as primary (set both via `setdefault` to be safe). STOP-HOST empirically confirmed: capture under `expandable_segments:True` HARD-FAILS with `cudaErrorStreamCaptureInvalidated` (fail-loud, safer than the silent-wrong framing) — keep mutual exclusion.

Minor anchor corrections (NOTE): floor-dup unification targets the method def at `action_masks.py:592` (`:621` is its first body line); learnable-fraction compare is the operative line `ppo_agent.py:1221`; `empty_cache` NOTE block is `governor.py:234-240`; `batch_emitter=emitters[0]` built at `vectorized.py:966`; `import torch` is EAGER at `train.py:9` (Open-Q2 premise of "lazy" is false — top-of-module placement still satisfies the constraint; keep the `is_initialized()` guard).

---

## 2. Verified Diagnosis (corrected anchors that survived review)

| Concern | Corrected anchor | Notes |
|---|---|---|
| Trainer class | `VectorizedPPOTrainer` `@dataclass`, `vectorized_trainer.py:197`; `__post_init__:268`; `device: str` field `:263`; `run():318` | Context's `VectorizedTrainer`/`__init__` was wrong. |
| Module logger | **No module-level `_logger`**; only `self.logger` (instance, `:372`). | **Confirmed.** `log_frag` must add `_logger = logging.getLogger(__name__)` at module top. |
| Env-state rebuild | `vectorized_trainer.py:408` `while`; list-comp `create_env_state` `:417-429`; no `del`. | Confirmed. |
| Fresh stream/batch | `env_factory.py:190-196`. | Path is `simic/training/env_factory.py`. |
| `empty_cache()` on eviction | `utils/data.py:396-401` (`evicted =`, then `empty_cache()`). | Confirmed. |
| `empty_cache` avoidance rationale | `tolaria/governor.py:233-240` NOTE block. | Off-by-one in context (233, not 234). |
| PPO FP32 fence | `ppo_agent.py:824-851` (`autocast(enabled=False)` `:833`, `.float()` casts). | The `B11-PT-01` fix; matched the FP32 rollout — see CRITICAL-1. |
| Enclosing AMP context | `vectorized.py:434-436` (`autocast(device_type="cuda", dtype=amp_dtype)`). | BF16 on Ada. AMP **does** reach the update; `:833` kills it. |
| Rollout `get_action` | `factored_lstm.py` via `vectorized_trainer.py:1474` — **NO autocast wraps it** → FP32 today. | Load-bearing for CRITICAL-1. |
| `evaluate_actions` | `factored_lstm.py:1215`; `MaskedCategorical` build `:1426-1428`; bundle `lstm_bundle.py:215`. | Confirmed. |
| Sync-free helpers | `_apply_floor_to_logits` `:891-960`, `_normalized_entropy_from_masked_logits` `:963-979`, `_sample_head` `:981-1029` — **nested closures in `get_action` (`:753`)**. | Must be hoisted to module scope. |
| MaskedCategorical | class `action_masks.py:522`; `validate: bool = True` `:542`; `_validate_action_mask` `.any()` `:499`; `_validate_logits` `.any()×2` `:513`; `Categorical` build `:590`; `_apply_probability_floor` `:621`; `MASKED_LOGIT_VALUE` `:582`. | Validation is `@torch.compiler.disable`. |
| Forced-step loop | `ppo_agent.py:534-540`; `data=get_batched_sequences` `:565`; `valid_mask` `:566`. | **`valid_mask` defined AFTER the loop** — confirmed; must be reordered/derived. |
| Q-loop | `ppo_agent.py:740-744` (under `no_grad` `:731`); `_compute_value` `factored_lstm.py:554`. | Telemetry only. |
| Finiteness gate | `ppo_agent.py:886-911` fast-path; `:913 if nonfinite_found`; KL early-stop `:1022`; learnable-fraction `.item()` `:1219-1221`; ratio_exploded `:1282-1286`. | Confirmed. |
| Compile sites | `helpers.py:226` (`mode="default", dynamic=True`); `factory.py:97`; `ppo_agent.py:1602`; `lstm_bundle.py:478` default `dynamic=True`. | Confirmed. |
| Compile gate | `vectorized.py:973-976`; threading `:985/:1043/:1068`; wiring `train.py:925-926,557,375`. | Confirmed. |
| Counterfactual path | `MorphogeneticModel.fused_forward()` `host.py:803`, invoked `counterfactual_eval.py:75`. **`force_alpha` is NOT in this path.** | Context correction confirmed. |
| Slot stage branches | `slot.py:2055 forward(alpha_override=...)`; STE `:2096`; blend `:2123-2144`. | ~6-8 specialized graphs (docstring unverified, non-blocking). |
| Data clone | `utils/data.py:591` `result[...] = (inp.clone(), tgt.clone())` on **default stream**; consumer `wait_stream` `vectorized_trainer.py:594` & `:892`; `record_stream` `:899-900`. | Source-view lifetime risk — see STOP-CLONE. |
| Governor snapshot | `governor.py:288` `v.detach().cpu()`; cadence `vectorized_trainer.py:189-191`; stream contract `:214-231`; `device` `:219-221`; rollback `load_state_dict` `~:458`. | Pinned-buffer aliasing — see STOP-SNAP. |
| `seed_gradient_ratio` | **`kasmina/isolation.py`**, ratio `:232`, emit `:237` (NOT `:172,232`, NOT `simic/attribution/`). | Host-loop telemetry; **P1-BF16 cannot affect it**; **P3-HOST can** (gate-flap). |
| Emitter | `batch_emitter` typed `Any` `vectorized_trainer.py:235`; concrete `VectorizedEmitter`, `on_batch_completed` `emitters.py` (`:455` reads `env_states` synchronously, stores no refs). | New `on_allocator_stats` must be added on the concrete class. |
| `EnvFactoryContext` | `@dataclass(frozen=True)` `env_factory.py:110-129`, **18 fields**, last `group_id`; built `vectorized.py:1305-1324`; env→device map `:866`. | Context said 19; harmless. |
| `del` safety | No `env_states` reference between `:1832` and the loop end / `finally`. | Confirmed safe. |
| Non-existent | `mutants/` directory does **not** exist. | Ignore that warning entirely. |

---

## 3. Phased Roadmap

> Conventions for every item: **ID · Files+anchors · Change (before/after) · Deps · Validation · Rollback · Risk.** Deferred/high-effort items are marked **[DEFERRED]**.

---

### PHASE 0 — Instrument & Quick Wins (zero/low risk)

#### P0-FRAGPROBE — Reusable `log_frag(device, tag)` probe (land first)
- **Files:** `src/esper/simic/training/vectorized_trainer.py` — module scope (helpers live ~`:185-195`), call sites `:408` (batch start) and between `:1848`/`:1849` (batch end).
- **Change:** Add a module logger **(reviewer-corrected: `_logger` does not exist)** and a read-only probe (`memory_stats` does not sync).
  ```python
  # module top of vectorized_trainer.py (after imports)
  import logging
  _logger = logging.getLogger(__name__)

  def log_frag(device: str | torch.device, tag: str) -> None:
      """Read-only CUDA fragmentation probe (no sync, no empty_cache).
      frag_gap = reserved - allocated = memory the allocator holds but can't hand out."""
      if not torch.cuda.is_available():
          return
      dev = torch.device(device)                 # authorized device-normalization (CLAUDE.md #6)
      if dev.type != "cuda":
          return
      stats = torch.cuda.memory_stats(dev)
      allocated = torch.cuda.memory_allocated(dev) / 1024**2
      reserved = torch.cuda.memory_reserved(dev) / 1024**2
      _logger.info(
          "frag[%s] dev=%s alloc=%.0fMB reserved=%.0fMB active=%.0fMB frag_gap=%.0fMB nalloc_retries=%d",
          tag, dev, allocated, reserved,
          stats["active_bytes.all.current"] / 1024**2,
          reserved - allocated, stats["num_alloc_retries"],
      )
  ```
  Call sites:
  ```python
              while batch_idx < total_batches:
                  log_frag(self.device, f"batch{batch_idx}.start")
  ...
                  episodes_completed = batch_epoch_id
                  log_frag(self.device, f"batch{batch_idx}.end")
                  batch_idx += 1
  ```
- **Deps:** none. Instrument before everything.
- **Validation:** `grep -n "log_frag" src/esper/simic/training/vectorized_trainer.py` → 1 def + 2 calls. Run produces `frag[...] frag_gap=... nalloc_retries=...` at INFO (debug telemetry level satisfies it).
- **Rollback:** delete def + 2 calls + the `_logger`/`logging` lines if unused.
- **Risk:** negligible (read-only INFO). This is a real telemetry probe — it stays.

#### P0-ALLOC — Set `PYTORCH_CUDA_ALLOC_CONF` before any CUDA context
- **Files:** `src/esper/scripts/train.py` — top of module (after docstring `:2`, before `import argparse`); `spawn` at `:1030` re-imports the module in workers (they inherit `os.environ`).
- **Change:**
  ```python
  """Training CLI for Simic RL algorithms."""

  import os

  # Allocator policy MUST precede the first CUDA allocation (and the spawn re-import).
  # expandable_segments cures per-stream segment stranding from the per-batch env rebuild;
  # gc_threshold caps reserved-pool bloat on the 16 GB cards.
  os.environ.setdefault(
      "PYTORCH_CUDA_ALLOC_CONF",
      "expandable_segments:True,garbage_collection_threshold:0.8",
  )

  import argparse
  ...
  import torch
  ```
  **(Reviewer mitigation W2/R3 — fail loud, not silent):** immediately after the `setdefault`, assert the context isn't already live:
  ```python
  import torch as _t
  if _t.cuda.is_initialized():
      import warnings
      warnings.warn(
          "PYTORCH_CUDA_ALLOC_CONF set after CUDA init; allocator policy will NOT apply.",
          RuntimeWarning, stacklevel=2,
      )
  ```
  `setdefault` lets an operator's exported override win (the documented PyTorch channel, not a back-compat shim — no second code path).
- **Deps:** none. **Mutually exclusive at runtime with P3-HOST `reduce-overhead`** (see STOP-HOST).
- **Validation:**
  `PYTHONPATH=src uv run python -c "import esper.scripts.train, os; print(os.environ['PYTORCH_CUDA_ALLOC_CONF'])"` → exact string. Runtime: P0-FRAGPROBE `frag_gap`/`num_alloc_retries` stop growing; `memory_stats(dev)["reserved_bytes.all.peak"]` lower.
- **Rollback:** delete the `os.environ.setdefault` block (+ the guard).
- **Risk:** low. Stable on PyTorch 2.x/CUDA 12.x on sm_89. No numerics/RNG impact. (Version confirm — see §7.)

#### P0-EMPTYCACHE — Delete the synchronous flush on precompute eviction
- **Files:** `src/esper/utils/data.py:396-401`; rationale `governor.py:233-240`.
- **Change (before):**
  ```python
      if augment_mode == "precompute" and cache_key not in _GPU_DATASET_CACHE:
          evicted = clear_gpu_dataset_cache(device=device, augment_mode="precompute")
          if evicted > 0:
              if torch.cuda.is_available():
                  torch.cuda.empty_cache()
  ```
  **(after)** — drop the now-unused `evicted` capture and dead branch (no `_evicted` rename — delete it):
  ```python
      if augment_mode == "precompute" and cache_key not in _GPU_DATASET_CACHE:
          clear_gpu_dataset_cache(device=device, augment_mode="precompute")
  ```
- **Deps:** pairs with P0-ALLOC (the flush fights `expandable_segments`). Land together.
- **Validation:** `grep -rn empty_cache src/esper/utils/data.py` → none. On a seed change (eviction), P0-FRAGPROBE shows no reserved-pool collapse-then-regrow; `_GPU_DATASET_CACHE` stays bounded across seeds.
- **Rollback:** restore the block.
- **Risk:** low-moderate. ~750 MB/seed now stays in the caching pool; `garbage_collection_threshold:0.8` reclaims under pressure. Watch reserved high-water across many seed rotations.

#### P0-TF32 — Enable TF32 matmul + cuDNN TF32 at trainer init
- **Files:** `vectorized_trainer.py:268` (`__post_init__`, before `ActionExecutionContext`).
- **Change:**
  ```python
      def __post_init__(self) -> None:
          # Ada sm_89 TF32: ~2× FP32 matmul/conv on tensor cores at ~1e-3 rel error.
          torch.set_float32_matmul_precision("high")
          torch.backends.cudnn.allow_tf32 = True
          self.action_execution_context = ActionExecutionContext(
  ```
  Do **not** also set the deprecated `torch.backends.cuda.matmul.allow_tf32` (redundant second path).
- **Deps:** none. Independent.
- **Validation:** `python -c "import torch; torch.set_float32_matmul_precision('high'); print(torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)"` → `high True`. PPO `kl`/`clip_fraction`/`value_loss` within run-to-run noise.
- **Rollback:** delete the two lines.
- **Risk:** low; ~1e-3 rel error, deterministic given fixed inputs.

**Phase 0 landing order:** P0-FRAGPROBE → P0-ALLOC → P0-EMPTYCACHE → P0-TF32.

---

### PHASE 1 — Eager-mode Compute & Correctness (RL-critical, parity-gated)

> **Numerics reviewer mandate:** every BF16 parity test must run in the **shipping precision** (not `amp_dtype:"off"`). The single highest-value test is the **epoch-0 frozen-weight joint-ratio ≈ 1** check (CRITICAL-1).

#### P1-EVAL — Sync-free `evaluate_actions` (hoist shared helpers); land FIRST, parity-gated
- **Files:** `factored_lstm.py:1426-1428` (replace), closures `:891-960/:963-979/:981-1029` (hoist), `:1000-1002` (validate gate pattern); `action_masks.py:621` (floor dup); upstream `lstm_bundle.py:215`.
- **Change:**
  1. **Hoist** the three closures out of `get_action` to **module-level** functions (delete the in-`get_action` copies; update `_sample_head` and the OP-head block `:1041-1054` to call the hoisted versions):
     ```python
     def _apply_floor_to_logits(logits, mask, min_prob): ...
     def _normalized_entropy_from_masked_logits(masked_logits, mask): ...
     def _masked_log_prob(masked_logits, action):
         all_log_probs = F.log_softmax(masked_logits.float(), dim=-1)   # FP32 seam
         return all_log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
     ```
     **(HIGH-1 mitigation):** inside `_apply_floor_to_logits`, force `effective_floor = torch.tensor(min_prob, dtype=torch.float32)` and operate on FP32 logits **unconditionally** — never inherit BF16 from the call-site dtype.
  2. **Replace** `:1426-1428`:
     ```python
     min_prob = probability_floor.get(key) if probability_floor else None
     # Upcast seam: heads run BF16 (P1-BF16); log_softmax/entropy/floor stay FP32 for stable ratios/KL.
     masked_logits = logits_flat.float().masked_fill(~mask_flat, MASKED_LOGIT_VALUE)
     if min_prob is not None and min_prob > 0:
         masked_logits = _apply_floor_to_logits(masked_logits, mask_flat, min_prob)
     log_probs[key] = _masked_log_prob(masked_logits, action_flat).reshape(batch, seq)
     entropy[key] = _normalized_entropy_from_masked_logits(masked_logits, mask_flat).reshape(batch, seq)
     if MaskedCategorical.validate:
         _validate_action_mask(mask_flat)
         _validate_logits(logits_flat)
     ```
     (`probability_floor.get(key)` is an authorized owned-dict read, mirroring `_sample_head`.)
  3. **(HIGH-1 / W1 / R4 — IN SCOPE, not deferred):** unify the duplicate floor so it cannot drift. Make `MaskedCategorical._apply_probability_floor` (`action_masks.py:621`) delegate to the hoisted `_apply_floor_to_logits` (one-liner). The no-legacy rule mandates removing the second copy now.
- **Deps:** none to land; **P1-BF16 requires this**.
- **Validation:**
  - **V1 (FP32 algebra gate):** `test_evaluate_actions_parity` — old `MaskedCategorical` path vs new path, `torch.testing.assert_close(rtol=1e-5, atol=1e-6)` for `log_prob` and `entropy`. **Hard gate before P1-BF16.**
  - **V1b (HIGH-1, BF16 regime):** floor parity with a **BF16 call-site** logits tensor → assert FP32-identical floored logits.
  - **V2 (cross-path):** single-step `evaluate_actions` log_prob == `get_action` log_prob (rollout-vs-update consistency) to 1e-5.
  - **V3 (HIGH-2 entropy):** entropy parity for (a) single-valid-action → 0 edge and (b) **2-valid-action sparse head with floor active**, in **both FP32 and BF16**.
- **Rollback:** restore the 3-liner; keep helpers hoisted (independently correct) or re-nest.
- **Risk:** high if floor/entropy diverge — unification + V1/V1b/V3 are the gate.

#### P1-BF16 — BF16 autocast through `evaluate_actions` + **rollout precision symmetry**
- **Files:** `ppo_agent.py:824-851` (delete inner fence); enclosing autocast `vectorized.py:434-436`; **rollout `get_action` `vectorized_trainer.py:1474`**; comment `action_masks.py:576-581`.
- **Change:**
  1. Delete the `autocast(enabled=False)` block + `.float()` casts; let BF16 flow:
     ```python
     # PPO update runs under the BF16 autocast from _run_ppo_updates (vectorized.py).
     # LSTM + heads in BF16; log_softmax/log_prob/entropy/floor upcast to FP32 at the
     # masked-logits seam (P1-EVAL). BF16 has FP32 exponent range → no GradScaler.
     result = self.policy.evaluate_actions(
         data["states"], data["blueprint_indices"], actions, masks,
         hidden=(data["initial_hidden_h"], data["initial_hidden_c"]),
         probability_floor=self.probability_floor,
         aux_stop_gradient=self.aux_stop_gradient,
     )
     ```
  2. **(CRITICAL-1 — MANDATORY):** wrap rollout `get_action` (`vectorized_trainer.py:1474`) in the **same** `autocast(device_type="cuda", dtype=amp_dtype)` so `old_log_probs` and update `log_probs` share BF16 backbone error → unbiased ratio. Verify the rollout buffer stores `old_log_probs` as **FP32** (it should: the log_prob has the FP32 seam).
  3. Delete the obsolete `action_masks.py:576-581` "the main fix is in ppo_agent.py which runs evaluate_actions outside autocast" comment (no-legacy). Keep the `logits.float()` upcast — it is now the **primary** seam.
- **Deps:** **P1-EVAL first** (FP32 seam must exist).
- **Validation:**
  - **V0 (CRITICAL-1 gate, shipping BF16 config):** rollout → store `old_log_probs` → `evaluate_actions` under shipping autocast on the **same buffer, unchanged weights** → assert `|joint_ratio − 1| < 1e-3` at epoch 0. **This is the gate that proves leg symmetry; it must run in BF16, not FP32.**
  - **V1 (parity, shipping precision):** capture `log_prob/entropy/value/approx_kl/clip_fraction/policy_loss/value_loss/explained_variance`; BF16-vs-FP32 baseline: `|Δapprox_kl|/(approx_kl+1e-6) < 5e-2`, `policy_loss` within 2%, per-head grad-norm within 5%.
  - **V2 (trajectory):** 50 updates BF16 vs 3-seed FP32 baseline; `explained_variance`/`approx_kl` within ±2σ (Karn `ppo_updates`).
  - **V3:** `TORCH_LOGS=+autocast` confirms LSTM/linear cast to bf16; **no `GradScaler`** instantiated. `lstm_c_rms`/`lstm_has_nan` panels stay healthy.
- **Rollback:** restore the `enabled=False` block + `.float()`; remove the rollout autocast wrap.
- **Risk:** **medium-high.** BF16 LSTM over 150 steps can drift (clamp `tanh(c/50)*50` `factored_lstm:1282` bounds it). **Fallback (default to it if V0/V3 fail):** keep the **LSTM in FP32** via scoped `autocast(enabled=False)` around `self.lstm(...)` (`factored_lstm:1278`), BF16 only heads/feature-net; promote full-BF16-LSTM only once V0 (epoch-0 ratio≈1) passes.

#### P1-VALID — Disable `MaskedCategorical.validate` in the training process
- **Files:** set once at trainer setup — `vectorized.py` near the perf-gate resolution (~`:973`).
- **Change:**
  ```python
  # Per-head .any()/.isnan() validation forces CPU syncs (action_masks.py:499,513) + graph
  # breaks. NaN/Inf is caught by the PPO finiteness gate; empty masks are a construction-time
  # invariant. Validation stays True by default for dev/eval processes.
  MaskedCategorical.validate = False
  ```
  No config flag (no feature-flag-for-old-behavior). A debug process sets the class attr itself — the documented `validate` toggle, not a shim.
- **Deps:** after P1-EVAL (new path honors the same flag). Synergistic with the compile cluster (removes `@torch.compiler.disable` breaks).
- **Validation:** 5-epoch run, no `ValueError` regressions; inject a NaN logit → finiteness gate still fires and records `finiteness_gate_failures`, optimizer step skipped. Profiler: `aten::any`/`aten::is_nonzero` from validation → 0 in training process.
- **Rollback:** remove the line (default `True`).
- **Risk:** medium. Empty mask now surfaces via the finiteness gate (one wasted update, source message lost) — keep validation ON in eval/CI.

#### P1-VEC — Vectorize the forced-step `.item()` loop **(reviewer-corrected ordering)**
- **Files:** `ppo_agent.py:534-540`; `data`/`valid_mask` at `:565-566`.
- **Change:** `valid_mask` is defined **after** this block — **move `data = self.buffer.get_batched_sequences(...)` and `valid_mask = data["valid_mask"]` above line 530**, OR derive the mask from `step_counts` with `torch.arange`. Then:
  ```python
  total_timesteps = sum(self.buffer.step_counts)
  if total_timesteps > 0:
      # forced_actions and valid_mask are both [num_envs, max_steps] bool on device.
      forced_count = int((forced_actions & valid_mask).sum().item())   # 1 sync, was 12
      forced_step_ratio = forced_count / total_timesteps
      usable_actor_timesteps = total_timesteps - forced_count
  else:
      forced_step_ratio = 0.0
      usable_actor_timesteps = 0
  ```
  If `get_batched_sequences` pads `forced_actions` beyond `step_counts`, the `& valid_mask` makes it robust; confirm shapes match at implementation.
- **Deps:** none (besides the local reorder).
- **Validation:** integer equality of `forced_step_ratio`/`usable_actor_timesteps` vs the old loop across 5-env and 12-env runs; unit test with hand-built `forced_actions`/`step_counts`.
- **Rollback:** restore the loop + original ordering.
- **Risk:** low; `forced_step_ratio` is a telemetry/D5 input — integer-equality test protects it.

#### P1-QLOOP — Batch the per-op Q-value telemetry loop
- **Files:** `ppo_agent.py:740-744` (under `no_grad`); `_compute_value` `factored_lstm.py:554`.
- **Change:**
  ```python
  op_indices = torch.arange(NUM_OPS, device=self.device, dtype=torch.long).reshape(NUM_OPS, 1)
  lstm_out_rep = lstm_out.expand(NUM_OPS, 1, -1).contiguous()   # contiguity (LOW-1)
  op_q_values = self.policy.network._compute_value(lstm_out_rep, op_indices).reshape(NUM_OPS)
  ```
  **Read `_compute_value` first** to confirm it supports a batch dim over op-conditioning.
- **Deps:** none.
- **Validation:** `assert_close(batched, looped, rtol=1e-5, atol=1e-6)`; `q_variance`/`q_spread` unchanged; `op_q_values` length stays `NUM_OPS` (asserted at `leyline/telemetry.py:937`).
- **Rollback:** restore the loop.
- **Risk:** low; telemetry-only, parity-tested.

#### P1-SYNC — Defer per-epoch host-truthiness syncs (finiteness fold + learnable-fraction + ratio-explosion)
- **Files:** `ppo_agent.py:886-911` (finiteness), `:1022` (KL early-stop — **leave**), `:1219-1221` (learnable-fraction), `:1251` (telemetry stack), `:1282-1286` (ratio_exploded).
- **Change:**
  1. **Finiteness fold (17 → 1):** stack new/old log-probs + values, one fused `&`; drill into the **unchanged** per-head attribution slow-path only on failure. **(MED-1):** assert dtype homogeneity before stacking (new = FP32-seam, old = BF16 if rollout is BF16):
     ```python
     all_new_lp = torch.stack([log_probs[k].float() for k in HEAD_NAMES])
     all_old_lp = torch.stack([data[f"{k}_log_probs"][valid_mask].float() for k in HEAD_NAMES])
     finite_flag = (torch.isfinite(all_new_lp).all()
                    & torch.isfinite(all_old_lp).all()
                    & torch.isfinite(values).all())
     if not bool(finite_flag):           # single sync
         ...  # existing per-head NaN/Inf attribution VERBATIM; populates head_nan_detected etc.
         continue
     ```
  2. **learnable-fraction (8 → 1):** after the grad-norm loop, batch the `== 0.0` compare via one `torch.stack([...]).tolist()`/single sync, then index — preserve `head_gradient_state_history` "not_learnable" labels bit-for-bit.
  3. **ratio_exploded (1 → 0 extra):** append the explosion flag to the already-batched `logging_tensors` stack at `:1251` so it rides the existing single GPU→CPU transfer; read `if logging_tensors[N]:` from the materialized host values. The rare `RatioExplosionDiagnostic.from_batch` build stays as-is.
  4. **KL early-stop `:1022`:** **NO CHANGE.** It is the irreducible per-epoch sync (the loop must branch to break; removing it breaks KL-adaptive early stop + entropy annealing's `train_steps`). Documented as the floor.
- **Deps:** after P1-EVAL (finiteness reads the new tensors).
- **Validation:** profiler — happy-path epoch syncs drop from ~(17+8+1+KL) to ~(1+1+KL). **(MED-1 NaN test):** inject NaN into a **single head** → folded path yields **byte-identical** `head_nan_detected`/`head_inf_detected`/`nonfinite_sources`/`finiteness_gate_failures`. `head_gradient_state_history` identical vs baseline over 5 epochs. `ratio_diagnostic` still recorded on injected explosion.
- **Rollback:** per-sub-change blocks revert independently.
- **Risk:** telemetry-sacred zone — slow-path attribution and grad-state labels MUST be preserved bit-for-bit (the validation gates).

**Phase 1 landing order (staged, NOT one big commit — R1):** P1-EVAL (commit, V1 gate) → P1-BF16 (commit, V0 gate) → P1-VALID → {P1-VEC, P1-QLOOP, P1-SYNC} any order.

---

### PHASE 2 — Structural Memory (the real fragmentation cure)

#### P2-FRAGMETRIC — Reserved-vs-allocated + retry telemetry into Karn (land first/alongside)
- **Files:** emit in `vectorized_trainer.py` after `:1832`, **before** the P2-DEL teardown; new method `on_allocator_stats` on the concrete `VectorizedEmitter` (trace from `emitters` construction `vectorized.py:940-965` — `batch_emitter` is typed `Any` at `:235`); new typed event/constants in **`leyline`**; **Karn DuckDB view wiring in the same change** (no silent drop).
- **Change:**
  ```python
  for device in sorted({es.env_device for es in env_states}):
      if torch.device(device).type != "cuda":
          continue
      stats = torch.cuda.memory_stats(device)
      batch_emitter.on_allocator_stats(
          batch_idx=batch_idx, device=device,
          allocated_bytes=stats["allocated_bytes.all.current"],
          reserved_bytes=stats["reserved_bytes.all.current"],
          fragmentation_bytes=stats["reserved_bytes.all.current"] - stats["allocated_bytes.all.current"],
          num_alloc_retries=stats["num_alloc_retries"], num_ooms=stats["num_ooms"],
      )
  ```
  Spec the payload as a typed `leyline` dataclass; **do not stub** (telemetry rule). **(W3):** call through the concrete typed emitter, not the `Any` alias, and `mypy`-check the new method in CI so a typo can't silently emit nothing.
- **Deps:** independent to implement; purpose is to validate P2-STREAMPOOL/P2-DEL/P0-ALLOC. Land first to capture baseline.
- **Validation:** baseline run `fragmentation_bytes/reserved_bytes ≈ 0.9`, retries climbing. Karn: `SELECT batch_idx, fragmentation_bytes, num_alloc_retries FROM raw_events WHERE event_type='allocator_stats' ORDER BY batch_idx` returns rows (confirms view wired) and trends flat post-fix.
- **Rollback:** remove emit + method + leyline event + view.
- **Risk:** very low (`memory_stats` is host-side, no sync). Risk is forgetting the Karn view → wire it in-change.

#### P2-STREAMPOOL — Persistent per-env CUDA stream pool
- **Files:** `EnvFactoryContext` `env_factory.py:110-129` (add field after `env_device_map`); creation `:190-196` → lookup; pool built before `EnvFactoryContext(...)` `vectorized.py:1305`; env→device map `:866`.
- **Change:**
  1. Field: `env_streams: list["torch.cuda.Stream | None"]`.
  2. Replace `:190-196` per-call creation with `stream = context.env_streams[env_idx]` (keep `env_device_obj = torch.device(env_device)` — still needed for scaler/autocast — moved above this block).
  3. Build once at `vectorized.py:1305`:
     ```python
     env_streams: list[torch.cuda.Stream | None] = [
         torch.cuda.Stream(device=torch.device(dev)) if torch.device(dev).type == "cuda" else None
         for dev in env_device_map
     ]
     ```
     and pass `env_streams=env_streams,` into `EnvFactoryContext(...)`.
- **Deps:** none. Foundational for P2-DEL.
- **Validation:** P2-FRAGMETRIC gap + `num_alloc_retries` stop growing. **Bit-identical `val_acc` per batch for same `--seed`** (streams carry no RNG — `env_factory.py:142-143` seeds via `manual_seed`); run twice, diff Karn `epochs.val_acc`. `len(context.env_streams) == n_envs`.
- **Rollback:** revert the 3 edits atomically.
- **Risk:** low. Frozen dataclass holds a list never mutated (frozenness honored; **do not** `.append` to it — W: frozen blocks rebind, not mutation). `record_stream` call sites (`vectorized_trainer.py:556-557,610,899-900`) now reference a stable object.

#### P2-DEL — Fenced `del env_states` + per-env `stream.synchronize()` at batch end
- **Files:** insert after `check_performance_degradation(...)` ends (`:1832`), before `batch_summary` (`:1834`), at loop-body indent; epoch-end sync pattern `:672-674`.
- **Change:**
  ```python
              # P2-DEL: on_batch_completed (:1793) is the last consumer of env_states.
              # Sync each persistent stream so no async kernel still references the
              # model/optimizer tensors we drop, then release refs for allocator reuse.
              for env_state in env_states:
                  if env_state.stream is not None:
                      env_state.stream.synchronize()
              del env_states
  ```
  **Never** `empty_cache()` here (governor.py:233-240 anti-pattern).
  **Governor contract proof (preserved):** the snapshot helper (`:189-191`) runs in the validation/governor phase, after epoch-end sync and **outside** any `with torch.cuda.stream(...)`. `governor.py:223-231` checks `current_stream == default_stream` at call time — depends only on whether a stream context is active, never on stream object identity/lifetime. P2-STREAMPOOL changes identity only; P2-DEL's sync is at batch end, outside any context. Contract holds.
- **Deps:** **P2-STREAMPOOL first.**
- **Validation:** `reserved_bytes.all.peak` plateaus; `num_alloc_retries` flat after warmup. **One short run under `CUDA_LAUNCH_BLOCKING=1` (`--max-epochs 2`) completes with no illegal-memory-access** (proves the fence). Bit-identical `val_acc`.
- **Rollback:** delete the 5-line block (GC reclaims on next loop rebind anyway).
- **Risk:** low-medium. Future code reading `env_states` after `:1832` → `NameError` (intended fail-loud). **(Systems W4):** `BatchSummary`/`reward_summary_accum` hold no GPU tensors — confirmed safe.

#### P2-RESET — Model/optimizer reuse via `reset_to_initial()` **[DEFERRED — NO-GO 2026-06-14, NOT BUILT]**
- **DECISION (2026-06-14): NO-GO — deliberately not built.** Measured on cuda:0 over 25 batches × 12 envs with P0-ALLOC + P2-STREAMPOOL + P2-DEL all landed (`--telemetry-file` ALLOCATOR_STATS, expandable_segments confirmed active): **allocated stable ~331 MB (no per-batch leak), reserved plateaued 1302→1366 MB by batch 12, num_alloc_retries = 0, num_ooms = 0.** The raw `fragmentation_bytes/reserved_bytes ≈ 0.75` exceeds the literal `>0.30` GO threshold, BUT that threshold was calibrated for the OLD pre-expandable allocator where high reserved/allocated meant dead-stream segment stranding. Under expandable_segments the reserved-VA gap is benign and reused (retries = 0 proves it); the per-stream stranding the plan worried about is CURED by P2-STREAMPOOL (no more ~1800 dead streams) and allocated no longer grows per batch. The ACTUAL harm indicators (retries, OOMs) are zero, so building the HIGH-RISK reset (state leak / RNG-order drift / governor-telemetry breakage) would add risk for no measured benefit. **Monitoring posture:** P2-FRAGMETRIC stays wired; if a real long run shows `num_alloc_retries` climbing or any `num_ooms`, flip to GO and build it then. P3-HOST is therefore also gated off (its hard dep is a GO here) — host stays eager / `mode="default"`.
- **Status:** Design only. Implement **only if** the go/no-go below is GO.
- **Go/No-Go (after P2-STREAMPOOL + P2-DEL + P0-ALLOC land, ≥20 batches on the real config):**
  - **NO-GO:** `fragmentation_bytes/reserved_bytes < 0.15` **and** retries flat **and** no OOMs → don't build it. *(Refined by the decision above: under expandable_segments, "retries flat AND no OOMs" is the load-bearing condition; the ratio is a leading indicator that the new allocator inflates benignly.)*
  - **GO:** ratio > 0.30, or retries climbing, or any OOM.
- **Files (for the implementer):** `parallel_env_state.py:211-227` (reset raises on non-dormant — the invariant to satisfy); `env_factory.py:145-188` (model+SGD construction to reproduce as initial); caller list-comp `vectorized_trainer.py:417-429`; new `MorphogeneticModel.reset_to_initial()` (`host.py`); governor re-snapshot site (`env_factory.py:243` — confirm).
- **Contract:** snapshot host-param state at construction (`detach().clone()`); `reset_to_initial()` force-prunes seeds to DORMANT **through the real Kasmina prune path** (telemetry must fire — no silent `slot.seed=None`), `load_state_dict(..., assign=False)` (reuse storage), `host_optimizer.state.clear()`; hoist `env_states` out of the loop and call reset+re-seed+`reset_episode_state` at batch top.
- **Determinism (MED-3 — load-bearing):** re-seeding with `make_env_seed` alone is **insufficient** — fresh-create burns N RNG draws on weight init that reset skips, shifting the global RNG stream position. **Snapshot `torch.get_rng_state()` immediately after model init in the fresh path and restore THAT state in `reset_to_initial`.** Parity gate must run **≥1 full episode post-reset** (so sampling/augmentation RNG fires) and compare **per-step action samples**, not just `val_acc`.
- **Validation:** bit-identical per-step actions reused-vs-fresh same seed; P2-FRAGMETRIC near-zero allocation growth; `seed_lifecycle` prune events at each batch boundary; governor `last_good_state` == restored initial weights; `seed_gradient_ratio` still emits.
- **Rollback:** revert caller hoist; restore in-loop `create_env_state` list-comp.
- **Risk:** high if rushed (state leak → silent corruption; RNG-order drift; telemetry/governor breakage). That's why it's gated.

**Phase 2 landing order:** P2-FRAGMETRIC → P2-STREAMPOOL → P2-DEL → measure → P2-RESET go/no-go.

---

### PHASE 3 — Compile Enablement + Host Path

> **Hard constraint (STOP-HOST):** `reduce-overhead` uses CUDA graphs (fixed captured addresses); `expandable_segments:True` (P0-ALLOC) remaps virtual address ranges to grow segments — **the two are incompatible**: a captured graph replaying against a remapped VA reads stale/wrong addresses (silent wrong activations, or rare illegal-access). If P3-HOST `reduce-overhead` is enabled, **P0-ALLOC's `expandable_segments` MUST be disabled** (and vice versa). Default posture: keep `expandable_segments` (memory win is proven, broad); use host `mode="default"` (no graph capture) unless a measured throughput case justifies flipping.

#### P3-DYN — `dynamic=True` → `dynamic=False` on PPO + host helper compiles
- **Files:** `helpers.py:226`, `factory.py:97`, `ppo_agent.py:1602`, `lstm_bundle.py:478` (default).
- **Change:** flip all four to `dynamic=False`; rewrite the `helpers.py:223` comment (no legacy text). Rollout buffers are statically padded `[num_envs, max_steps]` + `valid_mask` (`rollout_buffer.py:223-244`), LSTM `hidden=512, chunk=150` fixed → no real shape variation; `dynamic=False` lets Inductor specialize+fuse.
- **Deps:** inert until compile is on (P3-GATE). Must precede gate-narrowing.
- **Validation:** `TORCH_LOGS="recompiles,graph_breaks" ... --devices cuda:0 --force-compile`; no shape-driven recompiles on `_train_step_impl`/policy after warmup (recompiles only at seed-stage transitions). `grep "size mismatch"` absent steady-state.
- **Rollback:** revert four one-token edits.
- **Risk:** low-medium; a ragged tail recompiles once and caches. No correctness/telemetry impact. **(MED-3 determinism: cleared** — neutral vs existing compile baseline.)

#### P3-GATE — Compile-enablement strategy under `--sanctum`
- **Files:** gate `vectorized.py:973-976`; threading `:985/:1043/:1068`; wiring `train.py:925-926,557,375`.
- **Order (each a gate):**
  1. Land **P1 (sync removal + `validate=False`)** + **P3-DYN** + **P0-ALLOC** first. Until then the gate is correctly off.
  2. **Validate WITHOUT `--sanctum`** (option b: `quiet_analytics=False` → full compile). Clean signal; burn-in. Confirm single-graph steady state.
  3. **Validate `--sanctum --force-compile`** (option c) — proves compile survives the interactive path (emitters, governor snapshots, telemetry callbacks).
  4. **Only after (b)+(c) green, narrow the gate** — split policy vs host:
     ```python
     policy_compile_mode = compile_mode
     host_compile_mode = compile_mode if not quiet_analytics else "off"
     if not force_compile and quiet_analytics:
         host_compile_mode = "off"   # host eager under TUI; policy compiles
     ```
     Thread `policy_compile_mode` into the agent compile (`:985/:1043/:1068`), `host_compile_mode` into `EnvFactoryContext` (P3-HOST). Keep `--force-compile` as the host-under-TUI override.
- **Deps:** P1 + P3-DYN before step 4; P3-HOST depends on `host_compile_mode`.
- **Validation:** (b) `compile_enabled` telemetry (`vectorized.py:1117-1120`) shows compile on; (c) same under TUI; step 4: policy `compile_enabled=True`, host absent from `TORCH_LOGS="graph_breaks"`. **(W5):** extend `TrainingStartedPayload` to carry **both** policy and host compile status so the Karn panel doesn't misreport host after the split.
- **Rollback:** restore the single `effective_compile_mode` ternary; drop the split vars.
- **Risk:** medium; narrowing before P1 syncs are gone re-exposes the TUI to compile crashes — steps 2-3 gate it.

#### P3-HOST — Compile host `MorphogeneticModel`; fence counterfactual `fused_forward`
- **Files:** model finalize `env_factory.py:152-158`; `EnvFactoryContext` `:129`; context build `vectorized.py:1305-1324`; host fwd `batch_ops.py:185,331`; counterfactual `counterfactual_eval.py:75` → `host.py:803`; slot branches `slot.py:2096,2123-2144`.
- **Change:**
  1. Add `host_compile_mode: str = "off"` to `EnvFactoryContext`.
  2. After channels_last (`:158`):
     ```python
     if context.host_compile_mode != "off":
         model = torch.compile(model, mode=context.host_compile_mode, dynamic=False)
     ```
  3. **Fence the counterfactual path** (corrected: `fused_forward`, NOT `force_alpha`) at `host.py:803`:
     ```python
     @torch.compiler.disable
     def fused_forward(self, x, alpha_overrides): ...
     ```
  4. Pass `host_compile_mode=host_compile_mode,` at `vectorized.py:1324`.
  - **Mode:** default to `mode="default"` (no CUDA graphs) while `expandable_segments` is on. `reduce-overhead` is **opt-in only** with `expandable_segments` disabled (STOP-HOST). Document the incompatibility inline (R2): a comment at the `torch.compile(...)` site stating reduce-overhead + expandable_segments must not coexist.
- **Deps:** P3-GATE (`host_compile_mode`), P3-DYN. **P0-ALLOC interaction is the hard gate** (STOP-HOST).
- **Validation:** `TORCH_LOGS="graph_breaks,recompiles" ... --force-compile`: single steady-state host graph per slot-stage; `Recompiling function forward` correlates only with seed-stage transitions; **`fused_forward` never appears in any inductor/recompile line**; no graph_break inside `forward` (fix at source if any). Counterfactual val-loss bitwise-stable (eager either way). **(MED-2):** capture `seed_gradient_ratio` distribution for seeds within ±10% of `DEFAULT_GRADIENT_RATIO_THRESHOLD` across eager vs compiled-host; **assert no G2 gate-decision flips**; if flips occur, upcast grads to FP32 before the norm in `isolation.py` (`torch._foreach_norm`). Verify seed_gradient_ratio / gradient-health panels still populate under compile (fence the telemetry callback if it graph-breaks — **do not** strip telemetry).
- **Rollback:** `host_compile_mode="off"` (default); remove the wrap + `@torch.compiler.disable`.
- **Risk:** medium-high. CUDA-graph/fragmentation interaction (gated); stage-transition recompiles (bounded, epoch-cadenced); G2 gate-flap (MED-2 test + FP32 norm fallback).

#### P3-CLONE — Stream-local data clone (drop per-batch default-stream `wait_stream`)
- **Files:** `utils/data.py:583-591` (clone + comment); consumers `vectorized_trainer.py:589-595` and `:880-900` (`wait_stream` `:594/:892`, `record_stream` `:899-900`).
- **Change:**
  1. `data.py:591`: return the split view `result[global_env_idx] = (inp, tgt)`; update the `:583-588` comment (clone now happens stream-local at the consumer — delete the "eliminates consumer-side cloning" claim).
  2. Consumer (both loops) — clone inside the env stream, drop the `wait_stream`:
     ```python
     inputs, targets = env_batches[i]
     if env_state.stream:
         # (STOP-CLONE / R5) mark the source views in-use on this stream BEFORE the async clone,
         # so the allocator can't recycle the DataLoader batch buffer mid-copy.
         inputs.record_stream(env_state.stream)
         targets.record_stream(env_state.stream)
         with torch.cuda.stream(env_state.stream):
             inputs = inputs.clone()
             targets = targets.clone()
             inputs.record_stream(env_state.stream)
             targets.record_stream(env_state.stream)
     else:
         inputs = inputs.clone()
         targets = targets.clone()
     ```
- **Deps:** none hard. **Coordinate edit region with P2-DEL/P2-FRAGMETRIC** (same `vectorized_trainer.py` loop) — land P3-CLONE after Phase 2 to avoid merge conflicts.
- **Validation:** `nll_loss`/class-range assertion still passes (aliasing break preserved, relocated). Profiler: per-batch default-stream `wait_stream` gaps disappear; env-stream kernels overlap. Val-loss parity for fixed seed. **(STOP-CLONE):** run under load / multi-worker DataLoader and confirm no intermittent wrong-target corruption.
- **Rollback:** restore eager clone at `data.py:591` + the `wait_stream`/`record_stream` blocks.
- **Risk:** **medium — source-view lifetime.** The pre-clone `record_stream` on the **raw views** (R5) is mandatory: without it the DataLoader's next prefetch can overwrite the source before the async clone completes (the exact corruption the old in-iterator clone prevented). Mirror `record_stream` discipline exactly.

#### P3-SNAP — Pinned + `non_blocking` governor snapshot offload
- **Files:** `governor.py:282-292` (offload), `:288` (`v.detach().cpu()`), `device` `:219-221`, contract `:214-231`, rollback `load_state_dict` `~:458`.
- **Change:**
  1. `__init__`: `self._pinned_snapshot: dict[str, torch.Tensor] = {}`.
  2. Rewrite the offload — reuse a persistent pinned buffer per key, `non_blocking=True`, **then one stream sync BEFORE exposing `last_good_state`**:
     ```python
     with torch.no_grad():
         new_state: dict[str, Any] = {}
         for k, v in filtered_state.items():
             if isinstance(v, torch.Tensor):
                 if v.device.type == "cpu":
                     new_state[k] = v.detach().clone(); continue
                 buf = self._pinned_snapshot.get(k)              # authorized owned-cache read
                 if buf is None or buf.shape != v.shape or buf.dtype != v.dtype:
                     buf = torch.empty_like(v, device="cpu", pin_memory=True)
                     self._pinned_snapshot[k] = buf
                 buf.copy_(v.detach(), non_blocking=True)
                 new_state[k] = buf
             else:
                 new_state[k] = copy.deepcopy(v)
         if device.type == "cuda":
             torch.cuda.current_stream(device).synchronize()    # MUST be post-copy, pre-assignment
         self.last_good_state = new_state
     ```
- **Deps:** none. Respect the stream contract (the explicit sync satisfies "completes on default stream"). Coordinate with P3-HOST (snapshot runs at epoch boundary, outside any captured region).
- **Validation:** `num_alloc_retries`/pageable-host churn drop; profiler shows `Memcpy DtoH` overlapping default-stream compute. Rollback parity: `snapshot()` then `restore()` → restored `state_dict` bitwise-equal to pre-snapshot weights; `last_good_state` tensors `is_pinned()` on CPU. Contract `RuntimeError` at `:226` still fires for non-default-stream calls.
- **Rollback:** restore the dict-comp; delete `self._pinned_snapshot`.
- **Risk:** medium. **(STOP-SNAP):** `last_good_state` aliases the live pinned buffers; the next snapshot's `copy_` overwrites them. The training loop is single-threaded (snapshot and rollback can't interleave today), and the **post-copy/pre-assignment sync placement resolves the `non_blocking` race** — **this exact ordering is a mandatory code-review checkpoint.** If the loop ever becomes async, ping-pong two buffers per key. Pinned host-RAM: one host+fossilized state per env (bounded).

**Phase 3 landing order:** P3-DYN → (validate b, then c) → P3-GATE step 4 → P3-HOST (gated on STOP-HOST). P3-CLONE and P3-SNAP independent; land after Phase 2 (region coordination), P3-SNAP after the checkpoint review.

---

## 4. Measurement & Verification Harness

**Immediate, no-code A/B the user can run now (the headline memory lever as an env-var prepend):**
```
# BASELINE (current):
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --devices cuda:0 cuda:1 --telemetry-level debug --sanctum \
  --config-json configs/config-3slot-3seed-baseline-shaped.json

# WITH ALLOCATOR FIX (proves P0-ALLOC before touching code):
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8" \
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --devices cuda:0 cuda:1 --telemetry-level debug --sanctum \
  --config-json configs/config-3slot-3seed-baseline-shaped.json
```
Compare `frag_gap`/`num_alloc_retries` (P0-FRAGPROBE) across ~20 batches per card.

**Probes / telemetry:**
- `log_frag(device, tag)` (P0-FRAGPROBE) — per-batch start/end, per card. `frag_gap = reserved − allocated`.
- `on_allocator_stats` (P2-FRAGMETRIC) — `fragmentation_bytes`, `num_alloc_retries`, `num_ooms` into Karn `raw_events` (typed `leyline` event + DuckDB view).

**TORCH_LOGS:**
- Recompiles/breaks: `TORCH_LOGS="recompiles,graph_breaks"` (P3-DYN, P3-HOST).
- Autocast cast verification: `TORCH_LOGS=+autocast` (P1-BF16; confirm bf16 casts, no GradScaler).

**Parity / regression tests (must run in SHIPPING precision unless noted):**
| Test | Cluster | Gate |
|---|---|---|
| `test_evaluate_actions_parity` (FP32 algebra, 1e-5) | P1-EVAL V1 | blocks P1-BF16 |
| Floor parity, **BF16 call-site** | P1-EVAL V1b | HIGH-1 |
| Cross-path log_prob (rollout vs update, 1e-5) | P1-EVAL V2 | — |
| Entropy parity: 1-valid edge + **2-valid floor-active**, FP32+BF16 | P1-EVAL V3 | HIGH-2 |
| **Epoch-0 frozen-weight `joint_ratio≈1`, BF16 config** | P1-BF16 V0 | **blocks full-BF16-LSTM** |
| BF16 vs FP32 metric parity (rel tolerances) | P1-BF16 V1 | — |
| 50-update trajectory ±2σ | P1-BF16 V2 | — |
| Single-head NaN injection → byte-identical attribution | P1-SYNC | telemetry-sacred |
| Forced-step integer equality | P1-VEC | telemetry |
| Q-loop `assert_close` | P1-QLOOP | telemetry |
| Bit-identical `val_acc` same seed | P2-STREAMPOOL/P2-DEL | numerics-neutral |
| `CUDA_LAUNCH_BLOCKING=1` short run, no illegal access | P2-DEL | fence proof |
| Per-step action bit-parity over ≥1 episode | P2-RESET | determinism (deferred) |
| Snapshot→restore bitwise-equal; tensors pinned | P3-SNAP | rollback safety |
| `nll_loss` range OK under multi-worker load | P3-CLONE | corruption |
| G2 gate no-flip eager vs compiled-host near threshold | P3-HOST | MED-2 |

**A/B protocol:** for each phase, run baseline → apply phase → run with same `--seed`, compare (1) `frag_gap`/retries trend, (2) mean batch wall-clock over 20 batches, (3) Karn `ppo_updates` (`approx_kl`, `clip_fraction`, `explained_variance`) within stated tolerance, (4) `seed_gradient_ratio`/`lstm_has_nan` panels healthy.

---

## 5. Risk Register

| Risk | Sev | Likelihood | Mitigation | Rollback unit |
|---|---|---|---|---|
| **CRITICAL-1:** BF16 update + FP32 rollout → biased importance ratio | Critical | Certain if unaddressed | BF16-wrap rollout `get_action` (`vectorized_trainer.py:1474`); V0 epoch-0 ratio≈1 gate; fallback FP32-LSTM | P1-BF16 |
| **STOP-HOST:** `reduce-overhead` CUDA graphs + `expandable_segments` VA remap | High | Probable if both on | Mutually exclusive; default host `mode="default"`; inline incompat comment | P3-HOST / P0-ALLOC |
| **STOP-CLONE:** source view overwritten before async clone | High | Possible under load | `record_stream` raw views before clone (R5) | P3-CLONE |
| **STOP-SNAP:** pinned buffer aliasing overwrites mid-rollback | High (if async future) | Low today (single-thread) | Post-copy/pre-assignment sync (review checkpoint); ping-pong if async | P3-SNAP |
| Floor/entropy drift between two copies | High | Likely over time | Unify floor to one helper IN SCOPE (P1-EVAL); V1b/V3 BF16 tests | P1-EVAL |
| `_logger` NameError (P0-FRAGPROBE) | High | Certain if unfixed | Add `_logger = logging.getLogger(__name__)` | P0-FRAGPROBE |
| `valid_mask` used before def (P1-VEC) | High | Certain if unfixed | Reorder `data=`/`valid_mask` up or derive from `step_counts` | P1-VEC |
| BF16 LSTM 150-step divergence/NaN | Medium | Possible (HW-dep) | clamp `tanh(c/50)*50`; `lstm_has_nan`/`lstm_c_rms` watch; FP32-LSTM fallback | P1-BF16 |
| P0-ALLOC ignored (CUDA pre-init) | Medium | Low | `is_initialized()` warning (R3) | P0-ALLOC |
| `on_allocator_stats` silently dropped (`Any` emitter / unwired view) | Medium | Possible | Typed emitter + mypy CI + Karn view in same change | P2-FRAGMETRIC |
| G2 gate-flap under compiled host (MED-2) | Medium | Possible near threshold | No-flip test; FP32 grad-norm fallback | P3-HOST |
| Reserved high-water climbs across many seeds (P0-EMPTYCACHE) | Medium | Low | `gc_threshold:0.8` reclaim; monitor P2-FRAGMETRIC | P0-EMPTYCACHE |
| P2-RESET state leak / RNG drift | High | — (deferred) | RNG-state snapshot/restore; per-step-action parity; real prune path | P2-RESET |
| Empty mask undetected after `validate=False` | Medium | Low | finiteness gate catches; validation ON in eval/CI | P1-VALID |
| Stage-transition recompiles churn (P3-HOST) | Low-Med | Low | epoch-cadenced, bounded/cached | P3-HOST |
| TF32 perturbs convergence | Low | Low | ~1e-3; revert two lines | P0-TF32 |

---

## 6. Go/No-Go Gates Per Phase

- **Phase 0 → 1:** `expandable_segments` env string confirmed inherited; `frag_gap`/`num_alloc_retries` stop growing monotonically over 20 batches; no OOM; PPO `kl`/`clip_fraction`/`value_loss` within run-to-run noise. P0-FRAGPROBE emitting on both cards.
- **Phase 1 internal gates:** P1-EVAL **V1 (1e-5) must pass before P1-BF16**. P1-BF16 **V0 (epoch-0 joint_ratio≈1 in BF16) must pass** before promoting full-BF16-LSTM (else ship FP32-LSTM fallback). P1-SYNC single-head NaN test byte-identical. → Proceed when no KL explosion, `approx_kl` in range, `seed_gradient_ratio` populates, `lstm_has_nan=False` over 50 updates.
- **Phase 2 → 3 (and P2-RESET decision):** P2-FRAGMETRIC `fragmentation_bytes/reserved_bytes < 0.15` and retries flat and no OOMs over ≥20 batches → NO-GO on P2-RESET, proceed to Phase 3. If ratio > 0.30 / retries climbing / any OOM → GO on P2-RESET first. Bit-identical `val_acc`; `CUDA_LAUNCH_BLOCKING` run clean.
- **Phase 3 gates:** P3-DYN — no shape-driven recompiles steady-state. P3-GATE — compile validated **without** `--sanctum` (b) **and** `--force-compile` (c) **before** narrowing the gate. P3-HOST — single host graph per slot-stage, `fused_forward` never compiled, **G2 no-flip**, panels populate; **`expandable_segments` disabled iff `reduce-overhead` enabled**. P3-CLONE — no `nll_loss` corruption under load. P3-SNAP — sync placement review checkpoint passed; restore bitwise-equal.

---

## 7. Open Questions / Info Gaps (need a live run or quick check to close)

1. **PyTorch/CUDA version in lockfile** — confirm `expandable_segments` stability and the `reduce-overhead`+expandable incompatibility behavior for the pinned version (`uv pip show torch` / `pyproject.toml` / `uv.lock`). Drives STOP-HOST posture.
2. **Transitive CUDA init at `train.py` import** — `import torch` is lazy, but `from esper.nissa…`, `from esper.simic.training…` were not traced for module-scope `torch.cuda.*`. The R3 `is_initialized()` warning makes a miss loud; confirm it never fires on a real run.
3. **`batch_emitter` concrete type** — trace `emitters` construction `vectorized.py:940-965` to add `on_allocator_stats` on the right class (typed `Any` at `:235`).
4. **`get_batched_sequences` padding of `forced_actions`** — confirm shape matches `valid_mask` (P1-VEC); `& valid_mask` is the safeguard.
5. **`_compute_value` batch-dim support** — read before P1-QLOOP; add `.contiguous()` on the expanded view.
6. **DataLoader prefetch buffer reuse** (`num_workers`, pin) — determines the P3-CLONE source-view window severity; R5 `record_stream` is the fix regardless.
7. **P2-RESET governor re-snapshot site** (`env_factory.py:243`) — confirm before implementing the deferred reset.
8. **`slot.py:6-13` graph-count docstring** — non-blocking; informs P3-HOST recompile expectations.
