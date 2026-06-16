# Multi-Epoch Recurrent PPO via Anchored Reference Pass — FINAL Executable TDD Plan

| | |
|---|---|
| **Status** | READY TO EXECUTE (go) — all reviewer blockers/majors merged in |
| **Source spec** | `docs/superpowers/specs/2026-06-17-recurrent-ppo-multiepoch-design.md` (Approved design) |
| **Date** | 2026-06-17 |
| **Domain** | RL training (simic PPO controller) |
| **Reviewers** | Reality, Architecture, Quality, Systems (4/4 `approve-with-changes`; all `must_fix` resolved below) |
| **Discipline** | Strict TDD (RED→GREEN), no-legacy single path. Every behavior-changing step states the failing test, the expected RED, the change, and the GREEN command. |
| **Structure** | TWO PRs per the spec's locked decisions. PR1 lands the anchored-reference-pass at K=1 (zero behavior change, AMP grad-gate proof). PR2 plumbs `recurrent_n_epochs`, replaces the guard with the single-path assertion, flips K=4, regenerates goldens, runs the EV-lift validation. |

## Verified code anchors (line numbers re-verified 2026-06-17; authoritative over spec where drifted)

| Anchor | Location | Role |
|--------|----------|------|
| Epoch loop | `ppo_agent.py:863` `for epoch_i in range(self.recurrent_n_epochs):` | Internal K loop |
| Scored forward | `ppo_agent.py:898-906` `evaluate_actions(..., hidden=(data["initial_hidden_h"], data["initial_hidden_c"]), ...)` | Full-unroll TBPTT from θ₀ |
| GAE | `ppo_agent.py:547-551` | Once per update(), pre-loop |
| Normalizer update | `ppo_agent.py:638` `value_normalizer.update(valid_returns)` | Once per θ₀, pre-loop |
| EV (pre-update) | `ppo_agent.py:607-615` | Acceptance #5 evidence source |
| Telemetry no_grad forward | `ppo_agent.py:763-778` (`autocast(enabled=False)` + `no_grad`) | **Hazard context — see Step 1.2 sequencing note** |
| **old_log_probs source (fast path)** | `ppo_agent.py:1058-1067` (from rollout `data`) | **Swap target #1 (PR1)** |
| **Finiteness-gate `old_lp_stack`** | `ppo_agent.py:946-949` (from rollout `data`) | **Swap target #2 (PR1)** |
| **Finiteness slow-path drill-down** | `ppo_agent.py:969-976` (`data[old_key][valid_mask]`) | **Swap target #3 (PR1) — was omitted in draft** |
| **Ratio-explosion diagnostic** | `ppo_agent.py:1371` (`old_log_probs["op"].flatten()`) | **Covered by dict-level swap; assert in test (PR1)** |
| `head_grad_norms` metric | `ppo_metrics.py:131-132` — nested dict `metrics["head_grad_norms"][head]` | **Correct key for Step 1.3 (draft used wrong key)** |
| `valid_old_values` | `ppo_agent.py:1107` (dead under `clip_value=False`) | Do NOT touch |
| KL early-stop | `ppo_agent.py:1095-1103` | Breaks loop pre-loss |
| `early_stop_epoch` telemetry | `ppo_agent.py:1097` | Counts epochs that RAN, not wall epochs |
| train_steps++ | `ppo_agent.py:1417-1422` (once per update()) | Schedule tick |
| `total_train_steps` | `ppo_agent.py:158,228` (default 1_000_000, NEVER plumbed) | **See Step 2.2b (Systems major)** |
| ctor `recurrent_n_epochs` | `ppo_agent.py:142` (default None→1 at `:200`) | Plumb target (PR2) |
| C4 warning | `ppo_agent.py:188-200` | Rewrite (PR2) |
| H5 runtime warnings | `ppo_agent.py:251-272` | Delete/rewrite (PR2) |
| Checkpoint save | `ppo_agent.py:1460,1467` (`recurrent_n_epochs`/`total_train_steps` already persisted) | Verify round-trip (PR2) |
| Checkpoint load | `ppo_agent.py:1661-1666` (`**agent_config` spread) | Verify round-trip + old-ckpt safety (PR2) |
| Guard (to rewrite) | `vectorized.py:418-424` | Replace message with single-path assertion (PR2) |
| Agent ctor call | `vectorized.py:1090-1111` (omits `recurrent_n_epochs`) | Pass param (PR2) |
| `clip_value` default | `ppo_agent.py:131` (False) | Stays False |
| Config field | `config.py:71` `ppo_updates_per_batch: int = 1` | Add sibling `recurrent_n_epochs` (PR2) |
| Multiplier no-ops | `config.py:301,304` | Document (PR2) |
| `to_ppo_kwargs` | `config.py:297-323` | Thread param + test (PR2) — TEST-ONLY consumer |
| `to_train_kwargs` | `config.py:325-381` (ends 381, not 358) | Thread param (PR2) — PRODUCTION consumer |
| entrypoint | `vectorized.py:631` `train_ppo_vectorized(...)`; `phase_profiler:706` | Add param (PR2) |
| Golden test | `test_ppo_update_golden.py:45,123-140` (`log_prob_offset=0.1`, hardcoded ratios) | Regenerate (PR2 K=4; PR1 K=1) |
| Guard test | `test_vectorized.py:1057-1083` | Rewrite (PR2) |

> **Plumbing facts (verified):** `to_ppo_kwargs` is consumed ONLY by `tests/simic/test_config.py`. The production path is `config.to_train_kwargs()` unpacked into `train_ppo_vectorized`, which builds the agent ctor at `vectorized.py:1090` from explicit named args. Therefore the field can pass config tests while never reaching the production agent — **Step 2.2's `agent.recurrent_n_epochs == 4` assertion is the real production gate, and Steps 2.1+2.2 MUST land as ONE atomic commit** (a half-landed `to_train_kwargs` key crashes `TrainingConfig.run()`).

---

## PR1 — Anchored Reference Pass at K=1 (zero behavior change)

**Goal:** Insert a frozen-θ₀ anchor forward under the inherited BF16 `policy_amp_context`; source `old_log_probs`, the finiteness-gate stack, AND the slow-path drill-down from `ref_log_probs`. At K=1 the anchor and the epoch-0 scored forward share an identical full-unroll at θ₀, so epoch-0 ratio==1.0 within FP tolerance and approx_kl==0. Existing K=1 callers see no behavior change. The AMP nonzero-per-head-grad gate is the load-bearing safety test.

**Files:** `src/esper/simic/agent/ppo_agent.py`; `tests/simic/test_anchor_reference_pass.py` (new); `tests/simic/test_ppo_update_golden.py`.

### Step 1.1 — RED: epoch-0 soundness proof (Acceptance #1)
- **New test** `tests/simic/test_anchor_reference_pass.py::test_epoch0_ratio_is_one_at_theta0`: build a K=1 LSTM `PPOAgent` (CPU, `target_kl=None`), fill the buffer via the golden helper pattern (`_fill_buffer`, **NO** `log_prob_offset` — store real rollout log_probs), call `agent.update()`, assert `metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-5)`, `ratio_max`/`ratio_min` within `1e-5` of 1.0, `metrics["approx_kl"] == pytest.approx(0.0, abs=1e-6)`.
- **Expected RED:** today `old_log_probs` comes from the sampling rollout forward, not the unroll forward, so `ratio_mean != 1.0`. (Spec acceptance #1: `|ratio−1|<eps`, not bitwise; `no_grad`, never `inference_mode`.)

### Step 1.2 — GREEN: implement the anchor forward + baseline swap (ALL FOUR read sites)
- In `ppo_agent.py.update()`, after the normalizer block (`:638`) and before the epoch loop (`:863`), add ONE `torch.no_grad()` anchor forward calling the SAME `evaluate_actions(data["states"], data["blueprint_indices"], actions, masks, hidden=(data["initial_hidden_h"], data["initial_hidden_c"]), probability_floor=..., aux_stop_gradient=...)` as the loop, producing `ref_log_probs = {head: result.log_prob[head].detach() for head in HEAD_NAMES}`.
- **Swap all four read sites** (draft only listed two; Quality+Systems blocker):
  1. `old_log_probs` (`:1058-1067`) → read each head from `ref_log_probs`.
  2. finiteness-gate `old_lp_stack` (`:946-949`) → stack from `ref_log_probs`.
  3. **finiteness slow-path drill-down (`:969-976`)** → read `ref_log_probs[key]` instead of `data[old_key][valid_mask]`, and change the attribution string from `old_log_probs[{key}]` to `ref_log_probs[{key}]` so a NaN in the anchor is correctly attributed, not silently blamed on rollout/values.
  4. **ratio-explosion diagnostic (`:1371`)** is covered transitively because it reads the `old_log_probs` dict — but Step 1.4 must assert this explicitly.
- **CRITICAL — AMP sequencing (Architecture + Quality blocker):** The telemetry forward at `:763-778` runs with `autocast(enabled=False)`, which evicts it from the BF16 cast-weight cache. The anchor, placed between `:638` and `:863`, is therefore the **first BF16-autocast forward touching every head Linear** — exactly the hazard documented at `:763-778`. The mitigation is NOT `autocast(enabled=False)`: run the anchor **INSIDE** the inherited BF16 `policy_amp_context` so it primes the cast-weight cache under grad-capable conditions exactly as the epoch-0 scored forward would, preserving the FP32 masked-logits seam (`factored_lstm.py:1365-1370`) that keeps `ref_log_probs` FP32-symmetric. Use `torch.no_grad()`, NEVER `inference_mode()`. Add an inline comment contrasting with the telemetry forward: *"UNLIKE the telemetry forward at :763-778 (autocast disabled to stay OUT of the cache before backward), the anchor MUST run under BF16 autocast: it is the first cache touch before the epoch loop and must prime the cache identically to the epoch-0 forward."* **Step 1.3 is a DISCOVERY gate, not a regression lock:** if it fails RED on CUDA after this step, the anchor approach is unsound under BF16 AMP and the plan must be revised before merge.
- **Do NOT touch** `valid_old_values` (`:1107`) — dead under `clip_value=False`.
- **GREEN:** Step 1.1 passes. Command: `uv run pytest tests/simic/test_anchor_reference_pass.py::test_epoch0_ratio_is_one_at_theta0 -q`.

### Step 1.3 — RED→GREEN: AMP nonzero-per-head-grad gate (Acceptance #2, LOAD-BEARING)
- **New test** `test_anchor_reference_pass.py::test_epoch0_per_head_grad_norms_nonzero_under_amp`: skip if no CUDA. Build a CUDA LSTM agent, fill buffer, run `agent.update()` under `policy_amp_context(amp_enabled=True, resolved_amp_dtype=torch.bfloat16)` — **note exact arg names/types** (`helpers.py:57`): pass `torch.bfloat16` (the dtype object), NOT the string `"bfloat16"` (silently returns `nullcontext()` → no-op gate). Assert, following the verified pattern at `test_ppo.py:526-534`:
  ```python
  head_grad_norms = metrics["head_grad_norms"]  # nested dict, NOT metrics["head_{name}_grad_norm"]
  for head_name, norms in head_grad_norms.items():
      assert len(norms) > 0 and all(n > 0 for n in norms), f"{head_name} grad norm is zero"
  ```
  (Reality + Quality blocker: the draft's `metrics["head_{name}_grad_norm"]` key does NOT exist and would KeyError, neutralizing the gate.)
- **Discovery RED check:** before trusting GREEN, locally (not committed) wrap the anchor in `autocast(enabled=False)` and confirm a zero-grad RED, proving the gate discriminates cache poisoning. Then revert to the correct in-context form for GREEN.
- **Command:** `uv run pytest tests/simic/test_anchor_reference_pass.py::test_epoch0_per_head_grad_norms_nonzero_under_amp -q` (CUDA host).

### Step 1.4 — RED: no stale-baseline consumption + diagnostic coverage (Acceptance #4)
- **New test** `test_anchor_reference_pass.py::test_loss_path_reads_only_ref_log_probs`: after `get_buffer()` but before the epoch loop, sentinel-poison the rollout per-head log_probs (`data[f"{k}_log_probs"]` → NaN) via a monkeypatched/wrapped buffer return; run `update()`; assert `ratio_mean ≈ 1.0` and all losses finite (proving the loss path read `ref_log_probs`, not the poisoned rollout). **Add a control group** (poisoned data, anchor NOT applied / pre-swap baseline) that DOES produce a NaN ratio, to confirm the test discriminates (Systems nice-to-have). Add a second assertion that the ratio-explosion diagnostic path (`:1371`) and finiteness slow-path (`:969-976`) read `ref_log_probs` (poison the rollout, force a ratio-explosion via crafted advantages, assert the emitted diagnostic does not surface the poisoned NaN as `old_log_probs`).
- **Also assert** `data["hidden_h"][:,1:]` is referenced only by the telemetry/diagnostic section (`:728-758`), never the loss path. Supplement the runtime test with an inline code comment at the loss-path region: *"data[hidden_h][:,1:] is never indexed here; only data[initial_hidden_h] is used."*
- **Expected RED before 1.2, GREEN after.** Command: `uv run pytest tests/simic/test_anchor_reference_pass.py::test_loss_path_reads_only_ref_log_probs -q`.

### Step 1.5 — RED: K=1 behavior-invariance for existing callers (Acceptance #9 at K=1)
- **New test** `test_anchor_reference_pass.py::test_k1_finiteness_and_update_performed_unchanged`: assert `metrics["ppo_update_performed"] is True`, no finiteness-gate failures recorded, and `value_loss`/`return_mean`/`return_std` finite.
- **Expected GREEN:** value path untouched; anchor consumes no RNG (`ResidualLSTM` dropout=0.0; contribution `Dropout(0.1)` is aux-only/detached — `factored_lstm.py:359-365,470-478`). For determinism safety, set `enable_contribution_aux=False` in test agents (or confirm the anchor's `no_grad` + detached aux suppresses Dropout influence on main log_probs).

### Step 1.6 — K=1 golden test reflects the anchor; resolve the liveness signal (Acceptance #8, partial)
- `test_ppo_update_golden.py` currently injects `log_prob_offset=0.1` (`:125`) to force ratio≠1 as a liveness proof; under the anchor, epoch-0 ratio==1.0 regardless of the offset, so the offset becomes **vestigial dead code** and the existing hardcoded goldens (`ratio_mean==0.4489...`, `clip_fraction==1.0`, `:136-140`) become invalid.
- **Resolution (Architecture major):** **Remove `log_prob_offset` from `_fill_buffer` entirely** (no-legacy: it is now dead). Capture the RED (old goldens fail), regenerate the K=1 goldens (`ratio_*`, `policy_loss`, `approx_kl`), and add a comment: *"epoch-0 ratio==1.0 by construction (anchored reference pass) — this is the correct non-trivial assertion, not a tautology; the ratio path's liveness at epoch>0 is proven by the K=4 goldens in PR2."* Full K=4 regeneration with the documenting comment completes in PR2 Step 2.6.
- **Command:** `uv run pytest tests/simic/test_ppo_update_golden.py -q`.

### Step 1.7 — Determinism/replay at K=1 (precursor to Acceptance #3)
- **New test** `test_anchor_reference_pass.py::test_two_identical_seed_updates_match_k1`: seed (`torch.manual_seed`), build agent A, fill buffer, `update()`; repeat identically for agent B; assert all scalar metrics (`policy_loss`, `value_loss`, `approx_kl`, `ratio_mean`, `explained_variance`) match `<1e-6`. **Pin to CPU/FP32** (Quality nice-to-have: BF16 mantissa makes bitwise determinism unreliable; if a GPU variant is added, widen tolerance and document).
- **Command:** `uv run pytest tests/simic/test_anchor_reference_pass.py::test_two_identical_seed_updates_match_k1 -q`.

### Step 1.8 — Full PR1 verification gate
- `uv run pytest tests/simic/test_anchor_reference_pass.py tests/simic/test_ppo_update_golden.py tests/simic/test_policy_amp_context.py -q`
- Regression set: `uv run pytest tests/simic/test_ppo.py tests/simic/test_vectorized.py -q` (the `vectorized.py:418` guard is UNCHANGED in PR1 — `ppo_updates_per_batch>1`+LSTM still raises).
- Confirm no behavior change for K=1 production paths; `recurrent_n_epochs` stays pinned to 1 everywhere (ctor default unchanged).

**PR1 rollback note:** PR1 is a single self-contained change to `ppo_agent.py.update()` plus new tests and a golden regeneration. Revert = revert the anchor insertion and restore the four rollout-sourced read sites and the K=1 goldens (git revert of the PR commit). No config/schema/checkpoint surface is touched, so no migration concerns; K=1 production behavior is byte-equivalent pre/post, so a revert cannot strand any in-flight run.

---

## PR2 — Plumb `recurrent_n_epochs`, replace guard, flip K=4, regenerate, validate

**Goal:** Single no-legacy path: plumb K through config→run()→ctor→checkpoint; replace the external-loop guard message with the `ppo_updates_per_batch==1`-when-LSTM single-path assertion; rewrite C4/H5 to the anchored invariant; set K=4; regenerate goldens under K>1; prove EV lifts off 0.

**Files:** `config.py`, `vectorized.py`, `ppo_agent.py`, `test_config.py`, `test_vectorized.py`, `test_ppo.py`, `test_ppo_checkpoint.py`, `test_ppo_update_golden.py`, `test_anchor_reference_pass.py`; `CLAUDE.md` (telemetry-hidden-buffer note per spec §4).

### Step 2.1 + 2.2 — RED→GREEN: config field + kwargs threading + entrypoint + ctor (ONE ATOMIC COMMIT)
> **Architecture + Systems blocker:** these were two sequential RED/GREEN cycles in the draft, leaving a window where `to_train_kwargs()` emits a kwarg `train_ppo_vectorized` rejects, crashing `TrainingConfig.run()`. Collapse into ONE commit. Write BOTH tests first (RED), then green all four changes together.
- **Tests (both RED first):**
  - `test_config.py::test_recurrent_n_epochs_threads_through_kwargs`: assert `TrainingConfig(recurrent_n_epochs=4).to_ppo_kwargs()["recurrent_n_epochs"] == 4` and `.to_train_kwargs()["recurrent_n_epochs"] == 4`; **extend the existing `inspect.signature` drift tests** (`test_config.py:120-136`) rather than duplicating — they are the atomicity enforcer and will stay RED until all four changes land.
  - `test_vectorized.py::test_recurrent_n_epochs_reaches_agent`: call `train_ppo_vectorized(..., recurrent_n_epochs=4, lstm_hidden_dim=16, ppo_updates_per_batch=1, n_episodes=1, ...)` (CPU smoke) and assert the constructed `agent.recurrent_n_epochs == 4` (capture via returned handle or a monkeypatched ctor spy). **This is the real production gate** — do NOT declare 2.1 green until this passes.
- **Changes (single commit):** (a) add `recurrent_n_epochs: int = 1` to `TrainingConfig` (`config.py`, beside `:71`); (b) add `"recurrent_n_epochs": self.recurrent_n_epochs` to `to_ppo_kwargs` (`:297-323`) and `to_train_kwargs` (insert near `:377`, the function ends at `:381` — NOT 358); (c) add `recurrent_n_epochs: int = 1` to `train_ppo_vectorized` signature (`vectorized.py:631`); (d) pass `recurrent_n_epochs=recurrent_n_epochs` into the `PPOAgent(...)` ctor call (`:1090-1111`). Add a comment at `config.py:301,304` documenting that the `*ppo_updates_per_batch` multipliers are no-ops for recurrent runs (K is within-batch; `ppo_updates_per_batch` pinned to 1 for LSTM).
- **Command:** `uv run pytest tests/simic/test_config.py tests/simic/test_vectorized.py::test_recurrent_n_epochs_reaches_agent -q`.

### Step 2.2b — Plumb `total_train_steps` OR document the limitation (Systems major)
> `total_train_steps` defaults to 1_000_000 at `ppo_agent.py:228`, is read by `_get_penalty_schedule` (`:1113`) and persisted in the checkpoint (`:1460`), but is NEVER plumbed through `config.py`/`vectorized.py`. At K=4 this makes the entropy-floor penalty schedule visibly stuck in the early-training 1.5x-boost band for all practical run lengths (a pre-existing debt that K=4 telemetry surfaces).
- **Decision (pick ONE, do not leave implicit):**
  - **(a) Plumb it** alongside `recurrent_n_epochs` in the SAME atomic surface: compute `total_train_steps = n_episodes * ppo_updates_per_batch` (reduces to `n_episodes` for LSTM), thread `to_train_kwargs` → `train_ppo_vectorized` signature → ctor call. Add `test_config.py::test_total_train_steps_threads_through` asserting the computed value reaches `agent.total_train_steps`. **Preferred** — it removes dead-default debt and isolates the K variable in Step 2.10.
  - **(b) If out of scope:** add `test_ppo.py::test_total_train_steps_default_is_documented_limitation` asserting `agent.total_train_steps == 1_000_000` and add a TODO comment `# TODO: [FUTURE FUNCTIONALITY] total_train_steps is not config-plumbed; schedule stays in early-training band for short runs` per the deferred-functionality convention.
- **Command:** `uv run pytest tests/simic/test_config.py -k total_train_steps -q` (option a) or `tests/simic/test_ppo.py -k total_train_steps -q` (option b).

### Step 2.3 — RED→GREEN: rewrite the guard to the single-path assertion (Acceptance #7)
- **Rewrite test** `test_vectorized.py:1057-1083` → `test_lstm_requires_ppo_updates_per_batch_one`: assert `ppo_updates_per_batch>1` with `lstm_hidden_dim>0` raises (match the new message). Keep the stub-agent pattern (no buffer.reset / normalizer.update reached).
- **Change:** keep the `if ppo_updates_per_batch > 1 and agent.lstm_hidden_dim > 0:` condition at `vectorized.py:418-424` (it IS the `==1`-when-LSTM assertion) and rewrite ONLY the `ValueError` message/rationale to state the anchored invariant: *"recurrent policies use the internal `recurrent_n_epochs` multi-epoch path; the external `ppo_updates_per_batch` loop is disallowed for LSTM because it would re-run GAE and mutate the EMA value normalizer K times per rollout."* Keep ONE location (`_run_ppo_updates`) so the stub-agent harness still applies. **Acknowledge limitation** (Architecture + Systems nice-to-have): this guards the `_run_ppo_updates` callsite, not a direct `PPOAgent(...).update()` call — note this in the rewritten C4 comment; do NOT add a ctor-level guard in this PR (would require threading `ppo_updates_per_batch` into the ctor — out of scope).
- **Command:** `uv run pytest tests/simic/test_vectorized.py::test_lstm_requires_ppo_updates_per_batch_one -q`.

### Step 2.4 — RED→GREEN: rewrite C4/H5 warnings to the anchored invariant (correct RED ordering)
> **Quality major:** the draft's single no-warning test cannot produce a RED before the change. Use a two-test structure so the deletion is regression-locked.
- **(a) Liveness/RED gate first** `test_ppo.py::test_staleness_warning_fires_before_delete`: with `warnings.catch_warnings(record=True)`, construct `PPOAgent(recurrent_n_epochs=4, clip_value=False, lstm...)` and assert the staleness `UserWarning` IS emitted (GREEN before change, confirms the warning exists at `:251-272`).
- **(b) Delete** the H5 runtime `warnings.warn` block (`:251-272`); confirm test (a) now goes RED, then **remove test (a)** (no-legacy).
- **(c) GREEN gate** `test_ppo.py::test_no_recurrent_staleness_warning_at_k4`: assert NO `UserWarning` about staleness on the same construction.
- **Rewrite the C4 comment** (`:188-200`) to the new invariant: scored forward is full-recompute TBPTT from θ₀; GAE + normalizer run once per update() pre-loop; the only live per-epoch baseline (`old_log_probs`) is anchored at θ₀ via the reference pass, so multi-epoch recurrent PPO is mathematically exact; `clip_value` stays False (re-enabling under K>1 needs an anchored `ref_values` — explicitly a separate task); `early_stop_epoch` counts epochs that RAN, not wall epochs (finiteness `continue` can desync them — see Step 2.8). Document the per-step hidden buffer (`rollout_buffer.py:439-440`) as telemetry-only (Q(s,op) diagnostic, `ppo_agent.py:728-758`) — do not delete; add the same note to `CLAUDE.md` per spec §4.
- **Command:** `uv run pytest tests/simic/test_ppo.py -k staleness -q`.

### Step 2.5 — RED: checkpoint round-trip preserves K + old-checkpoint safety (Architecture major)
- **New tests in** `test_ppo_checkpoint.py`:
  - `test_recurrent_n_epochs_survives_save_load`: build agent with `recurrent_n_epochs=4`, `save()`, `PPOAgent.load()`, assert `loaded.recurrent_n_epochs == 4`. Save already persists it (`:1467`); load spreads `**agent_config` (`:1661-1666`). If GREEN on first run, it is a REGRESSION-LOCK — run it BEFORE PR2 changes to establish baseline green, rerun at PR2 end.
  - `test_old_checkpoint_without_recurrent_n_epochs_loads_as_k1`: construct a checkpoint dict LACKING the `recurrent_n_epochs` key, `PPOAgent.load()`, assert `loaded.recurrent_n_epochs == 1`. Locks the None→1 default safety net for pre-PR2 checkpoints (backward-compatible *behavior*, not backward-compat *code* — permitted). Do NOT bump `CHECKPOINT_VERSION`; document that old checkpoints resume at K=1 as the intended silent default.
- **Command:** `uv run pytest tests/simic/test_ppo_checkpoint.py -k recurrent -q`.

### Step 2.6 — RED→GREEN: flip K=4, regenerate goldens under K>1 (Acceptance #8)
- **Edit** `test_ppo_update_golden.py`: **parametrize** over `recurrent_n_epochs ∈ {1, 4}` (Quality nice-to-have: parametrize, do not branch — keeps K=1 and K=4 side-by-side and regression-locks K=1 drift). With the anchor, epoch-0 ratio==1.0; epochs 1-3 measure true π_θk/π_θ0 drift (this is where ratio-path liveness is proven). Capture RED, regenerate ALL goldens (`policy_loss`, `value_loss`, `entropy`, `approx_kl`, `ratio_*`, `clip_fraction`) for both K, add the documenting comment: *"epoch-0 ratio==1.0 by construction (anchored reference pass); K=4 goldens capture intended multi-epoch drift; baseline change vs pre-anchor is deliberate (see 2026-06-17 design)."*
- **Command:** `uv run pytest tests/simic/test_ppo_update_golden.py -q`.

### Step 2.7 — Determinism/replay at K=4 (Acceptance #3)
- **New test** `test_anchor_reference_pass.py::test_two_identical_seed_updates_match_k4`: as Step 1.7 but `recurrent_n_epochs=4`; assert two identical-seed `update()` runs match `<1e-6`. CPU/FP32-pinned.
- **Command:** `uv run pytest tests/simic/test_anchor_reference_pass.py::test_two_identical_seed_updates_match_k4 -q`.

### Step 2.8 — KL early-stop + finiteness interleaving at K=4 (Acceptance #6)
- **New test** `test_anchor_reference_pass.py::test_kl_early_stop_fires_sanely_k4`: craft a buffer that produces deterministic policy drift across epochs (Quality nice-to-have: inject a high-variance advantage/reward sequence so drift is real, not trivially zero). Run K=4 with finite `target_kl`; assert `approx_kl` stays under the `1.5·target_kl` trigger when within budget; construct one case that trips early-stop mid-loop and one that does not; assert `early_stop_epoch` is neither always 0 nor never.
- **Add the interleaving case** (Systems major): corrupt epoch-1 numerics to trigger the finiteness `continue` (`:995`), then assert epoch-2's `early_stop_epoch` telemetry is attributed to epochs that RAN (documented in the Step 2.4 C4 comment).
- **Command:** `uv run pytest tests/simic/test_anchor_reference_pass.py::test_kl_early_stop_fires_sanely_k4 -q`.

### Step 2.9 — Value-loss non-regression at K=4 (Acceptance #9)
- **New test** `test_anchor_reference_pass.py::test_value_loss_same_order_k1_vs_k4`: same fixed-seed buffer under K=1 and K=4; assert `value_loss`, `return_mean`, `return_std` finite and same order of magnitude (value path untouched; normalizer updated once per update() in both).
- **Command:** `uv run pytest tests/simic/test_anchor_reference_pass.py::test_value_loss_same_order_k1_vs_k4 -q`.

### Step 2.10 — EV-lifts-off-0 validation (Acceptance #5, HEADLINE EVIDENCE) — CI-GATED
> **Quality major:** the headline criterion must be a reproducible, gated test, not a manually-observed trace. `pytest.ini` `addopts` excludes `slow` from default CI.
- **Committed gate:** place a short deterministic K=4-vs-K=1 paired run (same seed/data, identical `gain=0.01` value-head init in BOTH legs to isolate K) in `tests/integration/` marked `@pytest.mark.integration`, asserting pre-update `explained_variance` (`ppo_agent.py:607-615`) at K=4 strictly increases past a low threshold relative to K=1 over N updates and does not regress. **Capture a concrete numeric EV bound** by pre-running the fixed-seed CPU K=4 patch before merge (Architecture nice-to-have) — do NOT ship an arbitrary `>0.3` placeholder.
- **Named PASS command (the merge gate for Acceptance #5):** `uv run pytest -m integration tests/integration/test_ev_liftoff_k4.py -q`.
- **Full validation run (PR evidence):** launch a real K=4 run via `scripts/train.py`/`train_ppo_vectorized` with `recurrent_n_epochs=4`, `clip_value=False`, reward fix `575482d7`, `phase_profiler=True` (default `:706` — free K=4/K=8 characterization data lands in telemetry), collect the EV trace, attach it to the PR. Acceptance: EV trends from ~0 toward >0.3–0.8, no regression vs paired K=1.
- **Secondary lever (note only, do NOT apply preemptively):** if EV still lags at K=4, the `gain=0.01` value-head init (`factored_lstm.py:504-511`) is a follow-up lever — prefer `0.1` (the contribution-predictor regression gain), NOT `1.0` (contested/reverted), only after K>1 lands.

### Step 2.11 — Full PR2 verification gate
- `uv run pytest tests/simic/test_anchor_reference_pass.py tests/simic/test_ppo_update_golden.py tests/simic/test_vectorized.py tests/simic/test_config.py tests/simic/test_ppo_checkpoint.py tests/simic/test_ppo.py tests/simic/test_policy_amp_context.py -q`
- `uv run pytest -m integration tests/integration/test_ev_liftoff_k4.py -q`
- Confirm acceptance criteria 1–9 each map to a GREEN test (table below); attach EV trace as headline evidence.

**PR2 rollback note:** PR2 touches the config/entrypoint/ctor plumbing surface, the guard message, C4/H5 comments, and goldens. Revert = revert the PR commit, restoring K-pinned-to-1 behavior; because old checkpoints load as K=1 (Step 2.5) and `CHECKPOINT_VERSION` is unchanged, a revert strands no checkpoints and no in-flight run (a K=4 checkpoint reverting to K=1-only code resumes at K=1 with no crash, since `recurrent_n_epochs` is still a ctor param). The single risk surface is a half-landed Step 2.1/2.2/2.2b split — mitigated by the mandated single atomic commit and the `inspect.signature` drift test acting as the atomicity tripwire.

---

## Acceptance-criterion → test traceability

| # | Criterion | Test | PR |
|---|-----------|------|----|
| 1 | Epoch-0 soundness (ratio==1.0, kl==0, `\|ratio−1\|<1e-5`, `no_grad`) | `test_epoch0_ratio_is_one_at_theta0` (1.1) | PR1 |
| 2 | AMP nonzero-per-head-grad gate (load-bearing; `head_grad_norms` nested-dict key; `torch.bfloat16` dtype) | `test_epoch0_per_head_grad_norms_nonzero_under_amp` (1.3) | PR1 |
| 3 | Determinism/replay `<1e-6` (CPU-pinned) | `test_two_identical_seed_updates_match_k4` (2.7); precursor `_k1` (1.7) | PR2 (PR1) |
| 4 | No stale-baseline consumption; all 4 read sites swapped; `hidden_h[:,1:]` never in loss path | `test_loss_path_reads_only_ref_log_probs` (1.4) | PR1 |
| 5 | EV lifts off 0 (paired K=4 vs K=1, CI-gated `-m integration`) | `test_ev_liftoff_k4` + validation run (2.10) | PR2 |
| 6 | KL boundedness; `early_stop_epoch` sane incl. finiteness interleave | `test_kl_early_stop_fires_sanely_k4` (2.8) | PR2 |
| 7 | Single-path enforcement assertion | `test_lstm_requires_ppo_updates_per_batch_one` (2.3) | PR2 |
| 8 | Golden regeneration + documenting comment; offset removed | `test_ppo_update_golden` K=1 (1.6) + parametrized K=4 (2.6) | PR1/PR2 |
| 9 | Value-loss non-regression | `test_k1_finiteness_..._unchanged` (1.5) + `test_value_loss_same_order_k1_vs_k4` (2.9) | PR1/PR2 |

## Sequencing & dependencies
- PR1 merges before PR2: the anchor is the safety precondition for K>1. The AMP grad-gate (1.3, DISCOVERY) and epoch-0 soundness (1.1) prove the anchor correct in isolation at K=1.
- Within PR2: Steps **2.1 + 2.2 + 2.2b(a)** are ONE atomic commit (config→run()→ctor) with the guard rewrite (2.3) in the same PR. The `inspect.signature` drift test is the atomicity tripwire.
- Step 2.10's CI-gated EV test + validation run is the gating headline evidence; do not declare done until observed.

## Residual risks (carried into the plan)
1. **AMP cast-cache poisoning (highest, silent):** mitigated by Step 1.2 (anchor under BF16 `policy_amp_context`, FP32 seam) and gated by Step 1.3 treated as a DISCOVERY test on CUDA. If 1.3 fails RED, the anchor approach is unsound under BF16 — STOP and revise before merge. **CUDA host required to fully discharge this risk; CPU CI cannot.**
2. **Liveness blind spot at epoch 0:** ratio==1.0 is correct-by-construction, so K=1 goldens alone cannot prove the ratio path is wired — proven instead by K=4 epoch>0 goldens (2.6) and the poison-control test (1.4).
3. **`total_train_steps` dead default (1M):** surfaced by K=4 telemetry; resolved by Step 2.2b(a) plumbing or explicitly deferred with a TODO (2.2b(b)).
4. **`early_stop_epoch` attribution under finiteness `continue`:** documented in the C4 comment (2.4) and tested by the interleave case (2.8).
5. **Guard scope:** the single-path assertion guards `_run_ppo_updates`, not a direct `PPOAgent.update()` call (documented limitation, 2.3).
6. **Perf headroom unproven:** K=4/K=8 characterization deferred; `phase_profiler=True` collects the data for free during Step 2.10.

## Critical files
- `/home/john/esper-lite/src/esper/simic/agent/ppo_agent.py`
- `/home/john/esper-lite/src/esper/simic/agent/ppo_metrics.py`
- `/home/john/esper-lite/src/esper/simic/training/vectorized.py`
- `/home/john/esper-lite/src/esper/simic/training/config.py`
- `/home/john/esper-lite/tests/simic/test_anchor_reference_pass.py` (new)
- `/home/john/esper-lite/tests/simic/test_ppo_update_golden.py`
- `/home/john/esper-lite/tests/simic/test_vectorized.py`
- `/home/john/esper-lite/tests/integration/test_ev_liftoff_k4.py` (new)

---

## Confidence Assessment

**Overall Confidence: High** (aggregated from 4× `approve-with-changes`; every `must_fix` independently re-verified against source on 2026-06-17).

| Finding | Confidence | Basis |
|---------|------------|-------|
| Anchor missing; 4 read sites need swap | High | Reality+Quality+Systems converged; I verified `:946-949`, `:969-976`, `:1058-1067`, `:1371` read rollout `data` |
| `head_grad_norms` is a nested dict (not `head_{name}_grad_norm`) | High | Reality+Quality; verified `ppo_metrics.py:131-132` and `test_ppo.py` pattern |
| `policy_amp_context(amp_enabled, resolved_amp_dtype: torch.dtype)` — string degrades silently | High | Reality verified `helpers.py:57` signature |
| Anchor is first BF16 cache touch (hazard, not mitigation) | High | Architecture+Quality; verified telemetry `autocast(enabled=False)` at `:778`, anchor placement after it |
| 2.1/2.2 must be atomic (`to_train_kwargs` is production path) | High | Architecture+Systems; verified `to_train_kwargs:325-381`, ctor `:1090` explicit args |
| `total_train_steps` never plumbed (1M dead default) | High | Systems; verified zero hits in config.py/vectorized.py, default at `:228` |
| EV-liftoff acceptance is the headline; needs CI lane | Moderate | Quality reasoning sound; threshold value unverified until pre-run captured |

## Risk Assessment

**Implementation Risk: Medium** (silent AMP failure mode is High-severity but Certain-to-catch via Step 1.3; everything else is mechanical with explicit tests).
**Reversibility: Easy (PR1) / Moderate (PR2)** — PR1 touches no schema; PR2 touches config/checkpoint surface but old checkpoints load as K=1 with no version bump.

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Anchor poisons BF16 cast cache → zeroed head grads, silent no-learning | High | Possible (Certain if `autocast(enabled=False)` misapplied) | Step 1.2 in-context anchor; Step 1.3 DISCOVERY gate on CUDA |
| Half-landed config plumbing crashes `TrainingConfig.run()` | High | Likely if split | Mandated single atomic commit (2.1/2.2/2.2b); signature drift test tripwire |
| Stale rollout read at `:969-976`/`:1371` misattributes NaN source | Medium | Possible | Swap all 4 sites (1.2); poison-control test (1.4) |
| EV criterion only manually observed, never gated | Medium | Certain if left as draft | CI `-m integration` test with concrete bound (2.10) |
| `early_stop_epoch` misattribution under finiteness skip | Low | Possible | C4 doc + interleave test (2.8) |

## Information Gaps
1. [ ] **EV-liftoff numeric threshold** (Quality): the concrete K=4 EV bound for the gated test must be captured from a pre-merge fixed-seed CPU run; not yet known.
2. [ ] **CUDA availability for Step 1.3** (Architecture/Reality): the load-bearing AMP gate is a no-op on CPU-only CI; a CUDA host is required to truly discharge Residual Risk #1.
3. [ ] **`total_train_steps` policy decision** (Systems): plumb (2.2b-a) vs defer (2.2b-b) is a scope call for the implementer/owner.
4. [ ] **Step 1.3 discovery outcome** (Architecture): whether PyTorch's autocast cache semantics actually materialize the hazard is empirically unconfirmed until 1.3 runs on CUDA — if it fails RED, the anchor architecture itself needs revision.
5. [ ] **Synthesis-specific:** no reviewer load-tested K=4 perf headroom; deferred but flagged.

## Caveats & Required Follow-ups
**Before relying on this plan:**
- [ ] Confirm each blocker's resolution closes its originating finding (read-site swap covers all 4; grad-norm key is nested; AMP arg is `torch.bfloat16`; 2.1/2.2 atomic; slow-path+diagnostic swapped).
- [ ] Run Step 1.3 on a CUDA host BEFORE treating the anchor as proven (it is a discovery gate, not a regression lock).
- [ ] Decide Step 2.2b (plumb vs defer `total_train_steps`) with the owner.

**Assumptions:** Priority weighting reflects this project's risk appetite (silent training failures rank highest); reviewer scope boundaries were respected; the spec's locked decisions (two PRs, `clip_value` stays False, one schedule tick per rollout) hold.

**Limitations:** This synthesis does NOT re-run the full test suite or execute the anchor; it verifies line/key/signature facts the blockers depend on. It does not cover concerns outside the four lenses (no legal/compliance/accessibility surface here). If a reviewer's reasoning about PyTorch autocast cache semantics is wrong, Step 1.3 is the empirical backstop.