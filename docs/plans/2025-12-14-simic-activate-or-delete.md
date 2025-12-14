# Simic: “Activate or Delete” Dead Code Plan (2025-12-14)

## Context

We audited `src/esper/simic/` and found two classes of problems:

1. **Unwired “real” features**: parameters, modules, and telemetry hooks that clearly match design intent but are not exercised by the runtime training path.
2. **True dead code**: code not used by runtime or tests (and not meaningfully “ready to wire”), which violates the repo’s “no legacy code” policy by accumulating inert surface area.

This plan turns the audit into a sequence of small, reviewable PRs that:

- **Activate** code that adds value to PPO training/diagnostics, or
- **Delete** code/flags/params that are redundant or misleading.

Esper/esper-lite is **pre-release**. We prefer fast convergence to the intended architecture over API stability, so this plan assumes we will make **breaking changes** (CLI flags, function signatures, internal APIs) and will not keep compatibility paths.

## Ground Rules / Contracts

## Operator Feedback (incorporate into execution)

This plan is intentionally ruthless: it prioritizes **architectural honesty** over backward compatibility, and it deletes “still functional but worse” paths once the better system exists.

Operator assessment summary:

- This is a **very strong plan**: ruthless in the right way, favoring design intent over compatibility.
- The **Ground Rules** are the anchor; specifically, “seeds and slots are effectively interchangeable” justifies deleting complexity like `max_seeds_per_slot`.
- Strongest elements called out by the operator:
  - **PR0**: delete `max_seeds_per_slot` (correct first move; simplifies the mental model immediately).
  - **PR1**: multiple PPO updates per batch (big sample-efficiency win; keep entropy anneal semantics correct).
  - **PR7**: make `TrainingConfig` real (prevents long-term CLI drift).
- Implementation nuance watch-outs (explicitly required):
  - **PR1 (PPO updates + normalization):** the observation normalizer must update **once per new batch of collected data** (rollout), not per optimization step. The rollout buffer’s observations are static; policy updates are not.
  - **PR4 (MaskedCategorical):** ensure the chosen finite mask value behaves like “-inf after softmax” across dtypes. For FP16, the representable range is smaller; `-1e4` is typically sufficient (since `exp(-1e4)` underflows to 0), but validate with unit tests and avoid `-inf`/`dtype.min` pitfalls that can trigger NaNs in some softmax implementations.

### Seed/slot semantics (operator confirmed)

- **Seeds and slots are effectively interchangeable** in this codebase.
- The codebase enforces **one seed per slot** (a slot either has a seed or it doesn’t; fossilized seeds are permanent).
- Therefore, **`max_seeds_per_slot` is nonsensical** and must be removed everywhere (API + CLI + docs + constants).
- We explicitly do **not** add “max germination attempts”; Tamiyo should learn cull/grow/fossilize timing and policies.

### No legacy code policy

- When we remove or change an API/flag, we **delete the old code completely** and update all call sites and tests in the same change.
- No compatibility shims, no deprecated aliases, no “support both old and new”.
- If two code paths exist and one is the “new / better” system, we **delete the older / worse** path even if it still works. No “fallback” implementations kept around “just in case”.

### Scope

In scope:
- `src/esper/simic/` and its runtime wiring via `src/esper/scripts/train.py`.
- Reward selection wiring via `src/esper/runtime/tasks.py` and reward configs in `simic/rewards.py`.
- Tests under `tests/` that cover Simic behavior and public CLI wiring.

Not in scope:
- Algorithmic redesign of PPO beyond making existing knobs real and diagnostics actionable.
- Multi-node / DDP changes (explicitly deferred where already documented, e.g. `SeedSlot.force_alpha()`).

---

# Deliverable

After completing this plan:

1. **No no-op knobs**: every CLI/config knob either affects runtime behavior or does not exist.
2. **No dead Simic code**: every Simic module/class/function is either:
   - used by runtime training, or
   - used by tests as an intentional contract/invariant, or
   - deleted.
3. **Telemetry escalation works**: anomalies trigger temporary DEBUG-mode telemetry automatically.
4. **Masking is correct and self-validating**: invalid-action states raise a clear state-machine error early and entropy is computed over valid actions.
5. **Reward families are explicit**: contribution-based and loss-based rewards are selectable and both are truly wired.
6. **Configuration is real**: `TrainingConfig` is loadable from CLI and maps cleanly onto runtime behavior (no “paper fields”).

---

# Risk & Complexity Assessment (per PR)

Scale:
- **Complexity:** S / M / L / XL (engineering effort + surface area touched)
- **Risk:** Low / Med / High (combined runtime-correctness + training-dynamics + integration risk)

| PR | Scope | Complexity | Risk | Primary risk drivers | Notes / mitigations |
|---|---|---:|---:|---|---|
| PR0 | Delete `max_seeds_per_slot` + related plumbing | S | Low | Pure deletion across CLI/vectorized/leyline | Do first; `rg`-driven delete; no compatibility path. |
| PR1 | Implement `ppo_updates_per_batch` + fix annealing | M–L | Med | Subtle interactions: buffer clearing, KL early-stop, `train_steps`, normalizer update timing | Keep default 1; add unit tests; normalizer update once per rollout batch only. |
| PR2 | Telemetry auto-escalation + cheap NaN/Inf anomaly signal | S–M | Low–Med | Risk of “too much debug telemetry” during unstable runs | Gate behind `TelemetryConfig`; tick once per batch; tests for escalation lifetime. |
| PR3 | Delete unused dual-gradient collector API | S | Low | Pure deletion; minimal behavioral change | Confirm zero call sites; run telemetry pipeline tests. |
| PR4 | Wire `MaskedCategorical` into policy network | L | High | Core action sampling/log_prob/entropy changes; compile sensitivity; hard failure if any mask is all-false | Dedicated PR; add tests for entropy and all-false masks; choose FP16-safe mask value. |
| PR5 | Activate `RatioExplosionDiagnostic` in PPO metrics + telemetry | M | Med | Metric aggregation/serialization: mixing dict + numeric metrics can break averaging logic | Keep diagnostic out of numeric averaging; compute only on anomaly. |
| PR6 | Add `--reward-family` and wire loss rewards | M–L | Med | Reward scaling/sign semantics; telemetry schema complexity; user-facing CLI rules | Fail fast on invalid flag combos; keep defaults conservative; add branch-coverage test. |
| PR7a | Prune `TrainingConfig` + add strict JSON loading + contract tests | S–M | Low–Med | Risk of deleting fields still assumed by tests/docs | Keep config minimal and honest: only fields wired to runtime; delete YAML/imagenet paper; add signature-level tests to prevent drift. |
| PR7b | Cut over PPO CLI to `TrainingConfig` (delete redundant flags) | M–L | Med | User-facing breakage; merge conflicts in `train.py`; easy to keep “two config surfaces” | Minimize diff to telemetry/TUI wiring; delete no-op flags like `--update-every`; update README + scripts in same PR; add parser tests. |
| PR8 | Cleanup sweep (remove dead imports/APIs) | S–M | Low–Med | Mostly mechanical; test updates needed if deleting `normalize_observation` | Defer to end to reduce rebase churn; let `ruff` guide. |

## Suggested execution order (to reduce risk)

Even though PRs are numbered, the lowest-risk path is:

1. **PR0 → PR3**: delete nonsensical knobs and dead APIs first.
2. **PR1 → PR2 → PR5**: make documented training/telemetry behaviors real; improve diagnostics.
3. **PR4 → PR6**: change core sampling/masking, then expand reward families.
4. **PR7a → PR7b → PR8**: consolidate config surface last, then do final cleanup.

# Work Plan (small PRs)

## PR0 — Remove `max_seeds_per_slot` everywhere (delete, don’t repurpose)

### Why
`max_seeds_per_slot` is redundant given the one-seed-per-slot model. Keeping it is misleading and violates “no dead knobs”.

### Changes
- Delete the CLI flag and plumbing:
  - `src/esper/scripts/train.py`: remove `--max-seeds-per-slot` and the call-site arg `max_seeds_per_slot=...`.
- Delete the vectorized API parameter:
  - `src/esper/simic/vectorized.py:200`: remove `max_seeds_per_slot` from signature and docstring.
  - Remove any mention of per-slot seed limits in comments.
- Delete the Leyline constant:
  - `src/esper/leyline/__init__.py`: remove `DEFAULT_MAX_SEEDS_PER_SLOT` export if it has no remaining call sites.
- Update docs referencing it (including archived plans if they’re being used as live references elsewhere; otherwise leave archives alone).

### Acceptance criteria
- `rg -n "max_seeds_per_slot|DEFAULT_MAX_SEEDS_PER_SLOT" src/ tests/` returns nothing.
- CLI help no longer advertises the removed flag.

### Tests
- `uv run pytest -m "not slow"` (or at minimum: simic + CLI-related tests).

---

## PR1 — Make `ppo_updates_per_batch` real in vectorized PPO

### Why
The parameter exists and is documented as a feature, but it is currently a no-op.

### Implementation details
- In `src/esper/simic/vectorized.py`:
  - After episode collection, run **N PPO updates on the same rollout**:
    - For updates 0..N-2: call `agent.update(clear_buffer=False)`
    - Final update: call `agent.update(clear_buffer=True)`
  - Aggregate metrics across update calls into a single “batch metrics” dict:
    - Use mean for scalars like `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `ratio_max/min`, `explained_variance`.
    - Preserve “worst case” fields where helpful (e.g. max of `ratio_max`, min of `ratio_min`).
  - Add an **optional “stop early across updates”** rule:
    - If `target_kl` is enabled and an update returns `approx_kl` above the internal early-stop threshold, stop running additional updates for this batch to avoid overshooting.
  - Ensure observation normalizer update stays **once per batch** (after PPO updates succeed) and is still skipped on Governor rollback.

### Entropy annealing semantics
`PPOAgent.train_steps` increments once per `update()` call. With multiple updates per batch, naive annealing completes too quickly.

Fix: keep the CLI semantics (“anneal over X episodes”) by converting episodes→train steps using:

- `batches_for_anneal = ceil(entropy_anneal_episodes / n_envs)`
- `entropy_anneal_steps = batches_for_anneal * ppo_updates_per_batch`

This preserves “episode-based” intent while respecting extra updates per batch.

### Acceptance criteria
- Setting `ppo_updates_per_batch=2` visibly results in two PPO updates per batch (telemetry + logs + metrics change).
- Entropy annealing completes at roughly the configured episode count (not half/quarter due to multiple updates).

### Tests
- Add a focused unit test in `tests/simic/` for “update called N times” by mocking/stubbing `PPOAgent.update` or instrumenting `PPOAgent.train_steps`.
- Run:
  - `uv run pytest tests/simic -m "not slow"`
  - `uv run pytest tests/integration/test_vectorized_factored.py -m "not slow"`

---

## PR2 — Telemetry auto-escalation on anomalies (make it real)

### Why
`TelemetryConfig` is designed for temporary escalation to DEBUG mode, but vectorized PPO currently never calls `escalate_temporarily()` or `tick_escalation()`.

### Changes
- In `src/esper/simic/vectorized.py`:
  - On anomaly detection (post-PPO update), if `telemetry_config.auto_escalate_on_anomaly`:
    - call `telemetry_config.escalate_temporarily()`
  - At each batch boundary (once per PPO-update batch), call `telemetry_config.tick_escalation()` to count down.
  - Use `telemetry_config.should_collect("debug")` consistently to gate debug-only collectors:
    - reward component telemetry (already gated)
    - any other debug-only emissions we add in PR4/PR5
- Feed numerical instability signals into anomaly detection cheaply:
  - Treat any non-finite PPO metric (`not math.isfinite(x)`) as `(has_nan/has_inf)` for `AnomalyDetector.check_all`.
  - This enables “numerical_instability” anomalies without expensive per-parameter scans.

### Acceptance criteria
- After a detected anomaly, `telemetry_config.effective_level == DEBUG` for the next N batches and then returns to configured level.
- Reward component telemetry begins emitting automatically during the escalation window.

### Tests
- Extend/add tests in `tests/simic/`:
  - A unit test that simulates an anomaly and asserts escalation counter behavior.
  - A unit test that non-finite metrics trigger numerical anomaly in `AnomalyDetector.check_all`.

---

## PR3 — Remove unused dual-gradient collector API (delete dead code)

### Why
`collect_dual_gradients_async()` is implemented but has **zero call sites**; vectorized training re-implements a slightly different collection strategy inline. Keeping unused APIs invites drift.

### Changes
- Delete from `src/esper/simic/gradient_collector.py`:
  - `collect_dual_gradients_async()`
  - Any now-unused supporting code/imports.
- Remove unused imports in `src/esper/simic/vectorized.py` (`collect_dual_gradients_async` is currently imported but unused).

### Acceptance criteria
- `rg -n "collect_dual_gradients_async" src/ tests/` returns nothing.
- Existing gradient ratio telemetry still works (it uses the “squared sum” dict + `materialize_dual_grad_stats`).

### Tests
- `uv run pytest tests/simic/test_gradient_collector_enhanced.py -m "not slow"`
- `uv run pytest tests/integration/test_telemetry_pipeline.py -m "not slow"`

---

## PR4 — Wire `MaskedCategorical` into `FactoredRecurrentActorCritic` (activate + validate)

### Why
`MaskedCategorical` and `InvalidStateMachineError` are currently dead, but they solve two real problems:
1. **State machine safety**: fail fast when a mask has no valid actions.
2. **Correct entropy under masking**: compute entropy over valid actions only (and normalize by valid-action max entropy).

### Implementation approach (high-level)
- In `src/esper/simic/tamiyo_network.py`:
  - Use unmasked logits from heads + per-head masks to construct `MaskedCategorical` distributions.
  - Compute:
    - `sample()` (or argmax when deterministic)
    - `log_prob(actions)`
    - `entropy()` (already normalized to [0, 1] relative to valid count)
- Resolve the “mask value” mismatch:
  - Today, network masking uses `_MASK_VALUE = -1e4` (chosen for FP16 stability).
  - Today, `MaskedCategorical` uses `torch.finfo(dtype).min` (which is risky for FP16).
  - Update `MaskedCategorical` to use a stable finite mask value (match `_MASK_VALUE` semantics) so FP16/bfloat16 remain safe.

### Acceptance criteria
- If a mask is all-false for any batch element, action sampling raises `InvalidStateMachineError` with actionable context.
- PPO entropy uses mask-aware normalized entropy (not penalized by invalid actions).
- Existing tests still pass; add a test for masked entropy normalization behavior.

### Tests
- Existing:
  - `uv run pytest tests/simic/test_action_masks.py -m "not slow"`
  - `uv run pytest tests/integration/test_vectorized_factored.py -m "not slow"`
- Add:
  - A unit test ensuring entropy is 1.0 for uniform over valid actions (and unaffected by masked-out logits).

---

## PR5 — Activate `RatioExplosionDiagnostic` (make ratio anomalies actionable)

### Why
`RatioExplosionDiagnostic` is implemented and tested but unused; it should feed anomaly telemetry with concrete “what went wrong” data.

### Changes
- In `src/esper/simic/ppo.py` (`PPOAgent.update()`):
  - When ratio anomalies are detected (max too high / min too low), construct a `RatioExplosionDiagnostic` for at least the `op` head:
    - `ratio = exp(new_log_prob - old_log_prob)`
    - `actions = op_actions`
  - Return it in the metrics dict as `ratio_diagnostic` (already a dict via `.to_dict()`).
- In `src/esper/simic/vectorized.py`:
  - When emitting anomaly telemetry events, include `ratio_diagnostic` if present.

### Acceptance criteria
- Ratio anomaly events include diagnostic indices/values/actions in telemetry payload.
- No measurable overhead in the non-anomalous case (diagnostic only computed on anomaly).

### Tests
- Extend `tests/simic/test_ratio_explosion.py` as needed for the new integration point.
- Add an assertion in a PPO unit test that a forced ratio explosion produces `ratio_diagnostic` in returned metrics.

---

## PR6 — Wire loss-primary reward path as a first-class reward family

### Why
`LossRewardConfig` and `compute_loss_reward()` are implemented + tested but unused by runtime training. This is classic “upgrade scaffolding”; it should become real or be deleted. We are choosing **real**.

### Proposed UX
- Add a new CLI selector:
  - `--reward-family {contribution,loss}` (default: `contribution`)
- Keep `--reward-mode {shaped,sparse,minimal}` as **contribution-family only**.
  - If `reward-family=loss`, reject contribution-specific flags that don’t apply (fail fast).

### Runtime wiring
- In `src/esper/simic/vectorized.py`:
  - If `reward-family=contribution`: current behavior (`compute_reward` dispatcher).
  - If `reward-family=loss`: compute reward via `compute_loss_reward()`:
    - Inputs already available each epoch: `signals.metrics.loss_delta`, `env_state.val_loss`, `seed_info`, `total_params`, `host_params`, terminal epoch.
    - Use task defaults: `task_spec.loss_reward_config` unless overridden.
- Emit reward telemetry in DEBUG for both families (update `RewardComponentsTelemetry` if needed, or add a parallel telemetry payload for loss rewards).

### Acceptance criteria
- Running PPO with `--reward-family loss` produces learning runs (at minimum: code executes end-to-end; reward values are finite; no crashes).
- Tests for loss reward remain meaningful and now cover the runtime integration layer.

### Tests
- Add a small integration test that instantiates vectorized training in “dry”/mock mode (if available) or at least verifies the reward selection branch is exercised.
- Run:
  - `uv run pytest tests/test_loss_primary_rewards.py -m "not slow"`
  - `uv run pytest tests/simic/test_reward_modes.py -m "not slow"`

---

## PR7 — Make `TrainingConfig` the real, loadable config surface (split to reduce risk)

### Why
`TrainingConfig` exists but is unused. Wiring it without pruning dead fields would reintroduce the “no-op knob” problem at a higher level.

### Research findings (risk drivers we can eliminate)
- `src/esper/simic/config.py` currently documents “YAML serialization” and includes an `ImageNet` preset, but there is **no ImageNet task** in `src/esper/runtime/tasks.py`. Both are **paper surface** and should be deleted.
- `TrainingConfig` currently includes many fields not accepted by `train_ppo_vectorized()` (e.g. governor/stabilization knobs, `gae_lambda`, `batch_size`, `n_epochs`, etc). If we “wire config” without changing runtime, these become **no-op config knobs**.
- `src/esper/scripts/train.py` includes `--update-every` for PPO but does not pass it to `train_ppo_vectorized()` → **dead CLI knob**.

### PR7a — Make `TrainingConfig` honest + add guardrails (low risk)
Goal: make `TrainingConfig` a *strict schema* for exactly what we can/will wire, and add tests that prevent future drift.

Changes:
- In `src/esper/simic/config.py`:
  - Delete paper docs (“YAML serialization”) and dead presets (`for_imagenet()`).
  - Ruthlessly prune config fields to only those that are (or will be, by PR0/PR1/PR6) **accepted by `train_ppo_vectorized()`**.
    - Keep: `n_episodes`, `n_envs`, `max_epochs`, `lr`, `clip_ratio`, `gamma`, entropy settings, `ppo_updates_per_batch`, LSTM sizing (`lstm_hidden_dim`, `chunk_length`), `use_telemetry`, slot/budget knobs (`slots`, `max_seeds`), and reward selection (`reward_family`, `reward_mode` if applicable).
    - Delete (unless separately wired later): governor/stabilization knobs, `gae_lambda`, `batch_size`, `n_epochs`, `value_coef`, `max_grad_norm`, etc.
  - Add strict JSON loading helpers:
    - `TrainingConfig.from_json_path(path)` using stdlib `json`.
    - Reject unknown keys (fail fast; no aliasing/compat keys).
  - Add `TrainingConfig.validate()` enforcing invariants:
    - `ppo_updates_per_batch >= 1`, `n_envs >= 1`, `max_epochs >= 1`
    - `chunk_length` matches `max_epochs` (unless/when we explicitly support mismatch)
    - reward flag compatibility (`reward_family=loss` rejects contribution-only fields)
- In tests (`tests/test_simic_config.py`, `tests/simic/test_config.py`):
  - Update expectations to match the pruned schema.
  - Add a **contract test** that prevents CLI drift at the config boundary:
    - `TrainingConfig().to_train_kwargs().keys()` must be a subset of `inspect.signature(train_ppo_vectorized).parameters` (and ideally cover all required config fields).
  - Add tests for strict JSON loading:
    - Unknown key → raises.
    - Roundtrip `to_dict()` → `from_dict()` preserves values.

Acceptance criteria:
- No `TrainingConfig` field exists that cannot affect runtime (by virtue of being passed to training).
- `for_imagenet` and YAML mentions are gone.
- Contract tests fail loudly if `train_ppo_vectorized()` signature changes without updating `TrainingConfig`.

### PR7b — Cut over the PPO CLI to `TrainingConfig` as single source (medium risk)
Goal: eliminate parallel CLI hyperparam surfaces and delete no-op flags in one clean cut.

Changes:
- In `src/esper/scripts/train.py` (PPO subcommand only):
  - Add `--config-json PATH` (required unless a preset is provided).
  - Add `--preset {cifar10,cifar10_deep,tinystories}` (align with `src/esper/runtime/tasks.py`; **no ImageNet**).
  - Delete redundant PPO hyperparameter flags that are now represented in `TrainingConfig`:
    - `--lr`, `--clip-ratio`, all `--entropy-*`, `--gamma`, `--no-telemetry`, etc.
    - Delete `--update-every` entirely (currently dead).
  - Build `TrainingConfig` from preset/JSON and call:
    - `train_ppo_vectorized(**config.to_train_kwargs(), device=args.device, devices=args.devices, task=args.task, …)`
    - Keep non-hyperparam runtime knobs in the CLI: `--task`, `--device`, `--devices`, dataloader perf knobs, telemetry outputs/TUI, etc.
  - Print/log the effective config (e.g., `config.summary()` or JSON dump) at start for reproducibility.
- Update docs/scripts in the same PR:
  - `README.md` CLI reference: remove deleted flags; show config-based usage.
  - `scripts/train_cifar.sh`, `scripts/train_tinystories.sh`: pass `--preset` (or a config path) instead of old flags.

Risk reduction tactics:
- Keep telemetry/TUI wiring unchanged; only touch the PPO argument surface and the final callsite.
- Add a parser test by factoring `train.py` to expose a `build_parser()` helper (so tests can validate `--config-json`/`--preset` behavior without running training).

Acceptance criteria:
- `esper.scripts.train ppo --help` shows no removed/no-op flags.
- PPO runs can be launched via `--preset` alone (plus standard task/device flags).
- A config JSON fully specifies all PPO hyperparameters; unknown keys fail fast.

### Acceptance criteria
- `TrainingConfig` is used by the CLI path (not just tests).
- No config fields exist that do nothing.
- A saved config JSON can reproduce a run’s key hyperparameters.

### Tests
- Update `tests/test_simic_config.py` to reflect any field changes.
- Add a CLI config parsing test (argparse) if test harness supports it.

---

## PR8 — Cleanup sweep: remove remaining dead code and misleading params

### Targets (non-exhaustive; confirm with `ruff` + `rg`)
- Remove unused imports and unused locals:
  - `src/esper/simic/vectorized.py`: remove unused `num_germinate_actions` local.
  - `src/esper/simic/ppo.py`: drop unused imports/objects (`safe`, telemetry imports, `get_hub`, `logger` if truly unused after earlier PRs).
  - `src/esper/simic/features.py`: remove unused `TensorSchema`/`TENSOR_SCHEMA_SIZE` imports if still unused.
- Remove unused APIs or make them used:
  - `src/esper/simic/features.py:normalize_observation`: either wire it into runtime or delete it + its tests.
    - Recommended: **delete** if RunningMeanStd is the canonical normalizer for PPO states.
- Ensure docs/comments match reality (no stale “Phase 2” promises).

### Acceptance criteria
- `uv run ruff check src/ tests/` clean (or at least no new warnings).
- No unused public parameters in Simic APIs.

---

# Validation Checklist (each PR)

Minimum per PR:
- `uv run pytest <relevant test files>`
- `uv run ruff check src/ tests/`

Full-suite (after PR8):
- `uv run pytest`
- `uv run mypy src/`

---

# Risks & Mitigations

- **Masking + entropy changes can affect training dynamics**:
  - Mitigation: keep entropy normalized to [0, 1] and add targeted tests for masked entropy behavior.
- **Multiple PPO updates per batch can cause policy drift**:
  - Mitigation: stop early across update calls when KL exceeds threshold; keep default `ppo_updates_per_batch=1`.
- **Reward family selection can confuse CLI**:
  - Mitigation: fail fast on invalid flag combinations and keep defaults conservative.

---

# Completion / Archiving

When the final PR lands and the repo is clean:
- Move this plan to `docs/plans/archive/` with a short “executed” note and links to the PRs (per `CLAUDE.md` archive policy).
