# 02 — Subsystem Catalog

**Method**: 10 review units (8 domains + Simic split into RL-core/rewards/training + a support cluster), each analyzed by a `codebase-explorer` agent reading source with Loomweave, then independently spot-checked by an `analysis-validator`. Validation: **3 PASS, 7 WARN, 0 BLOCK**. All validator corrections are merged below (✎).

**Authoritative reference correction**: The workflow prompt used a non-canonical "commandment" numbering. The ROADMAP's actual **Nine Commandments** are: (1) Sensors match capabilities, (2) Complexity pays rent, (3) GPU-first iteration, (4) Progressive curriculum, (5) Train Anything protocol, (6) Morphogenetic plane, (7) Governor prevents catastrophe, (8) Hierarchical scaling, (9) Frozen Core economy. The rules labeled "No Legacy", "No Defensive Programming", and "Telemetry as a contract" are **CLAUDE.md project rules / README design principles**, not ROADMAP commandments. Compliance evidence below is sound; only the labels are reconciled.

---

## Dependency layering (measured)

```
leyline (contracts)  ◄── everyone   |  imports nothing at runtime (0 cycles)
   ▲
   ├── kasmina   (host + seed lifecycle)      → leyline only
   ├── tamiyo    (decision policy)            → leyline (+ nissa in tracker.py)
   ├── tolaria   (execution + governor)       → leyline, nissa, kasmina(TYPE_CHECKING)
   ├── nissa     (telemetry hub)              → leyline
   ├── karn      (UI + analytics + store)     → leyline, nissa.output
   └── simic     (RL orchestration apex)      → tamiyo, kasmina, tolaria, nissa, karn, runtime, utils, leyline
```

Simic is the **only** domain that wires all others together (the composition root is `simic/training/env_factory.py`). leyline sits at the bottom with fan-in 197 and zero runtime outbound domain imports — except one structural wrinkle (see leyline §).

---

## Leyline — DNA / Shared Contracts  ·  7,277 LOC  ·  *validation: WARN*

**Responsibility:** Defines all cross-domain enums, protocols, tensor/telemetry schemas, action-space factoring, and default constants that every other domain imports.

**Key components:**
- `telemetry.py` (2,524 — **god-file**): `TelemetryEventType` enum, `TelemetryEvent`, ~24 frozen `slots=True` typed payload dataclasses unioned into a discriminated `TelemetryPayload`. `from_dict()` validators **raise** (e.g. ~378–408) rather than silently defaulting.
- `__init__.py` (1,199): constants/config registry + re-export facade — ~100 semantically-named constants (`LEYLINE_VERSION`, `DEFAULT_GAMMA`, `ENTROPY_FLOOR_PER_HEAD`, `OBS_V3_*`, `HEAD_NAMES`, `MASKED_LOGIT_VALUE`).
- `factored_actions.py` (526): `BlueprintAction`/`GerminationStyle`/`LifecycleOp` IntEnums; `STYLE_TO_KASMINA`; `to_blueprint_id()` raises `KeyError` on mismatch (fail-fast).
- `host_protocol.py` (98): `HostProtocol` (runtime_checkable) — the Train-Anything contract; torch imports are `TYPE_CHECKING`-only.
- `seed_protocols.py` (292), `governor_protocol.py` (83), `policy_protocol.py` (404 — defines **`PolicyBundle`** ✎ + `ActionResult`/`EvalResult`/`ForwardResult`), `output_protocol.py`.
- `stages.py` (96): `SeedStage` IntEnum + `VALID_TRANSITIONS` table — the single lifecycle FSM.
- `reward_config.py` (67): **`LossRewardConfig`** ✎ (rent/penalty knobs); `slot_config.py`/`slot_id.py` (316): the single `rNcN` coordinate system.

**Dependencies:** Inbound — all 7 domains + runtime + scripts. Outbound — **only** `esper.simic.telemetry.observation_stats` and `esper.simic.rewards.reward_telemetry`, via `TYPE_CHECKING` + function-local late imports (telemetry.py:610/1947/1968).

**Patterns:** Protocol-based dependency inversion (torch gated behind `TYPE_CHECKING`); make-illegal-states-unrepresentable (`GerminationStyle` collapses blend×alpha into 4 valid composites); typed discriminated telemetry payloads; fail-fast parsing; constants-as-contract registry.

**Concerns:**
- **[HIGH] Contracts hub imports back into a consumer (simic).** The only outbound deps are two simic telemetry types, pulled via function-local late imports whose docstrings admit "avoid circular dependency at module load time". A fan-in-197 DNA layer should have zero outbound domain deps. **Fix direction:** `RewardComponentsTelemetry`/`ObservationStatsTelemetry` are themselves telemetry contracts and belong *in* leyline.
- **[MED]** `telemetry.py` god-file (2,524 LOC) mixes enum + base event + payloads + ~250 parsing calls.
- **[LOW]** `__init__.py` is a 1,199-LOC constants monolith.

**Commandment / rule compliance:** C5 Morphogenetic plane **upheld** (single `slot_id` space); C2 rent **partial** (config knobs here, computation in simic); No-Legacy **upheld** (✎ note: `slot_id.py:19 _LEGACY_NAMES` is a clean-break *rejection* guard, not a compat alias); No-Defensive **upheld**; Telemetry contract **upheld** (24 typed payloads, raise-on-bad-data) modulo the circular wrinkle.

✎ **Validator corrections merged:** class is `LossRewardConfig` (not `LossPrimaryRewardConfig`); `PolicyBundle` (not `PolicyProtocol`); commandment-7 evidence rephrased around `_LEGACY_NAMES`.

**Confidence:** High.

---

## Kasmina — Stem Cells / Host + Seed Lifecycle  ·  5,821 LOC  ·  *validation: WARN*

**Responsibility:** The morphogenetic host network and per-slot seed lifecycle state machine (DORMANT→…→FOSSILIZED/PRUNED), grafting blueprint-instantiated seeds into a host via gradient-isolated alpha blending.

**Key components:**
- `slot.py` (2,831 — **god-file**): four concerns in one module — `SeedMetrics`, `SeedState` (class L386; `transition()` at L520 enforcing leyline `VALID_TRANSITIONS` ✎), `QualityGates` (G0/G1/G2/G3/G5), and `SeedSlot` (nn.Module engine: `germinate` L1313, `advance_stage`, `prune`, `forward`/blending, `step_epoch` lifecycle clock L2400).
- `host.py` (1,006): `CNNHost` + `TransformerHost` implementing `HostProtocol` via segment routing; `MorphogeneticModel` (holds `host: HostProtocol` + nn.ModuleDict of SeedSlots).
- `isolation.py` (252): `blend_with_isolation`, `ste_forward` straight-through estimator, `GradientHealthMonitor`.
- `blueprints/{registry,cnn,transformer}.py`: topology-scoped plugin registry; CNN/transformer seed factories.
- `alpha_controller.py` (204): alpha amplitude scheduling during BLENDING.

**Dependencies:** Inbound — simic/training (env_factory, vectorized, vectorized_trainer, parallel_env_state, static_final_replay), tolaria/environment, tamiyo/heuristic, runtime/tasks. Outbound — **leyline exclusively**.

**Patterns:** Protocol-based host inversion (seed engine never imports a concrete host); contract-enforced state machine (delegates legality to leyline); topology-scoped plugin registry; STE incubation (TRAINING-stage seeds learn at alpha==0 without altering host activations); structural gradient isolation (detach at boundary — "the structural guarantee is absolute"); torch.compile-friendly hot path (cached alpha tensors); fail-fast deserialization.

**Concerns:**
- **[HIGH]** `slot.py` 2,831-LOC god-file mixing four responsibilities (`SeedSlot` alone ~1,800 lines).
- **[MED] Silent topology default to `'cnn'`** when no TaskConfig provided ("to match slot-only tests") — a transformer host could germinate a CNN blueprint and only fail later at the shape-probe.
- **[LOW]** Intentional `forward_to_segment` duplication between CNNHost/TransformerHost; `host.py` couples to the `ConvBlock` building block.

**Commandment compliance:** C5 Train Anything **upheld** (seed engine imports only generic `torch.nn`; host bound as `HostProtocol`; HostProtocol lives in leyline); C6 Morphogenetic plane **upheld** (single `MorphogeneticModel`, one ModuleDict keyed `r0c0…`); No-Legacy **upheld** (rejects pre-Phase-1 checkpoints rather than shimming); No-Defensive **upheld**; Telemetry contract **upheld** (typed closed union; "dicts are not supported").

✎ **Validator corrections merged:** removed `simic/rewards/types.py` from inbound (it only references kasmina in a docstring, no import); `registry.py` is 140 LOC; `SeedState` class at L386, `transition()` at L520.

**Confidence:** High.

---

## Tamiyo — Brain / Decision Policy  ·  5,146 LOC  ·  *validation: WARN*

**Responsibility:** Converts host-training observations into seed-lifecycle decisions; provides a learned recurrent actor-critic (Policy V2) and a rule-based heuristic, plus feature extraction, action masking, and the `PolicyBundle` plumbing Simic's PPO loop consumes.

**Key components:**
- `networks/factored_lstm.py` (1,394 — **god-file**): `FactoredRecurrentActorCritic` — feature net → `ResidualLSTM` → 8 factored action heads + **op-conditioned Q(s,op) value head** + auxiliary contribution predictor; `BlueprintEmbedding`. `get_action()` (rollout) and `evaluate_actions()` (PPO update) share FP32 masking/floor/log-prob helpers to keep importance ratios unbiased under BF16.
- `policy/features.py` (889): hot-path Obs V3 extraction — `[batch, 116]` (23 base + 31·num_slots) + int64 blueprint indices, CPU-fill-then-single-H2D; `get_feature_size = 23 + 31·num_slots`; symlog compression; `-1.0` "unknown" sentinel.
- `policy/action_masks.py` (735): lifecycle-legality masks from leyline `VALID_TRANSITIONS`; `MaskedCategorical`; FP32 floor/log-prob/entropy helpers shared with the network.
- `heuristic.py` (403): `HeuristicTamiyo.decide()` rule-based controller.
- `policy/{lstm_bundle,factory,registry,heuristic_bundle}.py`: `PolicyBundle` adapter (registered `'lstm'`), registry-based instantiation.
- `tracker.py` (373): `SignalTracker` — emits typed telemetry to Nissa (the only non-leyline outbound).

**Dependencies:** Inbound — simic.training (batch_ops, vectorized_trainer, action_execution), simic PPO agent/policy creation. Outbound — leyline (contracts), nissa (tracker only), generic torch/nn.

**Patterns:** Protocol-boundary decoupling; single-source-of-truth action legality; **shared FP32 "seam"** so PPO importance ratios cannot drift under BF16 autocast; op-conditioned canonicalization of irrelevant heads; fail-fast contract-as-error; hot-path GPU discipline (multinomial over Categorical to avoid CPU-GPU syncs; `@torch.compiler.disable` on sync-forcing validators); typed observation semantics (out-of-range sentinel, not 0.0).

**Concerns:**
- **[MED]** `factored_lstm.py` 1,394-LOC god-file; op-conditional canonicalization **duplicated** between `get_action` and `evaluate_actions` (a divergence risk the code itself warns about).
- **[MED]** Stale dimension docstrings ("shape (30,)" / "30 dims per slot" vs actual 31).
- **[LOW]** `register_policy` validates structure (hasattr) but not signatures at runtime; heuristic HOLDING fossilize/prune blocks indefinitely on missing counterfactual (no age/timeout fallback).

**Commandment compliance:** C3 GPU-first **partial** (strong discipline, but a `threading.Lock` double-checked-locks the one-hot table cache on first-touch per device — warm-up only, not steady state); C5 Train Anything **upheld**; C6 plane **upheld**; C7 Governor **upheld** (policy only *receives* `allow_governor_override`, an unrelated alpha-schedule flag — no API to disable the watchdog); No-Legacy **partial** (`self.lstm_ln = nn.Identity()` retained "for backwards compat" — vestigial); No-Defensive **upheld**; Telemetry **upheld**.

✎ **Validator corrections merged:** dropped the stray "fan-in 76" rebuttal (no such claim existed); the substantive 116/128-dim figures are correct; minor off-by-one citations corrected.

**Confidence:** High.

---

## Simic-RL — PPO Core (agent / attribution / normalization)  ·  ~10K LOC  ·  *validation: WARN*

**Responsibility:** The on-policy PPO learner that trains Tamiyo's factored recurrent policy — GAE, per-head clipped surrogate, value normalization, rollout storage, counterfactual attribution — independent of reward design and the training loop.

**Key components:**
- `agent/ppo_agent.py` (1,696 — **god-file**): `PPOAgent`. `update()` alone is ~900 lines (god-method) mixing GAE, advantage stats, Q-value telemetry, finiteness gating, per-head metrics, aux loss, optimizer step, checkpoint I/O.
- `agent/rollout_buffer.py` (847): `TamiyoRolloutBuffer` — pre-allocated device-resident tensors; per-env GAE with truncation-vs-terminal bootstrap; idempotent advantage normalization.
- `agent/ppo_update.py` (470): pure PPO math — per-head + joint ratio metrics, per-head clipped surrogate, entropy floor, contribution aux loss.
- `agent/advantages.py` (92): applies leyline causal masks so heads with no causal effect get zero gradient.
- `agent/{ppo_metrics,types}.py`: `PPOUpdateMetrics` TypedDict (~120 fields) + builder encoding the finiteness-gate contract.
- `control/normalization.py` (437): `RunningMeanStd`, `RewardNormalizer`, `ValueNormalizer` (PopArt-lite).
- `attribution/counterfactual.py` (548): full-factorial / Shapley (antithetic, seeded RNG) / ablation strategies.

**Dependencies:** Inbound — simic/training (vectorized, ppo_coordinator, action_execution, parallel_env_state, policy_group, dual_ab, env_factory, normalizer_checkpoint), **simic/telemetry/emitters, simic/training/epoch_runner, simic/training/vectorized_trainer** ✎. Outbound — leyline, tamiyo.networks (lazy), simic.telemetry.

**Patterns:** Pure-function math layer (stateless `ppo_update.py`/`advantages.py`); **factored per-head trust region** (each head's ratio clipped independently, surrogates summed); single source of truth for causal masks; device-resident pre-allocated buffer (detach on `add()`); finiteness gate as fail-fast contract; PopArt-lite value normalization; single-sync GPU telemetry batching.

**Concerns:**
- **[HIGH]** `PPOAgent` god-file with a ~900-line `update()` method.
- **[MED]** Dead/impossible defensive branch: `if … "has_fresh_contribution" in data:` — the key is always populated, so the else branch is unreachable (mild No-Defensive violation).
- **[MED]** GAE backward loop is per-env Python loops under `@torch.compiler.disable` (acknowledged O(num_envs·steps) TODO).
- **[LOW]** `CounterfactualMatrix.baseline_accuracy/full_accuracy` silently return 0.0 on missing config (inconsistent with fail-loud siblings).

**Commandment compliance:** C2 rent **n/a** (consumes returns); C3 GPU-first **upheld** (device-resident buffers, batched syncs, no locks/queues — caveat: per-env GAE Python loop); C5 Train Anything **upheld** (no host types; only lazy-imports the *policy* network); No-Legacy **upheld** (checkpoint loader fails hard, no migration); No-Defensive **partial** (the impossible `in data` guard + silent 0.0); Telemetry **upheld**.

✎ **Validator corrections merged:** removed `leyline/policy_protocol.py` and `simic/training/config.py` from inbound (not importers); added emitters.py/epoch_runner.py/vectorized_trainer.py; field name is `ppo_update_performed`.

**Confidence:** High.

---

## Simic-Rewards — The Economy  ·  ~2.5K LOC  ·  *validation: PASS*

**Responsibility:** Computes the per-step/terminal scalar reward across **seven** modes, translating counterfactual contribution, accuracy delta, parameter rent, and lifecycle shaping into one signal plus a typed component breakdown.

**Key components:**
- `contribution.py` (1,514 — **god-file**): `ContributionRewardConfig` (50+ tuned params), `RewardMode` enum (7 modes), `FossilizedSeedDripState`, and 6 mode implementations. `compute_contribution_reward` (~377–873, ~500 lines, fan-in 165) implements SHAPED+ESCROW; `compute_basic_reward`, sparse, minimal.
- `rewards.py` (239): `compute_reward` dispatch on `config.reward_mode`; family selection.
- `types.py` (145): `SeedInfo` NamedTuple (+ `from_seed_state` adapter from kasmina), `ContributionRewardInputs`.
- `reward_telemetry.py` (266): `RewardComponentsTelemetry` (~40 fields summing to `total_reward`; `shaped_reward_ratio` hacking diagnostic).
- `shaping.py` (157): single canonical `STAGE_POTENTIALS` (Ng et al. 1999 PBRS); `loss_primary.py` (72): loss-family reward.

**Dependencies:** Inbound — `action_execution.execute_actions` (primary production caller), `helpers.run_heuristic_episode`, Karn/Nissa telemetry consumers, extensive tests (the bulk of fan-in 165). Outbound — leyline, `nissa.get_hub`.

**Patterns:** Typed-inputs + enum dispatch; **single canonical PBRS potential table** with gamma hard-pinned to leyline (raises on mismatch to protect policy-invariance); counterfactual-first attribution (keys on clean leave-one-out, **not** host-drift-confounded total); component-telemetry contract (components sum to total); anti-gaming hardening (sigmoid discount, ratio penalty, age-gated prune bonus).

**Concerns:**
- **[HIGH]** `contribution.py` 1,514-LOC god-file with a ~500-line single function (9+ reward components ✎ — validator's recount; in-code comment lists 7).
- **[MED]** 7 reward modes with **3 different rent formulas** (log-growth-ratio in SHAPED, flat param_penalty in BASIC, again in loss_primary).
- **[MED] SIMPLIFIED mode omits structural rent entirely** — if a run is configured SIMPLIFIED, C2 ("complexity pays rent") is **not** exercised though the same entry point is used.
- **[MED]** ~50 hand-calibrated magic floats embedded in code (e.g. `alpha_shock_coef=0.1958`) rather than leyline.
- **[LOW]** Lazy intra-function telemetry imports on the reward hot path; dead `INTERVENTION_COSTS.get(action, 0.0)` over a closed enum.

**Commandment compliance:** C2 rent **upheld** (param-ratio rent wired into every economy-relevant mode; SPARSE is literal accuracy−rent — caveat: SIMPLIFIED self-declares non-evidence); C6 plane **upheld** (one `STAGE_POTENTIALS`); No-Legacy **upheld**; No-Defensive **upheld**; Telemetry **upheld**.

✎ **Validator corrections merged:** removed malformed `reward_telemetry.py:861` citation; noted "9+" is the validator recount vs the in-code "7".

**Confidence:** High.

---

## Simic-Training — Vectorized Orchestration  ·  ~11K LOC  ·  *validation: PASS*

**Responsibility:** Orchestrates vectorized multi-environment PPO where each environment is a host network mid-training (1 RL step = 1 host epoch) — rollout collection, fused validation/counterfactual passes, lifecycle action execution, PPO updates, typed telemetry.

**Key components:**
- `vectorized_trainer.py` (3,033 — **GOD-FILE**): `VectorizedPPOTrainer` (~80-field dataclass) with the per-epoch hot loop. `run()` L2910; `_run_batch()` ~420 lines; `_run_fused_val_pass()` ~466 lines; `_build_action_inputs()` ~399 lines. Owns CUDA-stream discipline.
- `vectorized.py` (1,490): `train_ppo_vectorized()` entry point + setup/PPO-update driver. **Not** overlapping the trainer — entry/setup vs stateful per-batch executor (boundary is leaky — see debt).
- `action_execution.py` (1,634): `execute_actions()` applies factored lifecycle actions; `ResolveTargetSlot` Protocol (host-agnostic boundary); reward computation via `compute_reward`; governor approval gating.
- `parallel_env_state.py` (303): `ParallelEnvState` — per-env CUDA stream + device-resident pre-allocated accumulators (**the structural basis of GPU-first**).
- `telemetry/emitters.py` (1,362): `VectorizedEmitter` builds frozen typed leyline payloads.
- `ppo_coordinator.py` (580), `helpers.py` (1,042, `compute_rent_and_shock_inputs` wires the rent input), `config.py` (512), `handlers/*`, `dual_ab.py` (319).

**Dependencies:** Inbound — `scripts.train.main`, `dual_ab`, benchmark/profile/smoke scripts, tests. Outbound — leyline, simic.agent, **simic.rewards (the reward=accuracy−rent formula lives here)**, simic.control, tamiyo.policy, nissa, tolaria (TolariaGovernor), kasmina.host, utils.data.

**Patterns:** Inverted control flow (one DataLoader → per-env CUDA streams); **device-resident accumulators + single epoch-end sync** (one `stream.synchronize()` per env before any `.item()`); explicit CUDA stream correctness (`wait_stream`/`record_stream`); shared AMP precision factory; fail-fast over defensive masking; dependency-injection of callables to keep the giant class testable.

**Concerns:**
- **[HIGH]** `vectorized_trainer.py` 3,033-LOC god-file; ~80-field class; 200–466-line methods; 8-tuple returns thread state.
- **[MED] PPO metrics flow as untyped `dict[str, Any]`** through the update/telemetry path; `metrics.get(key, default)` everywhere — a renamed/dropped key silently degrades to a default (borderline no-silent-fallback). The payload *is* typed only at the final emit boundary.
- **[MED]** Reflective `dataclasses.fields` injection in the emit hot path (a rename silently skips injection).
- **[LOW]** Narrowed-but-swallowed exception for optional gradient stats; documented resume-seam gap (iterator cursor not serialized); tqdm `RLock` (display only, off hot path).

**Commandment compliance:** C3 GPU-first **upheld** (per-env streams, single sync per phase, batched `.cpu()`, no queue/lock on compute path — verified by reading the sync code, not just comments); C2 rent **partial** (wires the input; formula in rewards); C5 Train Anything **n/a** (binds Simic; uses nn only for `CrossEntropyLoss`); C7 Governor **upheld** (per-env field, approval-gate only, no disable path); No-Legacy **upheld**; No-Defensive **partial** (the untyped metrics-bag `.get()` defaults); Telemetry **partial** (strong at the wire; loose producer-side).

✎ **Validator corrections merged:** `_run_batch` tuple at L2899-2908; `AnalyticsSnapshotPayload` at telemetry.py:1668; "factory/setup" reworded to avoid the factory metaphor.

**Confidence:** High.

---

## Tolaria — Metabolism / Execution + Safety Governor  ·  869 LOC  ·  *validation: WARN*

**Responsibility:** A fail-safe watchdog (`TolariaGovernor`) that detects catastrophic host-training failures (NaN/Inf, lobotomy-to-random-guess, statistical divergence), rolls back to a last-known-good RAM snapshot, vetoes unsafe lifecycle mutations, and signals RL punishment — plus a model/device factory.

**Key components:**
- `governor.py` (629): `TolariaGovernor`. `check_vital_signs` 3-tier catastrophe detection with consecutive-panic debounce; `snapshot` offloads host+fossilized-seed state to pinned CPU buffers with a STOP-SNAP post-copy sync; `execute_rollback` restores host weights *first* then prunes live seeds; `preflight_lifecycle_mutation` vetoes before Simic applies side effects.
- `environment.py` (175): `validate_device` (canonical fail-fast device validator), `create_model` (builds `MorphogeneticModel` via TaskSpec). No host-specific nn imports.
- `__init__.py` (65): PEP-562 lazy-import facade.

**Dependencies:** Inbound — simic.training (env_factory constructs it; vectorized_trainer calls check_vital_signs/snapshot; action_execution calls execute_rollback/preflight; ppo_coordinator injects punishment reward; parallel_env_state holds + resets). Outbound — leyline, nissa (lazy, best-effort telemetry), kasmina.host (TYPE_CHECKING/duck-typed), runtime.

**Patterns:** Inverted control flow (governor is a *passive object driven by Simic's loop*, never awaits Simic); protocol-based boundary; fail-fast configuration (rejects `history_window < MIN_GOVERNOR_HISTORY_SAMPLES` rather than silently disabling detection); **safety-ordered rollback** (restore host weights before non-critical pruning; telemetry last as best-effort); documented stream contracts (snapshot requires default stream; rollback tolerates any).

**Concerns:**
- **[LOW] Cross-metaphor naming collision: "governor" overloaded.** `allow_governor_override` (action_masks.py:156, docstring :171) refers to the **alpha-schedule HOLD governor**, NOT the safety watchdog — could mislead a reader into thinking the policy can override the safety governor (it cannot).
- **[LOW]** Blocking CUDA syncs in `snapshot()` (every 5 epochs) and `execute_rollback()` (full device sync) — documented, confined to periodic/rare paths.
- **[LOW]** Broad `except Exception` around seed prune during rollback (stderr-only, no telemetry event — silent-degradation seam, but host already safe).
- **[LOW]** Optimizer-state clearing is the caller's responsibility — safety invariant split across two modules.

**Commandment compliance:** **C7 Governor un-disableable — upheld** (constructed unconditionally per-env; no enable/disable/skip/bypass flag exists anywhere by grep; panic decisions computed purely from loss stats; Tamiyo supplies zero input; `__init__` fails fast on detection-weakening configs); C3 GPU-first **partial** (control flow inverted, but snapshot/rollback force real syncs on periodic/rare paths); C5 Train Anything **upheld**; No-Legacy **upheld**; No-Defensive **partial** (the broad prune-except seam); Telemetry **upheld** (frozen typed `GovernorRollbackPayload`, single-authority rule enforced).

✎ **Validator corrections merged (significant):** commandment **renumbering** — Train Anything is **C5** (not 4), Governor is **C7** (not 6); the No-Legacy/No-Defensive/Telemetry items are **CLAUDE.md rules, not ROADMAP commandments**; pattern citations corrected from `vectorized.py` to `vectorized_trainer.py`.

**Confidence:** High.

---

## Nissa — Sensory / Telemetry Hub  ·  3,153 LOC  ·  *validation: WARN*

**Responsibility:** Receives carbon-copy typed telemetry events from all domains and fans them out via per-backend worker threads to display/aggregation backends (console, JSONL, wandb, blueprint analytics); provides the in-process `DiagnosticTracker` (gradient/loss-landscape/per-class sensors).

**Key components:**
- `output.py` (986 — **god-file**): `NissaHub` (async fan-out, main worker + per-backend `BackendWorker` threads, bounded queues, drop-on-overflow back-pressure); `ConsoleOutput`/`FileOutput`/`DirectoryOutput`; global singleton (`get_hub`/`emit`). `NissaHub.emit` is the domain fan-in point.
- `tracker.py` (682): `DiagnosticTracker` — backward hooks, per-layer gradient stats (one batched sync), sharpness via weight perturbation; None-vs-0.0 discipline.
- `analytics.py` (505): `BlueprintAnalytics` — `BLUEPRINT_COMPUTE_MULTIPLIERS` rent/compute-cost table (a C2 surface).
- `wandb_backend.py` (591), `config.py` (275, Pydantic `extra='forbid'`).

**Dependencies:** Inbound — simic.telemetry.emitters, simic.training, tamiyo.tracker, **simic.rewards.contribution (get_hub), simic.training.env_factory + action_execution (BlueprintAnalytics), tolaria.governor (get_hub)** ✎, karn.collector, scripts. Outbound — leyline, torch/numpy (tracker), pydantic/yaml, wandb (soft import).

**Patterns:** Typed-payload contract (`TelemetryEvent.data: TelemetryPayload | None`; isinstance-discriminate before access); **absent != zero discipline** (None renders 'n/a', never fabricate 0.0); protocol-based decoupling (`OutputBackend`); thread isolation / back-pressure (per-backend bounded queues, `put_nowait` drop-on-overflow); fail-fast Pydantic config.

**Concerns:**
- **[HIGH]** `output.py` 986-LOC god-file with a 134-line ✎ `_emit_summary` if/elif ladder over event-type strings (must stay in sync with the 31-member enum).
- **[MED] C1 coverage gap:** 6 event types (`PHASE_PROFILE_COMPLETED`, `MORPHOLOGY_CAUSAL_LOG`, `TOPOLOGY_MANIFEST_RECORDED`, `COUNTERFACTUAL_MATRIX_COMPUTED`, `EPISODE_OUTCOME`, `ALLOCATOR_STATS`) have **no structured Nissa-side handler** in analytics/wandb (only persisted by the separate Karn store). ✎ Console *does* render a generic line via an else-branch, so the operator isn't fully blind — the silent-drop is specific to `BlueprintAnalytics.emit`.
- **[MED]** Dead defensive `hasattr(event.event_type,'name')` guard (the enum is guaranteed); soft-fail log-and-return on payload mismatch in analytics/wandb vs hard-raise in output.py (inconsistent enforcement).
- **[LOW]** Global singleton hub created without locking; `DiagnosticTracker.__del__` relies on GC for hook removal.

**Commandment compliance:** **C1 Sensors match capabilities — partial** (rich gradient/landscape sensors, but the display/analytics tier ignores 6 event types with no structured handler); C2 rent **upheld** (`BLUEPRINT_COMPUTE_MULTIPLIERS`, `compute_cost`); No-Legacy **upheld**; No-Defensive **partial** (the unreachable hasattr guard + log-and-return); Telemetry **upheld** (strongest in this unit — typed data, isinstance-discriminate, raise on bad payload).

✎ **Validator corrections merged:** added the 4 missing inbound edges (notably the **tolaria→nissa** edge — the only cross-domain Tolaria dependency); `_emit_summary` is 134 lines; console else-branch caveat on C1.

**Confidence:** High.

---

## Karn — Memory / Operator UI + Analytics + Persistence  ·  22,539 LOC (largest)  ·  *validation: PASS*

**Responsibility:** Consumes the typed telemetry event stream and turns it into live TUI/web dashboards (Sanctum/Overwatch), a lossy in-memory analytics store with JSONL export, and a post-hoc DuckDB SQL surface (`esper-karn` MCP).

**Key components:**
- `sanctum/aggregator.py` (2,277 — **god-file**): `SanctumAggregator` — thread-safe stateful reducer folding the event stream into `SanctumSnapshot`; typed-payload dispatch raising `TypeError` on mismatch; Pareto/hypervolume sensors.
- `sanctum/schema.py` (1,717 — **god-file**): the `SanctumSnapshot` view-model — ~20 dataclasses (`EnvState`, `TamiyoState`, `RewardComponents` L1253 ✎, `DecisionSnapshot`) + pure analytic helpers.
- `store.py` (1,098): `TelemetryStore` — a **separate** in-memory analytics model; honestly self-describes as NON-proof-grade.
- `mcp/views.py` (896): `VIEW_DEFINITIONS` — ~13 DuckDB SQL views over `events.jsonl`. `mcp/server.py` (332): `KarnMCPServer` (FastMCP) — marked **ORPHANED** from live training.
- `collector.py` (777): `KarnCollector` (an OutputBackend feeding the store + anomaly detectors); `overwatch/backend.py` (431): WebSocket server reusing the same `AggregatorRegistry`; `ingest.py` (281): coercion (rejects bool-as-int, no legacy aliases); `pareto.py` (99): the param-ratio Pareto sensor.

**Dependencies:** Inbound — scripts/train.py, nissa (OutputBackend contract), simic/training (env_factory, parallel_env_state import karn types). Outbound — leyline, nissa.output, duckdb/psutil/numpy/textual, mcp (guarded import).

**Patterns:** Stateful stream-reducer; **single shared view-model, two front-ends** (Sanctum TUI + Overwatch web share one aggregation model — no logic duplication); typed-payload enforcement (raise on mismatch); fail-fast telemetry boundary (latches first exception into a fatal modal rather than swallowing); honest lossiness (store enumerates dropped proof-critical families); schema-as-SQL (DuckDB views evolve with payload fields, no server code change).

**Concerns:**
- **[HIGH] Two parallel, divergent telemetry data models** for the same stream: `store.py` `RewardComponents/EpochSnapshot/PolicySnapshot` vs `sanctum/schema.py` *different* `RewardComponents/EnvState/TamiyoState`. The store path is only `--export-karn`; the live UI uses sanctum. Duplicate modeling of one contract — drift risk.
- **[HIGH]** Three god-files >1,000 LOC (aggregator 2,277, schema 1,717, mcp/views 896-data); largest widget `sanctum/widgets/tamiyo/action_heads_panel.py` ✎ at 955.
- **[MED]** MCP query/persistence surface is **ORPHANED** from live training (standalone post-hoc CLI, not wired as an OutputBackend).
- **[MED]** Backward-compat language on the live path: `get_snapshot()` "for backward compatibility… legacy callers", picks "first group alphabetically" (silent single-group assumption in an A/B system).
- **[LOW]** Stale "legacy" reward-component display fields; a ~30-line future-work TODO block inline in the god-file.

**Commandment compliance:** C1 Sensors **upheld** (broad sensor set surfaced — PPO health, lifecycle, counterfactual matrix, Shapley, Pareto frontier); C2 rent **upheld** (surfaces `compute_rent`/`ratio_penalty`/`alpha_shock`, Pareto minimizes param_ratio); No-Legacy **partial** (read-only fields + "backward compatibility" vocabulary contradict the HARD rule though no dual code paths); No-Defensive **upheld** (`.get()` are counter-accumulation defaults; the broad excepts are hardware probes setting explicit 'unavailable'); Telemetry **upheld** (isinstance→raise on the live path; fatal-latch; honest lossiness).

✎ **Validator corrections merged:** `RewardComponents` at schema.py:1253; largest-widget path is under `widgets/tamiyo/`; function is `copy_snapshot()` in `snapshot_copy.py`.

**Confidence:** High.

---

## Support — Platform Glue / System Wiring  ·  ~3.1K LOC  ·  *validation: WARN*

**Responsibility:** Assembles the running system from CLI args: parses flags, selects task/preset, builds host models + dataloaders, wires telemetry backends, dispatches into Simic entrypoints.

**Key components:**
- `scripts/train.py` (1,154): CLI entry. `build_parser()` (~40 flags across `heuristic`/`ppo` subcommands); `main()` validates UI modes, constructs the Nissa hub + backends, maps overrides onto `TrainingConfig`, dispatches to train_heuristic / train_ppo_vectorized / train_dual_policy_ab / proof-baselines. Sets `PYTORCH_ALLOC_CONF` before first CUDA init; forces spawn start method.
- `runtime/tasks.py` (699): `TaskSpec` + 10 factory functions + `get_task_spec(name)`; `VALID_TASKS` frozenset (single source of truth).
- `utils/data.py` (1,142): CIFAR loaders (`load_cifar10`, `load_cifar10_gpu` with device-keyed GPU cache), GPU augmentation, three multi-env iterators (`SharedBatchIterator`, `SharedGPUBatchIterator`, experimental gather variant), TinyStories.
- `utils/loss.py` (126): `compute_task_loss` + `compute_task_loss_with_metrics` (device-resident correct/total, no `.item()` sync on hot path).

**Dependencies:** Inbound — simic.training.helpers + vectorized, tolaria.environment, `scripts/phase5_reward_calibration` (✎ — **not** proof_packet), tests. Outbound — leyline, kasmina.host, tamiyo, simic.training/rewards/telemetry, nissa, karn, torch/torchvision/datasets.

**Patterns:** Factory + registry (`get_task_spec` behind `VALID_TASKS`, wired into argparse `choices`); single source of truth for hyperparameters (pulls `DEFAULT_*` from leyline); parent-parser composition; device-resident hot path; explicit clone-after-`tensor_split` to break view aliasing across CUDA streams; allocator policy set before first CUDA context (fail-loud warning if already initialized).

**Concerns:**
- **[HIGH] Silent fallback to synthetic random data masks dataset-load failures.** `get_cifar10_datasets`/`load_cifar10` wrap CIFAR10 construction in `except Exception: warnings.warn(...); return …(mock=True)` — a broad catch substitutes `torch.randn/randint` garbage. **A training run can proceed to completion on noise** with only a `warnings.warn`; accuracy telemetry becomes meaningless. This is exactly the bug-hiding pattern CLAUDE.md prohibits.
- **[MED]** `train.py` 1,154-LOC god-file; `main()` ~570 lines with a nested `run_training()` closure; duplicated ~30-line profiler-kwargs blocks and verbatim-duplicated preset-selection if/elif chains.
- **[LOW]** Defensive `.get()` on self-owned `dataloader_defaults`; hand-maintained `excluded_keys` denylist that must stay in sync with `to_train_kwargs()`; `get_datasets` `else: raise NotImplementedError` is a latent divergence from the `VALID_TASKS` registry.

**Commandment compliance:** C3 GPU-first **upheld** (device-resident correct/total; GPU iterators return on-device tensors; channels-last; the only `threading.Event` is TUI/dataloader setup, not hot path); No-Legacy **upheld** (sets *current* alloc-conf key; experimental gather gated behind `--experimental`); **No-Defensive — VIOLATED** (the synthetic-data fallback); Telemetry **partial** (wiring is explicit/typed, but the data-load fallback lets accuracy be computed on noise without a typed signal).

✎ **Validator corrections merged:** removed `scripts/proof_packet` from inbound (it does not call `get_task_spec`); alloc-conf citation split (setdefault 12-15, fail-loud guard 17-28).

**Confidence:** High.
