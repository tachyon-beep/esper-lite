# Obs-pipeline engineering audit (Tamiyo) — PyTorch/plumbing view

Tracker: esper-lite-f7b46a5f52. Author: pytorch-expert (SME). Scope: **analysis only, no code changes.**
Companion to the drl-expert sufficiency audit (they own "what SHOULD the policy see"; this owns
"what CAN it see, what exists-but-doesn't-reach-it, and what it costs to change").

Primary sources read first-hand (file:line cited throughout):
`src/esper/tamiyo/policy/features.py`, `src/esper/leyline/__init__.py:655–711`,
`src/esper/simic/control/normalization.py`, `src/esper/simic/training/normalizer_checkpoint.py`,
`src/esper/tamiyo/networks/factored_lstm.py`, `src/esper/simic/training/vectorized.py`,
`src/esper/simic/training/vectorized_trainer.py`, `src/esper/simic/training/action_execution.py`,
`src/esper/tamiyo/policy/action_masks.py`, `docs/analysis/2026-07-14-r9-archival-record.md`.

SME protocol: **confidence tag on every finding.** Confidence reflects how directly I verified it in
source (HIGH = read the exact lines; MED = strong inference from read code + one indirect step;
LOW = plausible, needs a runtime measurement to confirm).

---

## 0. Ground truth: the exact obs the network consumes

Two facts frame everything below and are both **HIGH confidence** (read end to end):

1. **Every feature is fixed-constant normalized inside `batch_obs_to_features`** (min-max, `/max_epochs`,
   `/100`, `/num_slots`, `symlog/7`, clamp to `[-1,1]`), then the *entire vector* is passed through a
   **running EMA normalizer** (`RunningMeanStd`, `momentum=0.99`) before the LSTM. There is no per-feature
   normalization discipline downstream — the running normalizer treats all 120 (V3) / 129 (V4) dims
   identically, sentinel and one-hot included.

2. **Acting and training consume the identical normalized tensor — there is NO train/act skew.**
   Acting loop: `batch_obs_to_features` (vectorized_trainer.py:2005) → raw stashed for the deferred
   normalizer update (`:2039`) → `obs_normalizer.normalize(states_batch)` (`:2042`) → `get_action` on the
   normalized tensor (`:2082`) → `buffer.add(state=states_batch_normalized...)` (action_execution.py:1585).
   The normalizer `update()` is deferred to end-of-rollout (vectorized.py:528), so stats are frozen within a
   rollout. This is correct hygiene; I flag it explicitly because it was the one plausible verdict-changer
   and it came back clean. **HIGH.**

### Exact layout (verified against `features.py:837–992`)

Base block, dims **0–23** (24 dims, `OBS_V3_BASE_FEATURE_SIZE`):

| dim | feature | normalization | notes |
|----|---------|---------------|-------|
| 0 | epoch_norm | `min(epoch,max_epochs)/max_epochs` | epoch exactly recoverable (source-proof Q3) |
| 1 | val_loss | `symlog(x)/7` | unbounded loss compressed |
| 2 | val_accuracy | `/100` | |
| 3–7 | loss_history[5] | `symlog/7`, absent→sentinel | left-padded; absent = `-1.0` |
| 8–12 | acc_history[5] | `/100`, absent→sentinel | |
| 13 | num_training | `/num_slots` | |
| 14 | num_blending | `/num_slots` | |
| 15 | num_holding | `/num_slots` | **HOLDING and FOSSILIZED are summed here** (features.py:825) |
| 16 | last_action_success | bool | op-only echo, one step (source-proof Q4) |
| 17–22 | last_action_op one-hot[6] | one-hot | sampled **slot never fed back** |
| 23 | stable_val_acc (escrow) | `/100`, None→sentinel | |

Per-slot block, **V3 = 32 dims** each, offset `= 24 + slot_idx*slot_feature_size` (features.py:874):

| off | feature | off | feature |
|----|---------|----|---------|
| +0 | is_active (always 1.0) | +16 | alpha_mode_norm |
| +1..+10 | stage one-hot[10] | +17 | alpha_steps_total_norm |
| +11 | current_alpha | +18 | alpha_steps_done_norm |
| +12 | counterfactual_contribution* | +19 | time_to_target_norm |
| +13 | contribution_velocity **⚠ DEAD→0, see §P0** | +20 | alpha_velocity |
| +14 | blend_tempo (`/12`) | +21 | alpha_algorithm_norm |
| +15 | alpha_target | +22 | interaction_sum (`/10`) |
| +23 | gradient_norm (`symlog/7`) | +27 | gradient_health_prev |
| +24 | gradient_health | +28 | epochs_in_stage_norm |
| +25 | has_vanishing | +29 | counterfactual_fresh (`γ^age`) |
| +26 | has_exploding | +30 | seed_age_norm |
| | | +31 | escrow_credit_prev (`symlog/7`) |

`*` dim +12 = the **V3 `None→0` coercion** (features.py:906–910). **V4** (`obs_v4=True`) rewrites +12 to
shrink toward the sentinel with staleness and appends **+32 cf_observed / +33 cf_frozen / +34 cf_age_norm**
(features.py:895–902, 989–992). V4 slot block = 35 dims → 129 total for 3 slots.

---

## 1. Signals that EXIST but never reach the obs

`batch_obs_to_features` reads **only** three per-env objects — `TrainingSignals` (`signal`),
`dict[str,SeedStateReport]` (`reports`), `ParallelEnvState` (`env_state`) — and only a fixed subset of each.
Anything held on those objects but not read, or computed elsewhere, never reaches the policy. I verified every
citation in this table first-hand (read the exact lines) and confirmed the obs non-read with a targeted grep of
`features.py` (only hit is the `:408` "REMOVED in V3" comment). **Plumbing tier** per §5: **Tier 0** = value
already on one of the three received objects (add a read + a leyline dim constant); **Tier 1** = value lives on
the agent/trainer/anomaly side and needs new per-env threading.

### The reachability gate (both sweeps converged on this — it is the load-bearing engineering fact)
A `report.metrics` field is reachable by the obs **only if `SeedMetrics.to_leyline()` (slot.py:281–301)
copies it.** The trainer mutates the *kasmina-internal* `SeedMetrics`; the obs reads the *leyline snapshot*
rebuilt each epoch by `to_report()`→`to_leyline()`. `to_leyline` copies 19 fields — so `boost_received`,
`upstream/downstream_alpha_sum`, `seed_gradient_norm_ratio`, `gradient_norm_avg`, `seed_param_count`,
`host_param_count` etc. **are** genuinely Tier-0 reachable (they appear at slot.py:291–300). The one field the
leyline dataclass declares but `to_leyline` forgets is `contribution_velocity` → **that is the §P0 dead dim.**
So "Tier 0" for a `report.metrics` field means *both* "declared on `LeylineSeedMetrics`" *and* "copied by
`to_leyline`"; a new per-slot signal generally needs a `to_leyline` line too, not just a leyline field.

### Top exists-but-unplumbed inventory (ranked by host-informativeness; both sweeps merged + spot-verified)

| # | signal | computed at (file:line) | on which obj | tier | conf |
|---|--------|-------------------------|--------------|------|------|
| 1 | **`committed_val_acc` (+ `committed_acc_history`)** — host accuracy with **only FOSSILIZED seeds**; the provisional-vs-committed spread. Arguably the single most decision-relevant signal for a fossilize/prune policy; obs carries only the all-seeds `val_accuracy` (dim 2) | set `vectorized_trainer.py:1422`; `parallel_env_state.py:73-74` | `env_state` | 0 | HIGH |
| 2 | **`plateau_epochs`** — the exact germination trigger the heuristic controller keys on; RL policy is blind to it | `tracker.py:126-130`→packed `:195`; `signals.py:45`; consumer `heuristic.py:200` | `signal.metrics` | 0 | HIGH |
| 3 | **`train_accuracy`/`train_loss`** → train−val **generalization gap** (overfitting) + signed `loss_delta`/`accuracy_delta` | `tracker.py:186-194`; `signals.py:36,41`; `parallel_env_state.py:68-69` | `signal.metrics` / `env_state` | 0 | HIGH |
| 4 | **Full counterfactual structure** — pairwise `I_ij` matrix, all-off synergy baseline, solo-ON accs, Shapley **`std`** (attribution uncertainty), `InteractionTerm.`**`regime`** (synergy/interference/recovering). Policy gets one clamped LOO scalar (dim 12) + one summed `interaction_sum` (dim 22) where an uncertainty-tagged, regime-classified vector exists one reduction upstream | live matrix ephemeral in `vectorized_trainer.py:1360-1420`; analytics `counterfactual.py:157,175-182`, routed to telemetry `:1552-1559` | trainer locals / analytics | B/1 | HIGH |
| 5 | **`boost_received`, `upstream_alpha_sum`, `downstream_alpha_sum`** — interaction topology; **computed + copied by `to_leyline` (:298-300) yet explicitly REMOVED in V3** (features.py:408). Sibling `interaction_sum` DOES reach obs (dim 22) — asymmetry | `vectorized_trainer.py:1492-1506,1548-1549`; `reports.py:62,64,66` | `report.metrics` | 0 | HIGH |
| 6 | **`seed_gradient_norm_ratio`** (G2-gate "seed learning vs riding host", param-normalized) + **`gradient_norm_avg`** (smoothed grad, distinct from instantaneous dim 23) | `vectorized_trainer.py:1894-1902`; `slot.py:1972-1973`; copied `slot.py:291,294` | `report.metrics` | 0 | HIGH |
| 7 | **`seed_param_count`/`host_param_count`** — reward optimizes committed-gain-**per-param** (`contribution.py:736-744`) but policy sees **no** param count / capacity | `kasmina/slot.py:1431-1435`; copied `slot.py:292-293` | `report.metrics` | 0 | HIGH |
| 8 | **Pre-digested per-slot health**: `is_healthy`/`is_improving`/`needs_attention`; **`previous_stage`/`previous_epochs_in_stage`** (annotated "for PBRS") | `slot.py:572-574`, `:559-560` | `report` (top-level) | 0 | HIGH |
| 9 | **`host_max_acc`** + host **`best_val_loss`/`best_val_accuracy`** — regret / performance-ceiling; used by sparse reward, unobserved | `vectorized_trainer.py:1845-1847`; `tracker.py:197-199` | `env_state` / `signal.metrics` | 0 | MED-HIGH |
| 10 | **Gradient-drift EMA** (`norm_drift`/`health_drift`) — smoothed degradation vs. obs's 1-step lag (dim 27) only | `simic/telemetry/gradient_ema.py:70-71`; consumed `anomaly_detector.py:345-378` | telemetry/anomaly side | 1 | MED |
| 11 | **`acc_at_germination`** — the policy's own germination baseline (reward reference), unobservable to it; **raw dual-gradient norms** (`host_grad_norm`/`seed_grad_norm`, `gradient_collector.py:497-502`) collapsed to the ratio EMA and dropped | `action_execution.py:1025`; `parallel_env_state.py:76` | `env_state` / trainer locals | 0 / B | MED |

*(`total_improvement`/`improvement_since_stage_start` are reachable but the team deliberately gates on the
counterfactual instead — repeatedly labeled host-drift-confounded, `contribution.py:477-489,829-839`. Available,
but already reasoned out of the causal path — do not top-rank.)*

### Two structural observations from the inventory

- **The reduction sites are the real story.** The system computes a *full* attribution structure and throws away
  the rich part before the policy sees anything: the per-batch pairwise interaction matrix `I_ij` is collapsed to
  a per-seed running sum `interaction_sum += interaction` (vectorized_trainer.py:1491) → obs dim +22; and the
  episode-analytics `CounterfactualEngine` computes a `CounterfactualMatrix` (2ⁿ factorial), `ShapleyEstimate`
  (with **`std`**), and `InteractionTerm.regime` (counterfactual.py) that reach **telemetry only**. The policy is
  handed one clamped scalar where an uncertainty-tagged, regime-classified vector exists one reduction upstream.
  Whether it *should* get more is the drl audit's call; the engineering fact is the richer signal is **already
  computed** (Tier 0/1, not new instrumentation).
- **A reward/observation asymmetry.** The reward optimizes committed-gain-per-param (`host_params` denominator,
  `contribution.py:736-744`) and rolls in `plateau`, `loss_delta`, param counts — **none of which the policy can
  observe.** Six of the top-10 unplumbed items feed the reward or the heuristic controller but not the RL obs.
  This is the single highest-leverage framing for the drl audit: the policy is asked to optimize a target built
  from signals it cannot see.

### Correctly gate-excluded (inventoried, but MUST stay out of obs)
`TolariaGovernor` state (`tolaria/governor.py`; on `ParallelEnvState.governor`, parallel_env_state.py:43) and
`AnomalyDetector` reports (`simic/telemetry/anomaly_detector.py`) gate/divert the policy (rollback + punishment
reward), so their absence from the obs is **correct** — a controller must not observe the gate that overrides it
(the governor-independence invariant). Noted precisely because the governor is on `ParallelEnvState`, i.e.
**Tier-0-cheap to plumb but must not be** — a standing temptation to flag so nobody "helpfully" wires it in.

### Not gaps (verified reads — do not re-list as missing)
`stable_val_acc_for_observation` DOES reach obs (base dim 23, `_stable_val_acc_feature` features.py:177-182,869);
`epochs_since_counterfactual` → `counterfactual_fresh` (dim 29); `gradient_health_prev` → dim 27;
`contribution_states` is **flag-gated, not never-wired** — live under `obs_v4=True` (dims 12/32/33/34), maintained
but unread under the default V3.

### Do NOT recommend (vestigial / removed-by-design / confounded — flagged so nobody re-adds them blind)
- **`gpu_memory_used`, `gpu_utilization`, `epoch_duration_ms`** (`signals.py:71-72,79`): declared on
  `TrainingSignals` but **no write site populates them** — they sit at defaults. Not actually computed per-epoch.
- **`host_stabilized`** (`tracker.py:196`): computed + reachable but **deliberately removed** from obs
  ("Tamiyo learns stability from raw telemetry", features.py:305). Re-adding reverses a prior call.
- **`prev_slot_params`/`prev_slot_alphas`** (`parallel_env_state.py:107-108`): reset each episode, **no per-epoch
  populate site confirmed** — treat as declared-but-unpopulated.
- **`alpha_ramp_step`** (copied `slot.py:296`): reachable but **duplicates dim +18** (`alpha_steps_done`).

*Note: sweep-1 (raw gradient stats / vectorized.py per-epoch signals) and sweep-2 (SignalTracker / governor /
capacity) independently converged on items 1-9 above with matching file:line — a strong cross-check. Every cited
line in this section was spot-verified first-hand; the §P0 dead-dim, `committed_val_acc`, `plateau→heuristic`,
the `to_leyline` field list, and the `interaction_sum += interaction` reduction were re-derived from source.*

---

## 2. Pipeline pathologies (ranked by engineering blast radius)

### P0 — **DEAD OBS DIM (verified defect):** per-slot `contribution_velocity` (dim +13) is silently constant `0.0`  *(HIGH — read every link in the chain)*
A reserved, believed-live observation dimension is fed a constant. The chain, verified end to end:
- The trainer computes it every epoch as an EMA of Δcounterfactual and writes the **kasmina-internal**
  `SeedMetrics`: `seed_state.metrics.contribution_velocity = 0.7*... + 0.3*delta` (vectorized_trainer.py:1385–1389).
- The obs reads the **leyline snapshot** the report carries: `velocity = report.metrics.contribution_velocity`
  (features.py:913, dim `slot_offset+13`), and the report is rebuilt each epoch from `SeedMetrics.to_leyline()`.
- **`to_leyline()` (slot.py:281–301) never copies `contribution_velocity`.** It constructs `LeylineSeedMetrics(...)`
  with 19 fields; `contribution_velocity` is not among them (grep of slot.py: the field is touched only at :163
  default, :323 `to_dict`, :377 `from_dict` — the checkpoint path, never the leyline boundary). The leyline
  `SeedMetrics.contribution_velocity` therefore stays at its dataclass default `0.0` (reports.py:56).

Net: **3 obs dims (one per slot) are hardwired to 0** — a fossilize-lookahead signal (`contribution_velocity`,
"EMA of Δcontribution for fossilize lookahead", reports.py) that the policy is completely blind to, while the
pipeline pays to compute it. This is present in **both V3 and V4** (dim +13 is read outside the `if obs_v4`
branch). Nothing in code or comments justifies the omission → **plumbing bug, not a design choice.**
**One-line fix** (out of scope here — reporting only): add `contribution_velocity=self.contribution_velocity`
to the `LeylineSeedMetrics(...)` call at slot.py:301. **Recommend the F2 implementer fix this in the same
touch** — they will be editing this exact per-slot block. Verification for the fixer: assert
`report.metrics.contribution_velocity != 0` for a slot with ≥2 counterfactual measurements, or dump
`obs_normalizer.var[slot_offset+13]` (currently ≈0).

### P1 — Running EMA normalizer runs over the `-1.0` sentinel and all one-hot/binary dims  *(HIGH the mechanism; MED the impact)*
`obs_normalizer = RunningMeanStd((state_dim,), momentum=0.99)` (vectorized.py:1092) normalizes the whole
vector, including: the `OBS_V3_UNKNOWN_SENTINEL = -1.0` (`leyline:681`) placed *deliberately* outside the
`[0,1]` measured range of the history / cf-fresh / gradient-health dims; the 6-dim op one-hot; the 10-dim
stage one-hot; `is_active` (constant 1.0 for occupied slots); and the binary `has_vanishing/has_exploding`.

Two distinct effects — stated precisely to avoid overclaim:
- **The sentinel's distinguishability SURVIVES normalization.** `(x-mean)/std` is affine and monotone, so a
  `-1.0` sentinel stays an outlier relative to the `[0,1]` measured mass; it is *not* "erased." The only cost
  is that the decision boundary between "unknown" and "measured" becomes data-dependent and drifts with the
  EMA. Do **not** claim the normalizer destroys the sentinel — it doesn't.
- **Dead/sparse dims spike the clip on first activation — but only after long dormancy.** For a dim held
  constant, EMA variance decays geometrically `≈0.99^N`; reaching the `epsilon=1e-4` floor (denominator
  `sqrt(1e-4)=0.01`, a ~100× amplifier) takes **N≈900 normalizer updates** (`ln(1e-4)/ln(0.99)≈916`). So the
  ±10-clip spike on a dim's first real reading is acute **only** for dims dead across ≳900 updates —
  e.g. `cf_frozen` (0 for both r9 seeds), or a `pending` bit if commits are rare enough. A dim dead only a
  few dozen rollouts has `var≈0.99^30≈0.74` → a 0→1 flip normalizes to ≈1.16, no spike. **Verification for
  the implementer:** dump the per-dim `obs_normalizer.var` vector and count `obs_normalizer.update` calls
  before the first activation.
- **EMA attenuates a *persistent* signal only if its base rate rises.** A rare-but-persistent bit (pending
  held for W epochs, commits infrequent) keeps a low EMA mean, so `pending=1` normalizes to a stable positive
  every step — persistence preserved. It only erodes if the bit becomes common (base rate climbs → mean
  chases it → signal fades). Relevant to F2 (see §3).

### P2 — `num_holding` conflates HOLDING + FOSSILIZED into one base dim  *(HIGH)*
features.py:825 increments `num_holding` for `_HOLDING_VAL` **and** `_FOSSILIZED_VAL`. Base dim 15 therefore
cannot distinguish "seed parked in HOLDING (still reversible)" from "seed permanently fossilized." The
per-slot stage one-hot *does* separate them, so this is a base-level aliasing that a per-slot reader can
recover — but any policy behaviour keyed on the *count* of committed vs. holding seeds is reading a summed
channel. Cheap to split (add a 4th stage-count dim) if the drl audit wants it; costed in §5.

### P3 — Collapsed alpha heads imply low-variance input echoes (dead-dim suspects)  *(MED code / LOW empirics)*
r9 archival record (`2026-07-14-r9-archival-record.md:19`) measured **`alpha_speed` + `alpha_curve` heads
COLLAPSED** (max-prob 1.00, one action always). This is **learned degeneracy, not a mask artifact**:
`alpha_speed_mask`/`alpha_curve_mask` are `torch.ones(...)` — all-True causal masks that never restrict
(action_masks.py:266–267). Causal direction is **one-way**: collapsed heads → the alpha-schedule params they
choose are constant → their *echoed input dims* go low-variance, specifically `alpha_steps_total_norm` (+17)
and `alpha_algorithm_norm` (+21). It does **not** run the other way — the heads see the full shared LSTM
trunk, so two low-variance input dims cannot *explain* the collapse (the likelier cause is the documented
policy-wide floor gradient dead-zone, out of scope here). Dims that vary over schedule progression
(`alpha_steps_done` +18, `time_to_target` +19, `alpha_velocity` +20) are **not** expected constant.
**Answer to the audit's factual question — "are their input features informative or constant?":** +17 and +21
are the constant suspects; the rest are live. **Verification is trivial and I recommend it before any dim is
cut:** dump `obs_normalizer.var` per dim and flag dims with `var < ~1e-3` as effectively dead. A dead *input*
dim is near-free to leave in place; the finding matters only if the drl audit proposes replacing it.

### P4 — `interaction_sum` is the only cross-slot signal, and it is a scalar `/10` clamp  *(MED)*
Per-slot dim +22 (`interaction_sum/10`, clamped `[-1,1]`, features.py:929–931) is the sole channel carrying
"how this slot interacts with others." The richer pairwise interaction matrix `I_ij` **is** computed upstream
and **collapsed to this running sum** at `seed.metrics.interaction_sum += interaction` (vectorized_trainer.py:1491;
`boost_received = max(...)` at :1492 is dropped entirely, §1 #5). Flagged as a pipeline narrowing; whether it
*should* be widened is the drl audit's call.

### P5 — `float32` obs vs `bf16` rollout autocast  *(MED)*
The obs tensor is built `float32` (features.py:798) and `get_action` runs under `rollout_autocast()`
(vectorized_trainer.py:1799, 3031). Sentinel `-1.0`, one-hots, and `symlog` values are all exactly
representable in bf16, so no precision landmine today — but any *new* small-magnitude feature (e.g. a
`1/max_epochs`-scale countdown) would lose mantissa under bf16 autocast. Note for future feature designers.

---

## 3. Schema / versioning engineering — the F2 landmine list

How the switch works: `obs_v4` is a plain `bool` threaded end-to-end —
`TrainingConfig.obs_v4_contribution_state` (config.py:138) → `vectorized.py:787,1088,1194,1620` →
`batch_obs_to_features(..., obs_v4=)` and `get_feature_size(slot_config, obs_v4=)`. Layout is **fully
data-driven** from two leyline constants: `obs_dim = OBS_V3_BASE_FEATURE_SIZE + slot_feature_size*num_slots`
(features.py:791; identical formula in `get_feature_size` features.py:736). The per-slot write loop keys every
write off `slot_offset = BASE + slot_idx*slot_feature_size` (features.py:874). **This is the crux of every F2
landmine below.** All HIGH confidence (read the exact lines).

**F2 = one per-slot `pending_settlement` bool (+ explicit schema version), per PDR-0098 / source-proof §1.**
The implementer MUST apply the full edit set atomically — a half-application is a silent layout corruption:

1. **Add a new offset constant** in leyline, `OBS_V5_SLOT_OFFSET_PENDING = 35` (next free per-slot index).
2. **Bump `slot_feature_size` 35→36 in lockstep.** ← **THE off-by-one landmine.** `slot_offset` is
   `slot_idx * slot_feature_size`. If you write `obs[.., slot_offset+35]` without bumping the size, slot 1's
   `is_active` at `24 + 1*35 = 59` **collides with slot 0's pending bit at `24 + 35 = 59`.** Every slot after
   the first silently overwrites the previous slot's new dim. There is no shape check that catches this — the
   total `obs_dim` is unchanged, so `_validate_obs_normalizer_shape` (normalizer_checkpoint.py:41) passes.
   The only symptom is garbage training. Bump the constant and the size **in the same commit**.
3. **Bump the schema version** (`OBS_V4_FEATURE_SCHEMA_VERSION = 3`, leyline:695 → a new V5 value). This is
   what makes `restore_obs_normalizer_from_metadata` reject a stale-contract resume (normalizer_checkpoint.py:101,
   contract-equality check) and **force a re-warm** — exactly the desired behaviour, not a bug to route around.
4. **`base_feature_size` is hardcoded to `OBS_V3_BASE_FEATURE_SIZE` in the contract** (normalizer_checkpoint.py:32).
   Fine for a *slot-only* extension like F2. But it is a latent landmine: anyone who later adds a **base** dim
   must also update the contract, or the shape validation and the actual `obs_dim` silently disagree.
5. **Write-loop symmetry:** the V4 status dims are written under `if obs_v4:` at features.py:989–992. The
   pending bit needs the same guarded write **plus** initialization on the empty-slot path — slots with
   `report is None` `continue` at features.py:878 leaving pre-zeroed values, so `pending=0` for empty slots is
   already correct by construction (obs is `torch.zeros`), but confirm the intended empty-slot value is 0.

**Cost of the bump (all HIGH):**
- **Re-warm/retrain, mandatory.** `state_dim` changes → the network's first `Linear(state_dim+bp_embed, ...)`
  (factored_lstm.py:380) changes shape → weights are not resumable. The obs normalizer `mean/var` are
  `state_dim`-shaped and correctly rejected on contract mismatch (forces reset).
- **Checkpoint compat:** a V3/V4 checkpoint cannot resume into V5; the contract-equality guard already fails
  loudly with a clear message (normalizer_checkpoint.py:101–106). Good — no silent load of stale stats.
- **Byte-identity discipline:** V3 stays byte-identical only with the flag OFF. Arm-A of any A/B must emit the
  new dim as a **constant** (source-proof §3: "arm A emits a constant 0, schema identical across arms") — i.e.
  both arms run the SAME (larger) schema, arm-A's pending bit hardwired to 0, so the only difference is the
  value in that one dim. Do NOT gate the dim's *existence* on the arm, or the two arms have different
  `state_dim` and the comparison is confounded by network width.

**Architecture verdict:** the per-slot extension path is **clean and data-driven** (two constants + one guarded
write), *provided* step 2 is not forgotten. It does **not** invite off-by-one bugs structurally — the single
formula is the single point of truth — but it offers **no compile-time guard** if the two constants drift. If
the team expects repeated slot-dim extensions (V5, V6…), the cheap hardening is an `assert slot_feature_size ==
1 + NUM_STAGES + <count of per-slot writes>` at import time in leyline, so a forgotten bump fails fast.

---

## 4. LSTM input composition

**What enters the recurrent input, exactly** (factored_lstm.py:820–826, forward; :1357–1363 evaluate_actions;
both read first-hand — HIGH):

```
state_with_bp = cat([ normalized_obs , blueprint_embeddings_flat ], dim=-1)
features      = feature_net(state_with_bp)
lstm_out, h   = lstm(features, hidden)
```

- `normalized_obs`: the 120/129-dim EMA-normalized observation (§0).
- `blueprint_embeddings_flat`: `num_slots × 4` learned embedding dims (factored_lstm.py:821–822,
  `DEFAULT_BLUEPRINT_EMBED_DIM=4`, leyline:668), looked up from `blueprint_indices`. **This is the only thing
  concatenated beyond the obs.** Total LSTM feed = `state_dim + 12` for 3 slots.
- **NOT in the recurrent input:** action masks (applied to *logits* after the heads via `masked_fill`,
  factored_lstm.py:849–864 — the committed slot's masking is invisible to the network state); rewards (never an
  input, confirmed source-proof); the sampled slot/blueprint of the previous step (only the **op** is echoed,
  base dims 16–22, one step, global-per-env, overwritten by the next non-WAIT op).

**Enrichment options and their engineering cost:**

| option | mechanism | cost tier | notes |
|--------|-----------|-----------|-------|
| **Mask-as-input** (feed the op/slot masks into the state, not just logits) | concat mask bits into `state` before `feature_net` | **Tier B (obs-dim add)** | closes the "committed slot's masking is invisible" gap the source-proof named (Q2); adds `NUM_OPS + num_slots` dims; re-warm required; masks are already computed each step (lstm_bundle.py:113–121) so no new instrumentation |
| **Full action-history echo** (feed back the sampled *slot/blueprint*, not only op) | extend the base action-feedback block | **Tier B** | today only op is echoed for one step; a slot echo would let the LSTM attribute the previous action to a slot; +`num_slots` (+ blueprint) dims; re-warm |
| **Action-history stack** (last-k actions) | ring buffer of one-hots into state | **Tier B→C** | k× the echo cost; risks teaching recency shortcuts; the LSTM hidden state is *supposed* to carry this — prefer fixing hidden-state credit (out of scope) before widening input |
| **Reward-as-input** | concat previous normalized reward | **Tier B** | classic RL²; but this is a *value* signal and folding it into state risks the policy shortcutting the critic — drl-audit call, flagged not recommended from the plumbing side |

The blueprint-embedding path is the template for any *categorical* enrichment (embed, don't one-hot) and the
per-slot obs block is the template for any *continuous per-slot* enrichment. Both are already wired; new dims
are Tier B (below).

---

## 5. Cost-tier model for additions

Three tiers, defined by what they force. Costs are engineering-effort + run-cost, **HIGH** confidence on the
mechanism of each (they follow directly from §3–§4).

| Tier | What it is | Forces re-warm? | Forces new instrumentation? | Breaks flagged-OFF byte-identity? | Example candidates |
|------|-----------|-----------------|-----------------------------|-----------------------------------|--------------------|
| **A — telemetry-only** | signal is emitted to Karn/telemetry but **not** concatenated into obs | No | No (uses existing per-epoch computes) | No | any diagnostic the drl audit wants to *watch* before feeding; the "measure-first" default |
| **B — obs-dim add (signal already computed)** | new base/per-slot dim sourced from a field **already on** `signal`/`env_state`/`report` | **Yes** (state_dim↑ → first Linear + normalizer reset) | No | Only if the dim's *existence* is arm-gated (don't — hardwire a constant instead) | F2 pending bit; split HOLDING/FOSSILIZED count (§P2); mask-as-input; slot-echo; any §1 "already reachable" signal |
| **C — new instrumentation** | signal is **not** computed per-epoch today; needs a new compute wired into the trainer loop first, *then* Tier-B plumbing | **Yes** | **Yes** (new compute + its own tests + Karn schema) | No | per-layer gradient spectra; per-site activation stats; a per-slot LOO *matrix* (if not already computed — §1 sweep decides) |

**The `to_leyline` gate is a hidden half-tier inside Tier B (this is what bit §P0).** A per-slot signal that
lives on the *kasmina* `SeedMetrics` needs THREE edits, not one: (1) a field on `LeylineSeedMetrics`
(reports.py), (2) a copy line in `to_leyline` (slot.py:281–301), (3) the encoder write in `features.py`. Skipping
(2) produces a silently-constant obs dim with no error — exactly the `contribution_velocity` defect. Any
Tier-B addition sourced from `report.metrics` must verify all three, and should assert non-constancy at runtime
(`obs_normalizer.var[dim] > 0`) as the acceptance check. Signals already on `env_state`/`signal.metrics`
(committed_val_acc, plateau, train/val) skip the `to_leyline` gate — genuinely one-write Tier B.

Decision rule for the drl audit's likely candidates:
- **Per-slot host signals already on `report`/`env_state`** → **Tier B.** Cheap: one-to-three writes + two leyline
  constants + a re-warm (mind the `to_leyline` gate above). This is the bulk of §1's "exists-but-unplumbed" list.
- **Loss-curve / trend features** → **Tier A or B.** If the trajectory is already tracked (obs already carries
  `loss_history[5]`, `acc_history[5]`, `counterfactual_fresh`), a derived plateau/slope scalar is **Tier B**
  (compute in `batch_obs_to_features` from data already passed in — near-free CPU). If it needs a *new* tracker
  spanning rollouts, **Tier C**.
- **Quote/contribution-trend features** → **Tier B** if derivable from `ContributionState`/`epochs_since_cf`
  already on `env_state`; **Tier C** if it needs a new per-slot history buffer.

**Re-warm is the dominant recurring cost** — *any* Tier-B/C change resets `state_dim`, so batch the additions.
The F2 bit is the first schema change of this arc (source-proof §M); if a future V5 is coming anyway, this
audit's recommendation is to **land the drl audit's Tier-B candidates in the same schema bump** rather than
paying the re-warm/retrain tax once per dim. One re-warm for N dims, not N re-warms.

---

## Confidence ledger (SME protocol)

| finding | confidence | how to raise it |
|---------|-----------|-----------------|
| Obs layout §0 table | HIGH | read features.py:837–992 end to end |
| **§P0 `contribution_velocity` dead dim** | **HIGH** | verified the full chain: compute (vt:1385) → to_leyline omits it (slot.py:281–301, whole-file grep) → obs reads leyline snapshot (features.py:913) |
| §1 `committed_val_acc` top-signal | HIGH | env_state:73 + set vt:1422; obs uses only all-seeds val_acc |
| §1 inventory items 1–9 | HIGH | two independent sweeps converged on matching file:line; each spot-verified |
| No train/act skew | HIGH | traced acting loop 2005→2042→2082→buffer.add |
| P1 normalizer×sentinel mechanism | HIGH | read normalization.py in full |
| P1 impact magnitude (which dims spike) | MED→LOW | dump per-dim `obs_normalizer.var`; count updates-before-first-activation |
| P2 HOLDING/FOSSILIZED conflation | HIGH | features.py:825 |
| P3 collapsed-head input echoes | MED code / LOW empirics | per-dim var dump; the collapse itself is r9-measured |
| F2 off-by-one landmine | HIGH | slot_offset formula features.py:874 |
| `to_leyline` reachability gate | HIGH | read slot.py:281–301 field list |
| §P0 fourth link (`report.metrics = to_leyline()`) | HIGH | slot.py:570 read directly |
| §1 inventory items 10–11 (grad-drift EMA, raw dual-grad norms) | MED | anomaly/trainer-side; existence read, exact plumbing cost not traced end to end |

**Consolidated information gaps (SME honesty — do not over-trust these three):**
(a) §1 items 10–11 (grad-drift EMA, raw dual-grad norms) were read for *existence*; their exact Tier-1 plumbing
cost was not traced end to end. (b) §1 #9 host `best_val_*` "no consumer found" is a **grep-negative**, not proof
of deadness. (c) §P1 impact magnitude (which dims actually spike/attenuate under the EMA normalizer) is an
analytic bound, not a measurement — the `obs_normalizer.var` dump named in §P1/§P3 is the one runtime check that
converts these to HIGH.
