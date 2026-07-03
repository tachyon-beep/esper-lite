# PIN-E Placebo Noise-Floor Harness — Build & Run Plan

```yaml
# Plan Metadata
id: pin-e-placebo-harness
title: PIN-E placebo noise-floor harness (near-inert small-real seed run → tau calibration)
type: ready
created: 2026-07-03
updated: 2026-07-03
owner: claude-product-owner (agent), owner-signed via PDR-0014

urgency: high            # first leg of the Committed-Shapley enablement gate (esper-lite-f22a1d48a7)
value: >
  Measures the LOO/Shapley estimator's noise floor at zero contribution via a near-inert
  small-real placebo seed, yielding (1) the GATE-0 noise-floor report line and (2) the tau
  (shapley_synergy_noise_floor) recommendation the enablement gate is blocked on.

complexity: M            # new blueprint + schedule constant + 2 small plumbing kwargs + driver + analysis
risk: medium
risk_notes: >
  (1) Non-degeneracy: if the placebo's measured LOO is identically zero (quantized to the
  0-grid), the floor is vacuous — PDR-0014 reversal trigger fires and the method reverts to
  option (a) K-resampled val minibatches. (2) epsilon-dependence: the floor scales with the
  placebo's init perturbation; must be reported with an epsilon-sensitivity check, not as an
  absolute. (3) G2 stage-gate passage with seed_lr=0 is analytically argued but must be
  proven by an integration test before any GPU time is spent.

depends_on: []           # committed-shapley build (PDR-0013) already landed
blocks:
  - committed-shapley-enablement (esper-lite-f22a1d48a7, criterion 1)

status_notes: plan drafted from 3-scout evidence sweep; awaiting drl-expert + pytorch-expert review
percent_complete: 0

reviewed_by:
  - reviewer: drl-expert
    verdict: APPROVED_WITH_CHANGES (2026-07-03 — durable copy
      docs/analysis/2026-07-03-pin-e-placebo-drl-review.md. All 5 REQUIRED folded: R1 k=3
      co-resident placebos + tau from signed terminal (φ−c_paid) P99 w/ block-bootstrap (k=1 ⇒
      φ≡c_paid identity — single-seed run cannot calibrate the deadband); R2 D1/D2 deliverable
      split; R3 downward epsilon ladder + plateau test; R4 per-dwell-epoch gradient-health margin
      recording; R5 conservative-provisional lower-bound framing. Rec2 structural argument, Rec3
      masking-equivalence check, Rec4 per-epoch lr assert folded; Rec1 placebo-among-reals arm
      deferred (needs partial masking). Its Option-B endorsement is OVERRIDDEN by pytorch-expert's
      verified obs-encoder crash chain; its byte-comparability concern is folded into §2.3's cost
      acknowledgment — no cross-version paired comparisons.)
  - reviewer: pytorch-expert
    verdict: APPROVED_WITH_CHANGES (2026-07-03 — all 4 REQUIRED changes folded: §2.3 Option A
      mandatory w/ context-dependent mask availability; WI-3 complete call-site list; WI-4
      real-run-path lr assert + no-refetch guard + dataclasses.replace; WI-5 exact-epoch asserts.
      RECOMMENDED folded: 3×3 depthwise bias=False; §2.6 solo-not-all_off + INSTANT-α notes;
      single-tensor health note.)
```

Tracker: **esper-lite-94869250f1** (in_progress, blocks esper-lite-f22a1d48a7).
Authority: owner sign-off 2026-07-03 (PDR-0014). Hard rule: `shapley_synergy_scale` stays **0.0**
throughout — this harness measures with the term OFF; enabling is a separate owner decision.

---

## 1. Objective

Per `docs/analysis/2026-06-25-phase0-objective-and-instrumentation.md` §2 (PIN E): run
**near-inert SMALL-REAL** placebo seeds on the output path and measure the empirical distribution
of their credited counterfactual contribution. **Two deliverables, two estimators (drl R2 — do not
conflate):**
- **D1 (GATE-0 line):** per-stage mean (bias), std (noise floor), CoV of the per-step LOO
  `seed_contribution` (the standalone marginal).
- **D2 (tau):** the deadband for `gap = max(0, φ − c_paid − tau)` — calibrated on the **signed
  upper-tail quantile of `(φ − c_paid)` for a null player at the operating k (k=3)**, NOT on the
  single-seed LOO spread. At k=1, φ(s) = v({s})−v(∅) = c_paid(s) **identically** (Shapley
  definition + the c_paid amendment), so `φ − c_paid ≡ 0` and the term structurally never fires —
  a single-seed run cannot calibrate the deadband (drl R1).

The `noop` blueprint is the reviewer-rejected degenerate case (deterministic-zero delta ⇒
bit-identical leave-out logits ⇒ vacuous floor). Offline proxies from live runs are rejected
(selection on low contribution is circular); forcing known-null seeds is not selection.

## 2. Design (scout-verified)

### 2.1 Operating point: k=3 co-resident placebos, held at HOLDING (never fossilized)
Fossilized slots are excluded from the ablation family at `drip_fraction=0`
(`vectorized_trainer.py:1049-1053`); the only machinery measuring marginals over fossilized sets is
the committed-Shapley block, gated on `shapley_synergy_scale > 0` (`:1153-1156`) — off-limits.
Therefore the placebos are held at **HOLDING, α=1** (full blend amplitude — the same alpha-override
kernel the committed-Shapley block uses) for the remainder of every episode.

**k=3, not k=1 (drl R1):** three near-inert placebos co-resident (all three slots), because tau
gates the synergy contrast `φ − c_paid`, which is identically zero at k=1 and whose noise is a
zero-sum contrast of coalition marginals — not a fixed multiple of a single marginal's spread
(cross-config correlations through the shared fused pass make any correction factor unmeasurable
at k=1). k=3 matches `max_seeds=3` and the exact-factorial ceiling (k≤3).

**The full 2³ factorial is already logged at scale=0:** with 3 active non-fossilized seeds,
`_build_fused_val_configs` builds `main` (v(C)), 3 `solo` (leave-one-out), 3 `solo_on` (v({s})),
`all_off` (v(∅) — built for 2≤n≤4, `vectorized_trainer.py:1082-1087`), and 3 `pair` configs
(3≤n≤4, `:1090-1103`) = all 8 coalitions; `on_counterfactual_matrix` (`:1626`) logs them every
epoch. **φ(s) and c_paid(s)=v({s})−v(∅) are assembled OFFLINE by the analysis script** — the
committed-Shapley reward block never runs. The declared schedule fully masks the controller, so
exactly these 3 seeds exist — nothing else germinates.

### 2.2 Inertness mechanism: tiny-random single-layer delta + seed_lr=0
Constraints found by the scouts:
- Hard freeze (`requires_grad=False`) ⇒ G2 fails `gradient_health_not_measured`
  (`slot.py:821-829`) ⇒ seed never leaves TRAINING. **Dead.**
- Zero-init + lr=0 ⇒ upstream grads exactly 0 < 1e-7 vanishing threshold
  (`gradient_collector.py:366-373`) ⇒ health < 0.7 ⇒ G2 blocks. **Dead.**
- GroupNorm downstream of a tiny final conv renormalizes the output to O(1) ⇒ defeats inertness
  (all existing GN-fronted residual blueprints unusable as placebos).

**Chosen mechanism:**
- **New blueprint `placebo` (cnn):** residual `y = x + conv(x)` where `conv` is a single 3×3
  depthwise (or 1×1) convolution — the ONLY layer, weight init `normal_(std=1e-3)`, bias zeros,
  **no normalizer or activation after it**. Single-layer ⇒ no upstream tensors to vanish (the one
  weight tensor's grads are O(loss-grad × activations), health ≈ 1.0 ⇒ G2 clears); delta scale
  ≈ 1e-3·‖x‖ ⇒ near-inert but NOT bit-identical (non-degenerate). Param count ≈ dim·9+dim
  (depthwise) — small live rent.
- **`seed_lr_override: float | None = None` kwarg on `train_ppo_vectorized`**, applied to the
  task spec after `get_task_spec(task)` (`vectorized.py:892`); driver passes `0.0`. With SGD
  lr=0 the delta never moves for the entire run; `requires_grad` stays True so G2's gradient
  measurement is satisfied. `seed_lr` is task-level (`runtime/tasks.py:45`) and the placebo is
  the run's only seed, so the override cannot leak onto other seeds by construction.
- Style: `SIGMOID_ADD` (as existing schedules) ⇒ ADD blend `lerp(host, seed, α) = host + α·δ(x)`
  (`blend_ops.py:40-55`); `zero_init_final_layer` fires only for MULTIPLY (`slot.py:1431-1434`)
  so the blueprint's own init is authoritative.

### 2.3 Blueprint delivery — RESOLVED: Option A (enum member) with context-dependent availability
**(Fork adjudicated by pytorch-expert review 2026-07-03: Option B is IMPOSSIBLE, not merely messy.)**
The observation encoder — not the registry — is the binding constraint: `SeedState.to_report()`
raises `ValueError` for any `blueprint_id` not in the enum-derived `BLUEPRINT_ID_TO_INDEX`
(`slot.py:548-551`; import-time assert `len(BLUEPRINT_ID_TO_INDEX) == len(BlueprintAction)`,
`factored_actions.py:264`), and observations are built every step via `get_slot_reports()`
(`vectorized_trainer.py:2166`, `action_execution.py:1561`). A registry-string placebo crashes at
the first observation. Defanging the guard would violate the fail-loud rule and corrupt the obs
feature. Enum membership is a hard precondition for any germinable blueprint. Option A also keeps
the governor preflight honest (`preflight_blueprint_id = BLUEPRINT_IDS[blueprint_action]`,
`action_execution.py:774` — Option B would have fed it a decoy string).

**Adopted shape:**
- Add `BlueprintAction.PLACEBO` + `to_blueprint_id` mapping + registry module (WI-1). It flows
  into `BLUEPRINT_IDS`/`BLUEPRINT_ID_TO_INDEX` automatically.
- **Keep PLACEBO OUT of `CNN_BLUEPRINTS`** (`action_masks.py:240-249`, `factored_actions.py:457`)
  so no normal run can ever germinate it — BUT `_force_head_choice` raises if the forced index is
  masked off, so the placebo run needs **context-dependent availability**: a flag keyed off the
  active declared schedule that unions `{PLACEBO}` into the blueprint availability mask for that
  run only. Action masking is a first-class mechanism here; this is not a compat shim.
- **Acknowledged permanent costs (owner-visible):** blueprint head cardinality 13→14
  (`NUM_BLUEPRINTS`, `factored_actions.py:392`) changes the policy net's blueprint-head dim and
  `nn.Embedding(num_blueprints+1)` (`factored_lstm.py:158`) ⇒ pre-change policy checkpoints stop
  loading (shape mismatch) and post-change runs are not bit-identical to pre-change runs (RNG
  stream shift in the masked draw). Acceptable under No-Legacy; fresh-seed experiments unaffected;
  recorded here so no future A/B silently compares across the boundary.
- No new blueprint-delivery plumbing needed — the schedule constant drives the existing
  `apply_proof_baseline_action_controls` forcing path.

### 2.4 New declared schedule + dispatch generalization
`FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1` — three staggered placebo lifecycles (the schedule pins ONE
action per epoch): germinate s0@1/s1@2/s2@3 → advance-to-TRAINING s0@4/s1@5/s2@6 → after the
10-epoch G2 TRAINING dwell, advance-to-BLENDING s0@14/s1@15/s2@16 → advance-to-HOLDING
s0@17/s1@18/s2@19 (G3 permissive skips the blend dwell; INSTANT α ⇒ α=1 from BLENDING entry) →
**WAIT forever** (no fossilize step). All three at HOLDING α=1 from epoch 19 to max_epochs. Exact
epochs are finalized against the G2/G3 gate math in WI-5's integration test before any GPU run.
Skeleton follows `STATIC_FINAL_SOURCE_TOPOLOGY_STEPS` (`leyline/proof_baselines.py:133-154`) minus
fossilize, times three slots.

The dispatch is currently hardcoded to one schedule (`fixed_schedule_action_for_epoch`
`:187-198`; runner validation `vectorized_trainer.py:392-403`). Generalize: a
`DECLARED_SCHEDULES: dict[schedule_id, DeclaredSchedule(steps, version, expected_hash,
action_count)]` registry in leyline with `declared_schedule_action_for_epoch(schedule_id, epoch)`;
rewrite both call sites; **delete** the hardcoded function (No-Legacy — update all call sites in
the same commit). Import-time hash guard retained per schedule.

### 2.5 Run shape
- Config: new `configs/config-pin-e-placebo.json` derived from the causal config
  (`n_envs=12, max_epochs=150, compile off`), `n_episodes ≈ 30` (PPO rounds) ⇒ ~360 episodes/seed.
  **Primary population for tau = ONE sample per episode: the single true terminal epoch** (the
  term fires once at `epoch==max_epochs` on a converged host) — ~360 independent (φ−c_paid)
  triples/seed (3 null players/episode, within-episode correlated ⇒ block-bootstrap by episode,
  drl R1b/d). The ~130-epoch HOLDING window feeds D1 and secondary/detrended diagnostics only
  (heavily autocorrelated; effective N/episode ≈ 1 at a converged terminal). Sufficiency is
  bound to the block-bootstrap CI width vs the tau estimate, not raw sample count (drl Q6).
  Runtime estimate ~1h/run (scaling from 6.6h @ 200 rounds, run sheet `:27`) — verify by smoke
  before committing GPU time.
- Seeds 41–43 (one per GPU round; both RTX 4060 Ti currently idle). Driver: pilot-script pattern
  (`scripts/causal_contribution_r1_pilot.py:134-186`) — per-seed `train_ppo_vectorized(...)` with
  `telemetry_dir`, `group_id`, preflight asserting `use_telemetry=True` (G2 gradient-health is
  telemetry-coupled — esper-lite-4fe98055f7), `compile_mode="off"`, `seed_lr_override=0.0`,
  `shapley_synergy_scale==0.0`, declared-schedule provenance kwargs.

### 2.6 Measurement & analysis
Read `events.jsonl` raw (template: `scripts/causal_contribution_j_analyze.py`). Sources:
- Per-epoch LOO: `ANALYTICS_SNAPSHOT kind='last_action'` → `seed_contribution` (computed for
  `target_slot` on EVERY op incl. WAIT, `action_execution.py:831-835`) + `seed_stage`, `val_acc`.
  Cross-check against `COUNTERFACTUAL_MATRIX_COMPUTED` configs — for a SINGLE active seed there is
  NO `all_off` config (only built for 2≤n≤4, `vectorized_trainer.py:1082`); v(∅) is the `solo`
  config (slot forced α=0), which for one seed IS the empty coalition.
- **Alpha note:** the schedule uses `alpha_speed=INSTANT`, so BLENDING already sits at α=1 — the
  BLENDING/HOLDING per-stage split is nominal (both at full amplitude). That is intentional: φ is
  measured at full amplitude, so every on-path sample is collected in the φ-relevant regime.
- Per-episode integrals: `ANALYTICS_SNAPSHOT kind='seed_residency'`
  (`cf_weighted_integral`, `n_on_path_steps`, `n_none_steps`, `j_per_param`).

Outputs (`docs/analysis/2026-07-03-pin-e-placebo-noise-floor.md`):
1. **D1 — GATE-0 line:** per-stage (BLENDING, HOLDING) mean/std/CoV of per-step LOO
   `seed_contribution`. (BLENDING at INSTANT α=1 is nominal; if any α<1 window exists its floor
   will be quantization-degenerate — report as such, derive nothing from it.)
2. **D2 — terminal `(φ − c_paid)` distribution** for the 3 null players, one sample per episode
   at `epoch==max_epochs`, assembled offline from the logged 2³ factorial: report mean (bias) and
   **signed** quantiles P50/P90/P95/P99/max (only the positive tail leaks credit through
   `max(0,·)` — |·| conflates tails, drl R1a). **Quantization caveat:** the fused val pass scores
   the full shared test set every epoch (`vectorized_trainer.py:1355-1381`), so values sit on a
   100/|testset| pp grid — quantiles, not Gaussian fits.
3. **tau recommendation = P99 of signed terminal `(φ − c_paid)`** with a stated per-seed
   false-positive budget, **block-bootstrap CI (block = episode)**, and the run-provenance block.
   **Marked CONSERVATIVE-PROVISIONAL (drl R5):** the placebo floor is a LOWER BOUND on the
   operating (null-among-reals) noise — homogeneous-null coalition + near-identity-trained host +
   single-fire-at-converged-terminal all bias it downward vs enablement conditions (real
   high-variance co-residents on a co-adapted host). Recommend tau with an explicit safety
   margin; finalized by the first ON calibration run (gate criterion 6). Too-small tau
   over-credits freeloaders — the dangerous direction.
4. **Non-degeneracy verdict** (required for the report to count, phase-0 §2): (i) fraction of
   nonzero samples materially > 0; (ii) spread spans ≥ several grid steps; (iii) recorded delta
   scale; (iv) the STRUCTURAL same-noise-sources argument (full-testset eval ⇒ only host-drift +
   kernel-order + quantization noise, shared with real seeds by construction — drl Rec2 minimum).
   Degenerate ⇒ PDR-0014 reversal trigger (revert to option (a) K-resampled minibatches).
5. **Epsilon ladder DOWNWARD (drl R3):** arms at `std ∈ {1e-4, 1e-3}` (optional 5e-5 probe for
   the degeneracy wall). Acceptance: tau flat within CI across small-δ arms ⇒ genuine estimator
   floor; monotone-increasing in δ ⇒ signal-contaminated (δ-anchored) ⇒ do not bank.
6. **Masking-equivalence check (drl Rec3):** unit-level numeric check that alpha-override
   (HOLDING) and tensor-override (fossilized) blend paths agree for a null player, since offline
   φ masks HOLDING seeds while the live term masks fossils.

## 3. Evidence map (scout citations)
| Fact | Where |
|---|---|
| seed_contribution = val_acc − baseline_accs[target_slot], every op | action_execution.py:831-835 |
| Fossilized excluded from ablation at drip=0 | vectorized_trainer.py:1049-1053 |
| committed-Shapley block gated scale>0 | vectorized_trainer.py:1153-1156 |
| Fused val pass = full test set, all configs fused, every epoch | vectorized_trainer.py:1355-1381, 2720-2734 |
| Schedule constants + hash guard + WAIT default | leyline/proof_baselines.py:65-198 |
| Blueprint forced via mask head from schedule constant | vectorized_trainer.py:434-490, 466-471 |
| Seed optimizers: per-slot SGD(lr=task_spec.seed_lr), unfiltered params | batch_ops.py:144-285 |
| No per-seed lr/freeze knob exists | scout sweep (helpers.py:87-104; slot.py:2225-2252 blend-out only) |
| G2 permissive: ≥10 epochs dwell + gradient measured + health ≥0.7 | slot.py:805-860; leyline:691,696 |
| G5 permissive: no contribution threshold | slot.py:962-1009 |
| Vanishing threshold 1e-7 / health formula | gradient_collector.py:366-373, :29 |
| ADD blend lerp; zero_init only fires for MULTIPLY | blend_ops.py:40-55; slot.py:1431-1434 |
| TaskSpec.seed_lr resolved internally by name | vectorized.py:890-892; runtime/tasks.py:45 |
| seed_residency view columns | karn/mcp/views.py:547-570 |
| Driver/preflight pattern | scripts/causal_contribution_r1_pilot.py:134-186 |

## 4. Work items (TDD: failing test first for every new behavior)
- **WI-1 Placebo blueprint** (`kasmina/blueprints/cnn.py` + registry): **3×3 depthwise conv,
  `bias=False`**, residual `y = x + conv(x)`, `normal_(std=1e-3)` weight init, no norm/activation
  after the conv; factory takes `dim` only and raises on unexpected kwargs (repo convention,
  `cnn.py:24-25`). Tests — residual form (α=1 output = host + δ, δ/‖x‖ ≈ O(1e-3)); grads flow
  (the single weight tensor's grad norm > 1e-7 vanishing threshold on a synthetic batch — note:
  bias=False means ONE grad tensor, so vanishing_ratio is 0-or-1 and health is 1.0-or-0.5;
  analytically fine, tested anyway); param count = dim·9.
- **WI-2 Blueprint delivery (Option A per §2.3):** `BlueprintAction.PLACEBO` enum member +
  `to_blueprint_id` mapping; PLACEBO excluded from `CNN_BLUEPRINTS`; declared-schedule-keyed mask
  availability union. Tests — forced germination legal under the placebo schedule; normal-run
  masks NEVER offer PLACEBO; `to_report()` observation encoding works for an active placebo;
  every `NUM_BLUEPRINTS`-dependent shape/test updated in the same commit (13→14).
- **WI-3 Declared-schedule registry** (leyline): the 3-slot staggered placebo schedule
  `FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1` (§2.4) + expected hash + version +
  action count; `DECLARED_SCHEDULES` dispatch subsuming the WAIT-default; hardcoded
  `fixed_schedule_action_for_epoch` deleted, ALL call sites updated in the same commit; existing
  schedules (`FIXED_SCHEDULE_GERMINATE_R0C0_*`, `STATIC_FINAL_SOURCE_TOPOLOGY_*`) must survive as
  registry entries. **Complete change list (pytorch-review enumerated):** source —
  `leyline/proof_baselines.py` (defs + `__all__:253-278`), `vectorized_trainer.py:31-40,392-410`,
  `simic/training/proof_baselines.py:14-17,107-111`; tests/fixtures —
  `tests/simic/training/test_proof_baselines.py:27-31,167-188,378-513` (incl. the hash-drift
  guard; new schedule gets its own EXPECTED_HASH + drift test),
  `tests/scripts/test_proof_packet.py:14-17,74-83,521,940` (packet asserts schedule_id strings),
  `tests/simic/test_vectorized_correctness.py:11-14,279-309`,
  `tests/integration/test_gradient_measurement_telemetry_independent.py:25-28,96-100`,
  `tests/nissa/test_wandb_backend.py:19-22,697-722`, `tests/karn/mcp/test_views.py:10-13,166-211`.
- **WI-4 seed_lr_override** (`train_ppo_vectorized`): validation ≥0; applied post-`get_task_spec`
  via `dataclasses.replace` (TaskSpec is slots, not frozen); **the threading hazard is
  `helpers.py:520-522` re-fetching `get_task_spec` when `task_spec is None` — a silent lr>0
  fallback would corrupt tau with no crash.** Tests — assert `lr == 0.0` on the REAL run-path
  optimizer (`env_state.seed_optimizers[slot_id]` after a real `batch_ops` step), seed params
  bit-identical after N training batches, and no code path re-fetches the task spec after the
  override.
- **WI-5 G2/G3-passage integration test** (the go/no-go before GPU; CPU-runnable — all CUDA calls
  are stream-guarded; template: `tests/integration/test_gradient_measurement_telemetry_independent.py`):
  tiny run (max_epochs ≈ 22, 1 env) with the 3-slot placebo schedule + lr=0 + telemetry ON.
  **Exact-epoch asserts per slot** (per §2.4 stagger; finalized against the gate math here):
  each slot == TRAINING through its dwell, == BLENDING AT its scheduled epoch (G2: 10-epoch
  TRAINING dwell, gradient_measured True, health ≥ 0.7), α == 1.0 at BLENDING entry (INSTANT),
  == HOLDING AT its scheduled epoch, NEVER FOSSILIZED; `seed_contribution` non-None from the
  first α>0 epoch; all_off + pair configs present in the counterfactual matrix once ≥2 seeds
  active. **drl R4:** record (not just assert) each placebo's single-tensor grad L2 and computed
  gradient_health at every dwell epoch, asserting a MARGIN above the 1e-7 vanishing threshold —
  single-tensor health is a binary step function (1.0-or-0.5), so a transient sub-threshold batch
  would block G2 mid-dwell; prove the margin before GPU.
- **WI-6 Driver + config** (`scripts/pin_e_placebo_run.py`, `configs/config-pin-e-placebo.json`):
  preflight (telemetry ON, compile off, scale==0.0, lr override set, schedule provenance) +
  **per-epoch `lr == 0.0` assertion on the live seed optimizers** (drl Rec4 — catches any future
  scheduler wrapping silently).
- **WI-7 Analysis script** (`scripts/pin_e_placebo_analyze.py`): offline Shapley assembly —
  reconstruct all 8 coalition accuracies per epoch from `COUNTERFACTUAL_MATRIX_COMPUTED`
  (main/solo/solo_on/all_off/pair), compute φ(s) via exact factorial weights and
  c_paid(s)=v({s})−v(∅), emit the terminal signed (φ−c_paid) population + D1 per-stage stats +
  block-bootstrap CI. Unit tests on synthetic events.jsonl including a HAND-COMPUTED 2³ example
  (mirroring the committed-shapley test discipline); refuses runs whose preflight provenance is
  missing; cross-checks φ+c_paid identities (Σφ = v(C)−v(∅); k=1-subset identity on any single
  slot).
- **WI-8 Runs + memo**: smoke (1 seed, n_episodes=2) → timing check → full 3-seed run at
  `std=1e-3` + epsilon arms `{1e-4}` (optional 5e-5 probe) per drl R3 → analysis →
  `docs/analysis/2026-07-03-pin-e-placebo-noise-floor.md` + tau recommendation (P99 signed,
  CI, safety margin, CONSERVATIVE-PROVISIONAL label) posted to esper-lite-f22a1d48a7. GPU
  launches are within grant; **enablement is not**. Optional stretch (drl Rec1/Rec2, run only if
  owner wants the stronger evidence): a real-small-seed control arm (same harness, real blueprint)
  for the same-noise-sources clause; the placebo-among-free-reals arm is DEFERRED (needs partial
  masking the declared-schedule machinery doesn't support) and noted in the memo as the follow-up
  instrument if the first ON calibration shows the floor was underestimated.

## 5. Open questions for reviewers
1. ~~Fork §2.3~~ **RESOLVED (pytorch-expert): Option A mandatory** — Option B crashes at
   observation encoding (`slot.py:548-551`); context-dependent mask availability adopted (§2.3).
2. ~~tau statistic~~ **RESOLVED (drl R1):** signed P99 of terminal `(φ−c_paid)` at k=3,
   block-bootstrap CI — not P95(|LOO|), not k·std, not max. No bolt-on correction factor is
   justifiable; measure the contrast directly.
3. ~~epsilon design~~ **RESOLVED (drl R3):** downward ladder {1e-4, 1e-3} (+optional 5e-5) with
   the δ-insensitivity plateau acceptance test; 1e-2 probes the wrong direction.
4. ~~lr=0 vs tiny~~ **RESOLVED (drl Rec4 + pytorch (c)):** exactly-zero, and preferable (fixed-δ,
   no learning trend); no scheduler exists; SGD lr=0 is a true no-op incl. momentum/weight-decay;
   per-epoch lr assert added.
5. ~~host-regime confound~~ **RESOLVED (drl R5, named lower-bound):** the near-identity-trained
   host + homogeneous-null coalition + single-terminal-fire gaps all bias the floor DOWN — tau is
   a conservative-provisional lower bound with safety margin, finalized at the first ON
   calibration run.
6. ~~run shape~~ **RESOLVED (drl Q6):** sufficiency bound to block-bootstrap CI width vs tau
   (effective N/episode ≈ 1 at converged terminal), primary population = single terminal epoch
   per episode.

## 6. Acceptance criteria
1. All new unit/integration tests green; existing suites unaffected (`shapley_synergy_scale`
   remains 0.0 everywhere; no reward-path change); every `NUM_BLUEPRINTS`-coupled surface updated
   in the same commit as the enum change.
2. WI-5 G2/G3-passage test proves the lifecycle claim — including the per-dwell-epoch
   gradient-health margin recording (drl R4) — before any GPU run.
3. Full runs complete with preflight provenance recorded; analysis reproduces from raw events;
   φ/c_paid identity cross-checks pass (Σφ = v(C)−v(∅)).
4. Report contains BOTH deliverables kept distinct (drl R2): D1 per-stage GATE-0 line, and D2 the
   terminal signed (φ−c_paid) distribution with tau = P99 + block-bootstrap CI + false-positive
   budget + the CONSERVATIVE-PROVISIONAL lower-bound framing (drl R5) + non-degeneracy verdict
   incl. the structural same-noise-sources argument + the epsilon-ladder plateau verdict (drl R3)
   + the masking-equivalence check result (drl Rec3).
5. esper-lite-94869250f1 closed only when the tau memo is posted to the gate issue.
