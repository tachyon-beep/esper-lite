# Committed-Shapley Synergy Top-Up — Default-OFF Build Plan

```yaml
# Plan Metadata
id: committed-shapley-topup-build
title: Committed-Shapley synergy top-up — default-OFF build (retro-write delivery)
type: in-progress
created: 2026-07-02
updated: 2026-07-02
owner: Claude (product-owner session; owner build sign-off received 2026-07-02)

urgency: high
value: >
  Reward-side credit for LOO-undervalued efficiency-enabling stems (the (b) result, PDR-0009),
  built default-OFF behind shapley_synergy_scale=0.0. Unblocks the enablement gate (A/B on
  committed-J / corr(reward,J)) without changing any shipped behavior.

complexity: L
risk: medium
risk_notes: >
  (1) The no-op-at-0 guarantee is the stability story — any leak of the new path into the
  scale=0 regime is a silent behavior change on every run. (2) The retro-write mutates the
  rollout buffer between collection and GAE; an indexing error corrupts credit assignment
  silently. (3) HRA/CF-stream interaction is undefined by the design; must be excluded
  explicitly, not accidentally. All three are covered by mandated tests before the mechanism.

depends_on: []            # all evidence gates passed (PDR-0011, PDR-0012)
soft_depends: []
blocks:
  - reward-credit-enablement-gate   # tau/PIN-E placebo, off-switch-J hard gate, entrenchment monitor

status_notes: >
  Reviewed 2026-07-02 (both approved-with-changes; findings folded into WI text below —
  the reconciled WI text is authoritative). Implementation started same day.
percent_complete: 5

reviewed_by:
  - reviewer: pytorch-expert
    date: 2026-07-02
    verdict: approved-with-changes
    notes: >
      HIGH F1: WI-4 std_floor was inexpressible via divide_by_std (no floor param/std accessor;
      count<2 raw passthrough) — F2 bound would silently never apply; fix = explicit divisor API
      + required test. Also: fail-loud else on the _kind unpack chain + direct indexing (no .get);
      assert drip_fraction==0 and non-GATE committed slots when scale>0; name the buffer-derived
      rollback-exclusion mechanism; kernel bit-identity valid only for finite seed features
      (masked path = blend branch, not STE). Confirmed: GAE-after-seam ordering, no autograd/
      compile hazard on the reward-tensor write, natural-alpha choice, blowup bounded (k≤3,
      terminal-only). cuDNN batch-shape perturbation of terminal val_acc at scale>0 flagged as an
      ENABLEMENT-gate confound (pre-existing class), not a build blocker.
  - reviewer: drl-expert
    date: 2026-07-02
    verdict: approved-with-changes
    notes: >
      Full artifact docs/analysis/2026-07-02-committed-shapley-build-plan-drl-review.md.
      MED F-A: same F2 hole as pytorch F1 + __post_init__ must guard the bound ON when scale>0;
      normalized-space clamp made the PRIMARY bound. MED F-B: WI-10 needs truncated-vs-done,
      done-between-t-and-t_f, t_f==0/last, short-episode cases; WI-12 must assert the credited
      step IS the FOSSILIZE transition. F-C: sibling mutual-exclusion coupling decided (accepted
      confound, enablement gate re-checks dormancy). F-F: WI-8 rename shifts scale=0 telemetry
      KEYS (not training behavior) — needs its own serde test. Confirmed: rollback exclusion set
      == forfeit set (sufficient); cross-episode fossil credit structurally impossible (DORMANT
      assert at reset); GAE linearity; HRA hard-exclusion justified; k=1 ⇒ top_up≡0 by
      construction. All six §6 open questions resolved (see §6).
```

**Authority:** PDR-0012 (owner sign-off 2026-07-02, recorded on esper-lite-254175df90 comment 93).
**Design:** `docs/plans/concepts/2026-07-01-reward-credit-shapley-synergy-design.md` (mechanism), as
amended by GATE 2 (`docs/analysis/2026-07-02-gate2-learnability-result.md`: retro-write MANDATED) and
the reward-function-reviewer (`docs/analysis/2026-07-02-reward-credit-term-review.md`: F1–F8).
**Tracker:** esper-lite-254175df90.

---

## 1. Scope and non-goals

**In scope:** the default-OFF mechanism, its delivery, its telemetry, the five reviewer
build-conditioning changes (F5, F1/F8, F2, F4/F6, GAE unit test), and the naming-collision
resolution. `shapley_synergy_scale` stays `0.0` in every shipped config.

**Non-goals:** enablement (separate later gate: tau from PIN-E placebo, hard off-switch-J
efficiency + fossilize-count gate, entrenchment monitor, ON-run dormancy recheck); the n=10
magnitude run; Karn dashboard views for the new event (raw_events queries suffice; dedicated
view is follow-up); HRA stream routing (excluded by validation, see WI-5).

## 2. The mechanism (final form, reviewer-amended)

At `epoch == max_epochs`, per env, over the k committed (FOSSILIZED) slots C (k ≤ 3, assert):

```
v(S)      = fused-val accuracy (PERCENTAGE POINTS) with coalition S of fossilized slots at
            natural alpha, fossilized slots ∉ S alpha-masked 0.0, ALL non-fossilized active
            slots alpha-masked 0.0   (v(∅) = host only; v(C) ≅ existing `committed` config
            semantics with non-fossilized forced off)
phi(s)    = Σ_{S ⊆ C\{s}} |S|!(k−|S|−1)!/k! · ( v(S∪{s}) − v(S) )     # exact 2^k factorial
c_paid(s) = v({s}) − v(∅)          # F4/F6: terminal same-time standalone baseline
                                    # (REPLACES the design's fossilize-time seed_contribution snapshot)
gap(s)    = max(0, phi(s) − c_paid(s) − tau)          # tau = shapley_synergy_noise_floor
raw(s)    = min(shapley_synergy_cap, shapley_synergy_scale · gap(s))
G         = max(0, v(C) − v(∅))
top_up(s) = raw(s) · min(1, G / Σ raw)   if Σ raw > 0 else 0          # efficiency clamp
```

**Delivery (GATE 2 mandate, reviewer-corrected):** for each credited seed s with FOSSILIZE
decision at buffer step t_f in env e:

```
credit_buf = min( shapley_synergy_normalized_cap,                       # PRIMARY bound (buffer units,
               top_up(s) / max(running_reward_std,                      #   calibration-free)
                               shapley_synergy_std_floor) )             # secondary bound on the divisor
buffer.rewards[e, t_f] += credit_buf                                    # pre-GAE seam
```

NO clip from the normalizer, NO normalizer-stat update. The current `divide_by_std` API cannot
express this (no floor param, no std accessor, raw passthrough at count<2 — pytorch F1 / drl
F-A): the build adds an explicit accessor/parameter on `RewardNormalizer` and handles count<2
explicitly. GAE then runs untouched — its linearity delivers Δadv(t_f) = credit_buf exactly.

**Purity:** credit only seeds that actually reached FOSSILIZED (G5 passed), in envs not
rollback-contaminated this batch. The GATE 2 probe's all-effective-FOSSILIZE targeting was a
probe simplification; the build uses per-seed records (WI-3).

## 3. Evidence map (scout-verified, 2026-07-02)

| Seam | Location | Fact |
|---|---|---|
| v(S) evaluator | `kasmina/host.py:803` `fused_forward(x, alpha_overrides)` | Masks arbitrary slots incl. FOSSILIZED (`leyline/stages.py:84-91`: fossilized is active; `slot.py:2165` override is the lerp amplitude — 0.0 ⇒ host features unchanged) |
| Config assembly | `vectorized_trainer.py:989` `_build_fused_val_configs` | Terminal shapley configs exist TODAY but over NON-fossilized slots (`active_slot_list` excludes FOSSILIZED :1030-1042) and telemetry-only (result discarded; `contribution.py:43` docstring). `committed` config (:1093-1108) ≈ v(C) precedent. **v(∅) not computed today — new config.** |
| Units | `vectorized_trainer.py:1466` | `acc = 100.0 * correct/total` — pp end-to-end |
| Shapley engine | `simic/attribution/counterfactual.py:311` | Full factorial for n ≤ 4; deterministic, no RNG in that mode. Build uses a NEW pure module (WI-1) — no dependency on the telemetry-gated helper (`env_factory.py:270-279`; lesson of bug esper-lite-4fe98055f7) |
| Buffer | `simic/agent/rollout_buffer.py` | `rewards[num_envs, max_steps_per_env]` :287; `max_steps_per_env = max_epochs` (`vectorized.py:1196`) ⇒ ONE episode per env per buffer, flat t_f. GAE :516-666 (`@torch.compiler.disable`), resets `_advantages_normalized` :666 |
| Pre-GAE seam | `ppo_coordinator.py:289` `run_update` | After the empty-buffer guard (:302), before `run_ppo_updates_fn` (:341). Coordinator already holds `reward_normalizer` AND `env_reward_configs` (:357) ⇒ flags reach delivery with no new constructor params. `handle_rollbacks` (:108) precedes and knows rollback env indices |
| Normalizer | `control/normalization.py:241-257` `divide_by_std` | No clip, no stat update; std=max(1e-8, sample std); <2 samples ⇒ raw passthrough. Step rewards use `update_and_normalize` (clips ±10) at `action_execution.py:1096` — the credit is intentionally on the unclipped channel (F2 bounds it upstream) |
| Sibling 1 | `rewards/contribution.py:739-746, :1268-1277` | Per-step `synergy_bonus`: tanh(interaction_sum·0.5)·0.1 (weight HARDCODED), gated on attribution_discount ≥ 0.5 ∧ bounded_attribution > 0 |
| Sibling 2 | `contribution.py:1280-1290`; `handlers/fossilize.py:59-115, :215-221`; `action_execution.py:1081-1088`; `parallel_env_state.py:119, :257` | Hindsight credit: ledger → `pending_hindsight_credit` accrued at fossilize, flushed NEXT step pre-normalizer, zeroed at reset. Gates on host-drift-confounded `total_improvement` (asymmetry noted) |
| Reward-flag home | `contribution.py:~130-373` `ContributionRewardConfig` | Precedent: `shaped_attribution_clip` :323 mirrored on `TrainingConfig` (`config.py:161`) → `to_train_kwargs` :410 → `train_ppo_vectorized` params (`vectorized.py:747`) → `ContributionRewardConfig(...)` :968-976. `__post_init__` :350 validates. Config JSONs auto-accept new dataclass fields (`config.py:278` uses `fields(cls)`) |
| Partition/telemetry | `rewards/partition.py:40-54`; `leyline/telemetry_contracts.py:41,186,249` | ADDITEND_SIGN_MAP is per-STEP-additend exhaustive. The top-up is NOT an additend (bypasses compute_reward) — it gets its own event instead (WI-7) |
| State hazards | `vectorized_trainer.py:1350-1373, :1716-1720` | alpha_schedule set/cleared un-restored in val pass (terminal-only today, tolerable); model.eval() restored at NEXT epoch's train pass. New code must not extend these hazards |
| Fossilize record | `contribution.py:48, :1134, :1146` | `fossilize_epoch` recorded ONLY when drip_fraction > 0 — build needs an always-on (env, slot, t_f) record (WI-3), captured at execution time, not reconstructed |

## 4. Work items (TDD order inside each; tests precede mechanism)

**WI-1 — Pure math module** `src/esper/simic/rewards/committed_shapley.py`
`compute_committed_shapley_topup(coalition_accs, slots, *, scale, cap, tau) -> CommittedShapleyResult`
(frozen dataclass: per-slot phi/c_paid/gap/raw/top_up, G, sum_raw, clamp_binding, k). Exact
factorial weights; `assert k <= 3` (design: refuse sampled Shapley); pure floats, no torch.
*Tests:* hand-computed 2- and 3-slot cases; null-player ⇒ phi=0 exactly; c_paid identity;
tau deadband; per-seed cap; G-clamp binding and non-binding; Σraw=0 ⇒ all-zero; k=0/1 edges.

**WI-2 — Terminal coalition evaluation** (`vectorized_trainer.py`)
In `_build_fused_val_configs`: when `reward_config.shapley_synergy_scale > 0` AND
`epoch == max_epochs` AND env has k ≥ 1 fossilized slots, inject `committed_shapley` configs =
all 2^k fossilized-slot subsets (non-fossilized active slots forced 0.0; ON-members natural
alpha, matching `committed` config semantics). Unpack pp accs in `_run_fused_val_pass`, call
WI-1, store per-env result + per-seed records for delivery. **Gated ONLY on the flag — never
on use_telemetry.** Existing telemetry-only shapley-over-active path untouched.
*Reviewer additions:* the `_kind` unpack chain gets a fail-loud `else` (unknown kind ⇒ raise);
the coalition-accs → WI-1 handoff uses direct indexing, never `.get()` defaults (a missing v(S)
is a bug, not a zero); `scale > 0` asserts `drip_fraction == 0` (BASIC_PLUS puts fossilized
slots back into `active_slot_list` :1038-1041 — semantic overlap unhandled by design) and
asserts committed slots are non-GATE (the un-restored `alpha_schedule` hazard :1350-1373 must
not silently extend to a future GATE fossilized slot).

**WI-3 — Fossilize timestep records** (`action_execution.py` / `parallel_env_state.py`)
On SUCCESSFUL fossilize execution, append `(slot_id, buffer_step_idx)` to a per-episode
`env_state.fossilize_step_records`; reset in `reset_episode_state`. The step index is the same
counter `buffer.add` uses (passed explicitly — no epoch arithmetic reconstruction). Always on
(cheap, also closes the drip-gated `fossilize_epoch` gap noted by the scout).

**WI-4 — Retro-write delivery** (`ppo_coordinator.py` + `vectorized_trainer.py` handoff)
Trainer passes the per-env terminal credit records into `run_update` (new explicit parameter).
In `run_update`, after the empty-buffer guard, before `run_ppo_updates_fn`: for each env NOT
rollback-contaminated, for each (slot, t_f, top_up_raw > 0): apply the §2 delivery formula
(normalized-space clamp PRIMARY, std-floored divisor secondary; explicit `RewardNormalizer`
accessor — NOT bare `divide_by_std`, which floors at 1e-8 and passes raw through at count<2),
`buffer.rewards[env, t_f] += credit_buf`. No clip; no stat update. Emit WI-7 event per credit.
*Rollback exclusion mechanism (named per pytorch F5):* the proven buffer-derived mask from the
GATE 2 probe (`rollback_severity != 0 | rollback_transition_types != 0`, probe :177-179) — the
drl review verified this set equals the forfeit set `mark_terminal_with_penalty` touched.
*Required tests:* `std < std_floor ⇒ credit_buf == top_up/std_floor`; normalized-cap binding
(`credit_buf == normalized_cap` when the division exceeds it); count<2 path explicit.

**WI-5 — Config flags + plumbing + HRA exclusion**
`ContributionRewardConfig`: `shapley_synergy_scale=0.0`, `shapley_synergy_noise_floor=0.0` (pp),
`shapley_synergy_cap=0.0` (pp), `shapley_synergy_std_floor=0.0` (raw-reward units),
`shapley_synergy_normalized_cap=0.0` (buffer units — the PRIMARY F2 bound, drl F-A).
`__post_init__`: all ≥ 0; `scale > 0` ⇒ ValueError unless `cap > 0` AND `normalized_cap > 0`
(the F2 bound must be structurally ON whenever the term can pay — it cannot be silently absent
at enablement).
Mirror on `TrainingConfig` + `to_train_kwargs` + `train_ppo_vectorized` params + construction
site (the 7-step `shaped_attribution_clip` pattern). Validation: `scale > 0` with
`hra_value_decomposition` ⇒ ValueError + `# TODO: [FUTURE FUNCTIONALITY]` for CF-stream routing
(the GATE 2 probe asserted HRA-off; stream semantics for a retro-written credit are undefined).

**WI-6 — F1/F8 sibling reconciliation (PROPOSED: mutual exclusion)**
When `shapley_synergy_scale > 0`: the interaction-driven per-step bonus is not paid and
hindsight credit is not accrued (gate at `handlers/fossilize.py:215-218`; flush site left
untouched — pending stays 0). Rationale: the top-up REPLACES the ad-hoc synergy channels;
structural exclusion kills the wake-the-dormant-channel double-pay risk outright, vs
assert-dormant (fragile) or net-into-c_paid (mixes fossilize-time and terminal quantities).
At scale=0 (default) both siblings behave exactly as today. *Decision documented here per F1.*
*drl F-C decided:* the coupling IS accepted — the enablement A/B confound ({siblings off} +
{top-up on} move together) is bounded by measured near-total dormancy (62/359,987 steps) and
PDR-0012's enablement criteria already include an ON-run dormancy recheck; no independent
sibling-disable flag (config surface for a dormant channel is not worth the ablation purity).

**WI-7 — Telemetry (not deferred — CLAUDE.md)**
`leyline/telemetry.py`: `CommittedShapleyPayload` (env_id, slot_ids, per-slot phi/c_paid/gap/
raw/top_up in pp, t_f, credit_buf, G, sum_raw, clamp_binding, k, std_used) +
`TelemetryEventType.COMMITTED_SHAPLEY_TOPUP`; emitted at delivery. Serves enablement monitors
directly (G-clamp binds; fossilize-count). NOTE: the credit intentionally does NOT appear in
`RewardComponentsTelemetry`/ADDITEND_SIGN_MAP — it is not a per-step additend; the step-path
invariant Σ(additends) == reward_raw remains true, and the event carries the books for the
buffer-side delta.

**WI-8 — Naming resolution (F8, No-Legacy)**
Rename the legacy interaction-driven channel `synergy_bonus` → `interaction_bonus` across src +
tests + telemetry serde (`telemetry_contracts.py:80,143,186,249`, `partition.py:45`,
`karn/mcp/views.py:527`, emitters). All call sites in one commit. After this, "synergy" means
only coalition/counterfactual-derived quantities (the new term + Karn's counterfactual display
`total_synergy`). Caveat to surface: Karn queries over OLD runs' raw events keep the old key —
historical telemetry is not rewritten. New-term code/telemetry is `committed_shapley_*`;
config flags keep the PDR-0012 names (`shapley_synergy_*`) to avoid decision-record drift.
*drl F-F:* the rename shifts the scale=0 telemetry KEY stream (the channel still fires at
scale=0) — F5 byte-identity is a training-behavior claim, not a telemetry-key claim. WI-8 gets
its own serde round-trip test + Karn-view extraction test; golden REWARD tests won't catch a
key regression.

**WI-9 — Kernel verification tests (F4/F6)**
(a) `alpha_override=0` zeroing for ARBITRARY coalitions: at `SeedSlot.forward` level (override
0.0 ⇒ output identical to host features) and at `fused_forward` level on a 2-slot toy (config
with slot masked == config with slot physically detached, other slots in arbitrary states).
*pytorch F6 nuances:* exact equality is valid ONLY for finite seed features (`0·inf = nan`) —
use finite synthetic features and assert exactness, or assert-finite first; the masked path is
the blend branch (`torch.lerp` / multiplicative), NOT the STE branch (STE fires only when
`alpha_override is None`); do not assert the seed forward is skipped — it runs and is zeroed.
(b) pp-units: unit test on the unpack path (synthetic correct/total ⇒ acc in [0,100], ×100
applied) so a fraction-returning refactor fails loudly (100× gap corruption guard).

**WI-10 — GAE retro-write unit test (MANDATED — first test written)**
Buffer-level, `test_rollout_buffer_cf_gae.py:20` `_buf` convention: inject credit at t_f,
`compute_advantages_and_returns`, assert vs hand-computed advantages: Δadv(t_f) == credit_buf;
Δadv(t) == credit_buf·(γλ)^(t_f−t) for t < t_f; 0 for t > t_f (with dones respected); and the
probe's terminal identity (γλ)^(T−1−t_f) reproduction.
*drl F-B additions (required):* truncated-terminal vs true-done at T; a done STRICTLY between
t and t_f (backward propagation must stop at the episode boundary); t_f == 0; t_f == last
step; episode shorter than buffer width (step_counts < max_steps — off-by-one guard).

**WI-11 — No-op-at-0 verification (F5)**
(a) `_build_fused_val_configs` output structurally identical at scale=0 to pre-build (no new
configs, no new forward slices). (b) Existing golden reward tests pass unmodified
(`test_reward_golden.py`). (c) `run_update` at scale=0: rewards tensor bitwise unchanged.
(d) At scale>0 the path is RNG-neutral: full-factorial (no sampling) under
`torch.inference_mode()` + eval mode; integration test asserts torch RNG state identical
across the terminal eval, AND (drl F-E) BatchNorm running stats unchanged.
*Accepted no-op deviation (drl F-D, documented):* WI-3's always-on fossilize records are the
one scale=0 behavior delta — a pure bookkeeping append, never read at scale=0 (WI-4 is driven
by top-up records, which are empty); it also closes the drip-gated `fossilize_epoch` gap.

**WI-12 — Integration test (CPU, fixed schedule)**
Mirror `tests/integration/test_gradient_measurement_telemetry_independent.py` (production
FIXED_SCHEDULE lifecycle, mock dataloaders, n_envs=1): scale>0/cap>0/tau=0 ⇒ seed fossilizes,
coalition configs evaluated, top_up ≥ 0, `buffer.rewards[env, t_f]` changed pre-GAE by
credit_buf, event emitted; use_telemetry=False variant must behave identically (bug-4fe98055f7
regression posture); rollback-env exclusion path.
*drl F-B (required):* assert the credited step IS the FOSSILIZE transition —
`buffer.effective_op_actions[env, t_f] == LifecycleOp.FOSSILIZE` — not merely "a reward
changed"; the t_f == fossilize `buffer.add` step-index correspondence is the highest-risk
detail in the build.

## 5. Reviewer-condition → work-item map

| Condition | Work items |
|---|---|
| F5 true no-op at 0 | WI-2 (flag gate), WI-11 |
| F1/F8 sibling reconciliation + naming | WI-6, WI-8 |
| F2 scale-space safety | WI-5 (cap + std_floor flags), WI-4 (applied at delivery) |
| F4/F6 kernel + terminal c_paid | WI-9, WI-1 (c_paid = v({s})−v(∅)), WI-2 |
| GAE retro-write unit test | WI-10 |
| GATE 2 retro-write mandate | WI-3, WI-4 |
| Purity refinement | WI-3 (success-only records), WI-4 (rollback exclusion) |

## 6. Open questions — RESOLVED (specialist review, 2026-07-02)

1. **WI-6 choice:** **mutual exclusion** (drl endorsed; net-into-c_paid rejected — unit/timing
   mismatch; assert-dormant rejected — fragile). Coupling confound accepted per WI-6 note.
2. **F2 shape:** **both** — normalized-space clamp PRIMARY (calibration-free), std_floor
   secondary; `__post_init__` guarantees a bound is ON whenever scale > 0 (drl F-A/pytorch F1).
3. **HRA:** **hard ValueError** (drl verified: a retro-write lands only in the total/main
   stream; CF-stream routing is undefined — `rollout_buffer.py:585-663`).
4. **v(S) ON-member alpha:** **natural alpha** (both reviewers; forced-1.0 would mis-scale φ
   and G; matches the `committed` config precedent).
5. **c_paid drift:** **confirmed** — terminal `v({s})−v(∅)` fully replaces the fossilize-time
   snapshot; it is the correct synergy decomposition (φ − standalone; null-player → 0
   preserved). The design doc gets a status-header amendment noting the semantic shift: c_paid
   is now "the standalone/synergy split," not "what the dense channel already paid"; the
   double-pay guard is null-player + τ + cap + G-clamp.
6. **WI-8 blast radius:** **rename now** (No-Legacy mandates it; golden-guarded; plus the F-F
   serde/key test).

**Enablement-gate note (pytorch, forward-looking):** injecting 2^k configs enlarges the fused
terminal batch, which can perturb config-0 (main) val_acc bitwise via cuDNN algo/reduction-order
selection — a pre-existing confound class (K already varies with telemetry on/off). Point the
enablement A/B design at it; not a build concern (scale=0 adds zero configs).

## 7. Execution order

WI-10 (mandated test, RED) → WI-1 → WI-9 → WI-5 → WI-11(a,b baseline) → WI-3 → WI-2 → WI-4 →
WI-7 → WI-6 → WI-8 → WI-11(c,d) → WI-12 → full `tests/simic` + goldens + integration sweep →
adversarial code review (reward-function-reviewer on built code + pytorch/python reviewers).
