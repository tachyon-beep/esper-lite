# Multi-Epoch Recurrent PPO via Anchored Reference Pass — Design

- **Date:** 2026-06-17
- **Status:** Approved (design) — pending spec review, then implementation planning
- **Domain:** RL training (simic PPO controller). Reviewers: drl-expert + `yzmir-deep-rl` (design contest); `yzmir-pytorch-engineering` for the AMP detail.
- **Origin:** 22-agent design-contest workflow (`wf_ed65890d-f01`), grounded in code.

## Problem

The simic PPO controller's value function is dead: `explained_variance ≈ 0.000` across an
entire run, `value_mean` a near-zero constant against returns of std 7–13, `kl ≈ 0`
(policy not moving). Because advantages are therefore noise, the (separately fixed,
validated) reward signal — FOSSILIZE now pays +2.0 — cannot propagate into policy
improvement, so fossilization stays rare.

Two discriminating experiments isolated the cause:

| Run | Change | EV result |
|-----|--------|-----------|
| `telemetry_2026-06-16_122700` | reward fix, default critic config | EV pinned ~0.000, 10 updates |
| `telemetry_2026-06-16_160350` | `value_warmup_batches=0` | EV pinned ~0.000 (H1 **ruled out**) |

Removing the value-coef warmup changed nothing — the critic moves toward the target *mean*
but develops no output *variance*. The cause is structural: **the critic gets only one
gradient step per rollout** (`ppo_updates_per_batch=1`, `recurrent_n_epochs=1` force-pinned),
so it never accumulates enough steps to fit a high-variance sparse-return target. More
gradient steps per rollout is the lever — currently blocked for LSTM policies because
reusing rollout-time hidden states after a weight update corrupts gradients
(`vectorized.py:418-424` guard; `ppo_agent.py` C4 comment).

## Key reframe (verified in code)

We set out to implement R2D2-style hidden-state burn-in. The contest found this is
**unnecessary**: the scored forward is *already* full-recompute TBPTT.

- Each PPO epoch calls `evaluate_actions(..., hidden=(initial_hidden_h, initial_hidden_c))`
  (`ppo_agent.py:898-906`), reconstructing the entire 150-step hidden trajectory from the
  episode-start state under **current** weights via one `self.lstm(features, hidden)` call.
  Because `chunk_length == max_epochs == 150` and the BPTT invariant is enforced
  (`ppo_agent.py:285-297`), there is **no stale hidden state in the gradient graph**.
- The only genuine staleness is the PPO reference baselines read each epoch from the
  immutable `data` dict. Of those, the value-clip anchor `old_values` is **inactive**
  (`clip_value=False` default, `ppo_agent.py:131`; used only when `clip_value=True`,
  `ppo_update.py:341-347`). So the lone live staleness is **`old_log_probs`** — the
  importance-sampling ratio denominator (`ppo_agent.py:1058-1067`).

Fixing that one quantity makes multi-epoch recurrent PPO mathematically exact.

## Recommended design: Anchored Reference Pass

Drive K via the **existing internal `recurrent_n_epochs` loop** (`ppo_agent.py:863`), NOT
the external `ppo_updates_per_batch` loop. GAE (`ppo_agent.py:547`) and
`value_normalizer.update()` (`ppo_agent.py:638`) run once per `update()`, before the epoch
loop; the internal loop reuses them across all K epochs (correct, fixed-θ₀ PPO). The
external loop would re-run GAE and mutate the EMA normalizer K times on the same returns —
the double-counting blocker that sank the alternative designs.

1. **Anchor forward.** After the GAE/normalizer block and before the epoch loop, run one
   `torch.no_grad()` forward at frozen θ₀ — the *same* `evaluate_actions` call as the epoch
   loop, wrapped — to produce `ref_log_probs` (detached, per head). The MAIN policy/value
   path consumes no RNG (scores stored actions; `ResidualLSTM` dropout=0.0), so `ref_log_probs`
   and the ratio are unaffected by the anchor. NOTE (PR1, corrected): the network is in
   `training` mode throughout `update()`, so the aux contribution-predictor's `Dropout(0.1)`
   DOES draw one mask in the no_grad anchor — the update is deterministic-given-seed, not
   strictly RNG-free; that draw is multiplied to zero in the aux loss on non-fresh-measurement
   timesteps and never touches the policy/value heads (drl+pytorch review, 2026-06-17).
2. **Baseline swap (only loss-affecting change).** Source `old_log_probs` from
   `ref_log_probs` instead of the rollout (`ppo_agent.py:1058-1067`); update the
   finiteness-gate `old_lp_stack` (`ppo_agent.py:946-949`) to match. **Do not touch**
   `old_values` (dead under `clip_value=False`).
3. **Knob consolidation (no-legacy single path).** Plumb `recurrent_n_epochs` through
   `TrainingConfig` → `run()` → `PPOAgent` ctor (`ppo_agent.py:142`, currently a never-passed
   ctor arg) → checkpoint persistence. Replace the deleted guard with an **assertion**:
   `ppo_updates_per_batch == 1 when lstm_hidden_dim > 0`. Rewrite the C4/H5 warnings
   (`ppo_agent.py:188-200, 251-272`) to state the new invariant.
4. **Telemetry.** The per-step hidden buffer (`rollout_buffer.py:439-440`) stays — it feeds
   the Q(s,op) telemetry diagnostic (`ppo_agent.py:728-758`). Document it as telemetry-only;
   do not delete (CLAUDE.md).

At epoch 0 the ratio is 1.0 within FP tolerance (anchor and scored forward share the
identical full-unroll at θ₀); at epoch k>0 the ratio measures the true policy change
π_θk/π_θ0 under matched recurrent context. Textbook recurrent PPO.

### Rejected alternatives
- **R2D2 segmented burn-in:** unsound — re-warms the new policy's hidden state but leaves
  `old_log_probs` frozen, mixing two hidden conditionings in the ratio.
- **Staleness-bounded refresh / external-loop on-policy refresh:** route K through the
  external loop, double-counting GAE and the EMA value normalizer.

## Locked decisions
- **Rollout: two-PR incremental.** PR1 — land the anchor fix at K=1 (proves epoch-0
  ratio==1.0, zero behavior change, validates the AMP/grad-nonzero gate in isolation).
  PR2 — flip K>1 via `recurrent_n_epochs` once the anchor is proven safe.
- **K = 4 to start, then sweep** (lean on the existing KL early-stop, `ppo_agent.py:1095-1103`);
  characterize K=8 with a fixed-seed phase-profiler run before committing larger.
- **Schedule semantics:** one tick per rollout batch (`train_steps++` once per `update()`,
  `ppo_agent.py:1380`); K is within-batch refinement. Since `ppo_updates_per_batch` is pinned
  to 1 for LSTM, the `*ppo_updates_per_batch` multipliers at `config.py:301,304` become no-ops
  for recurrent runs (correct). Document.
- **`clip_value` stays False.** Re-enabling it under K>1 (would require anchoring `ref_values`
  too) is an explicitly separate, larger task — not smuggled in here.
- **Pin `ppo_updates_per_batch = 1` for LSTM** (single multi-epoch path).

## Risks
- **AMP cast-cache poisoning (highest, silent) — CONFIRMED & RESOLVED 2026-06-17 (PR1):** a
  `no_grad` anchor forward touching head Linears under autocast populates a graph-less
  cast-weight cache; the subsequent epoch-0 *grad* forward reuses those graph-less casts, so
  the backward to every action-head `nn.Parameter` is severed → all 8 head grads come back
  `None` (emitted as a NaN sentinel). The value head survives (its loss routes to it
  independently). The epoch-0 nonzero-per-head-grad gate (`test_epoch0_per_head_grad_norms_nonzero_under_amp`,
  CUDA/BF16) caught this on first run.
  **The original mitigation in this design was INVERTED and is wrong:** running the anchor
  "inside the same BF16 autocast" is what *causes* the poisoning, and `autocast(enabled=False)`
  does NOT break ratio symmetry (the FP32 masked-logits seam holds the ratio; an FP32 anchor
  drifts only ~4.4e-5 under BF16, and the committed acceptance #1 test runs on CPU/FP32 where
  it is exact). **Resolved fix (Option C, pytorch-expert-vetted):** keep the anchor under plain
  `torch.no_grad()` (so it *inherits* the caller's precision and the ratio stays exactly 1.0
  under AMP), then call `torch.clear_autocast_cache()` immediately after the anchor block to
  evict the graph-less casts so the epoch-0 forward re-casts each weight fresh under autograd.
  Empirically: Option C → grads healthy AND ratio exact; Option A (`autocast(enabled=False)`)
  → grads healthy, ratio 4.4e-5 BF16-only drift (acceptable fallback); Option B
  (`cache_enabled=False` BF16) → disqualified (forces BF16 on the CPU/FP32 path, fails the CPU
  ratio test). Note production `amp_dtype="auto"` resolves to **bfloat16** on Ampere+, so this
  was a real latent production bug, not a test artifact. **Keep gating with the epoch-0
  nonzero-per-head-grad test.**
- **Trust-region drift at K>1:** with θ actually changing across epochs, KL can grow faster;
  mitigated by KL early-stop + clip_ratio, but K needs a sweep.
- **Golden-test invalidation (deliberate):** `test_ppo_update_golden.py` injects
  `log_prob_offset=0.1` (line 125) to force ratio≠1; the anchor makes epoch-0 ratio==1.0, so
  goldens must be regenerated and documented as an intended baseline change.
- **Config plumbing blast radius:** `recurrent_n_epochs` wiring (config → run() → ctor →
  checkpoint) must land atomically with the guard removal (no-legacy single path).
- **Perf headroom unproven:** the "~1% of wall even at K=8" rests on the 1%-vs-99% telemetry
  hypothesis the phase profiler exists to test; validate with the characterization run.

## Acceptance criteria
1. **Epoch-0 soundness:** at θ₀, `ref_log_probs` == epoch-0 scored log_probs to |diff|<1e-5
   per head ⇒ joint ratio==1.0 within tolerance, `approx_kl==0`. (Assert |ratio−1|<eps, not
   bitwise — two BF16 forwards with an FP32 seam; use `no_grad`, never `inference_mode`.)
2. **AMP nonzero-grad gate:** epoch-0 per-head grad-norms strictly > 0 after the anchor runs.
3. **Determinism/replay:** two identical-seed `update()` runs produce <1e-6 identical
   losses/metrics at K=4 (deterministic given seed; the anchor's only RNG draw — the aux
   contribution-predictor's `Dropout` — is identical across two identically-seeded runs, so
   it cannot desync them; see the corrected Anchor-forward note above).
4. **No stale-baseline consumption:** loss path reads `old_log_probs` only from
   `ref_log_probs`; `data['hidden_h'][:,1:]` never indexed in the loss path.
5. **EV lifts off 0:** on a real K=4 run, pre-update `explained_variance`
   (`ppo_agent.py:599-615`) trends from ~0 toward >0.3–0.8 and does not regress vs K=1.
   (Headline evidence.)
6. **KL boundedness:** `approx_kl` stays under the early-stop trigger (1.5·target_kl);
   `early_stop_epoch` fires sanely (not always epoch 0, not never).
7. **Single-path enforcement:** `ppo_updates_per_batch>1` with `lstm_hidden_dim>0` raises the
   new assertion; rewritten `test_vectorized.py` guard test passes.
8. **Golden regeneration:** `test_ppo_update_golden.py` goldens regenerated and passing, with
   a comment documenting epoch-0 ratio==1.0 by construction.
9. **Value-loss non-regression:** `value_loss`, `return_mean/std` finite and same order of
   magnitude as K=1 (value path untouched).

## Files touched
`ppo_agent.py` (anchor forward, `old_log_probs` swap, finiteness-gate line, delete warnings,
comment, plumb `recurrent_n_epochs`), `vectorized.py` (replace guard with assertion),
`config.py` (add `recurrent_n_epochs` + thread it), the `run()` entrypoint signature,
checkpoint load path, `test_ppo_update_golden.py` (regenerate), `test_vectorized.py` (rewrite
guard test), plus new proof tests for criteria 1–4.

## Empirical context
- Reward fix (committed `575482d7`): FOSSILIZE −0.81 → +2.00, validated.
- `value_warmup_batches=0` (`telemetry_2026-06-16_160350`): EV unchanged ⇒ warmup not the cause.
- `gain=0.01` value-head init (`factored_lstm.py:504-511`) is a secondary lever if EV still
  lags at K=4 — note the contested history (was 1.0, reverted) and the sibling
  contribution-predictor's `gain=0.1` for regression targets; prefer 0.1, not 1.0, and only
  after K>1 is in.
