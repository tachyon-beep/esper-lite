# Handover — Recurrent PPO multi-epoch + P0 follow-ups

- **Date written:** 2026-06-17 (for pickup 2026-06-18 AM)
- **Branch:** `0.1.1` (HEAD `cf2cf549`). Feature branch `recurrent-ppo-multiepoch` (at `b9218273`) and its worktree at `/home/john/.config/superpowers/worktrees/esper-lite/recurrent-ppo-multiepoch` still exist; `0.1.1` was fast-forwarded onto it, so they share history. Nothing is pushed to any remote — all commits are local.
- **Host:** CUDA available (2 devices, bf16 supported). Run Python via `PYTHONPATH=src uv run ...`. CIFAR-10 present at `./data/cifar-10-batches-py`.

## TL;DR

Three commits landed on `0.1.1`, all reviewed (specialist sign-off, 0 blockers), full `tests/simic/` green (1149 passed):

| commit | what |
|--------|------|
| `356008c4` | **PR1** — anchored reference pass at K=1 (zero behavior change) |
| `b9218273` | **PR2** — multi-epoch recurrent PPO at K=4: config plumbing, guard→assertion, deleted staleness warning, K=4 goldens, EV-liftoff CI gate |
| `cf2cf549` | **P0-2** — removed the silent synthetic-data fallback in `utils/data.py` |

**The infrastructure is correct and done. The headline outcome is NOT: real-world EV-liftoff has not been achieved.** That is the main open work (see Next Steps).

## The honest open problem — real EV-liftoff

The spec's goal was to lift the critic's `explained_variance` off ~0 by giving it more gradient steps per rollout (K=4 vs K=1). Result:

- **Controlled CI gate PASSES** (`tests/integration/test_ev_liftoff_k4.py`, run with `-m integration`): on a *stationary* target, K=4 EV→0.47 vs K=1→0.09. This proves the *mechanism* (more steps → faster critic fit; K=4 drives `value_loss`→1e-5).
- **Real CUDA CIFAR run does NOT lift EV** (16 episodes, bf16): EV pinned ~0 at BOTH K=4 (max 0.017) and K=1 (max 0.007). K=4 is **not** better than K=1 on real, non-stationary returns (`return_std` swings 1→73). This reproduces the spec's *original* symptom (critic tracks the mean, develops no variance).

**Why:** EV is computed against **rollout-stored** values (`ppo_agent.py:597`), so liftoff is a *closed-loop* signal — it needs the critic to track a non-stationary cross-rollout target, which multi-epoch alone does not deliver. The critic was *gradient-starved, not dead*; multi-epoch is **necessary but not sufficient**.

**Two confounds in the short real run (so it is not a clean failure, just inconclusive):**
1. The `total_train_steps = n_episodes` plumbing (correct in principle) compresses the entropy-anneal schedule on SHORT runs → slot entropy collapsed to 0 by update 3. A long run anneals normally.
2. 16 updates is far too few against a target with `return_std` swinging 1→73.

## Critical technical context (do NOT relearn the hard way)

1. **AMP cast-cache poisoning (PR1).** A `no_grad` anchor forward under BF16 autocast poisons torch's per-parameter cast-weight cache → severs the epoch-0 backward → all 8 action-head grads come back NaN/None (value head survives). The plan's original mitigation was **inverted** (it said run the anchor *under* autocast — that *causes* it). The fix is `torch.clear_autocast_cache()` immediately after the anchor (`ppo_agent.py`, in `update()` right after the anchor block). The CUDA discovery test `test_epoch0_per_head_grad_norms_nonzero_under_amp` is the load-bearing gate — keep it and run it on CUDA. Production `amp_dtype="auto"`→bf16 on Ampere, so this was a real latent prod bug.
2. **EV semantics.** `explained_variance` (`ppo_agent.py:597`) uses `data["values"]` (rollout-stored), not fresh critic predictions. Any EV experiment MUST regenerate rollouts from the current network each cycle, or it is artifactual (a frozen-buffer replay shows EV flat by construction — I hit this).
3. **`clip_value` stays False under K>1.** Now hard-enforced by a `ValueError` in `PPOAgent.__init__` (value clipping would anchor on un-recomputed rollout `old_values`). Re-enabling under K>1 needs an anchored `ref_values` pass — a separate task.
4. **Guards added (review-driven):** `recurrent_n_epochs >= 1` and `clip_value`+K>1 both raise in `__init__`. Tests: `tests/simic/test_ppo.py::test_recurrent_n_epochs_below_one_raises`, `::test_clip_value_with_multiepoch_raises`.
5. **P0-2 side effect:** a missing/broken CIFAR-10 now ABORTS a run (RuntimeError) instead of silently feeding synthetic noise. Ensure the dataset is present before any real run. The explicit `mock=True` opt-in still exists for tests.

## Prioritized next steps

### 1. (P1) Real EV-liftoff — the headline that is still open
- **Run a PROPER long K=4 vs K=1 run** (hundreds–thousands of episodes, so `total_train_steps` is realistic and entropy anneals normally). Capture the pre-update `explained_variance` trend from telemetry (`events.jsonl` → parse `explained_variance`, or via Karn once it indexes the run dir).
- **Apply the spec's deferred secondary lever:** value-head init `gain=0.01 → 0.1` at `src/esper/tamiyo/networks/factored_lstm.py:504-511` (NOT 1.0 — that was contested/reverted; 0.1 matches the contribution-predictor regression gain). Re-run and compare EV.
- **Investigate the entropy collapse** on short runs — confirm it is the compressed-schedule artifact and not a deeper issue; it may need its own handling if it shows up on long runs.
- If EV still does not lift with K=4 + gain=0.1, the cause is structural → do P0-1 first.

### 2. (P0-1) The deeper critic fix — op-conditioned baseline / single-sample value target
- The PPO baseline is action-conditioned `Q(s,op)` and the GAE bootstrap is a single behaviour-sampled `Q(s',op')`. This caps EV regardless of K. Fix: train an op-independent `V(s)` head for the baseline/GAE, OR use the marginal `V(s)=Σ_op π(op|s)·Q(s,op)` (the batched all-ops path at `ppo_agent.py:792-798` already evaluates every op, so it's cheap).
- Files: `tamiyo/networks/factored_lstm.py:557-576,730-757`, `simic/agent/rollout_buffer.py:509-557`, `simic/agent/ppo_update.py:341-353`.
- Reviewers: `drl-expert` (mandatory) + `yzmir-deep-rl`. This is likely the real unlock for EV-liftoff.

### 3. (P0-3) Governor-rollback credit assignment
- `mark_terminal_with_penalty` zeroes the ENTIRE episode's intermediate rewards on rollback (`rollout_buffer.py:797-814`). Decide if full-zeroing is intended; if not, scale the penalty or apply to last-k transitions, and add a telemetry counter for rollback frequency + steps zeroed. Reviewer: `drl-expert`.

## How to run things

```bash
WT=/home/john/.config/superpowers/worktrees/esper-lite/recurrent-ppo-multiepoch  # or just the 0.1.1 main tree
cd "$WT"

# Fast PR gate (simic):
PYTHONPATH=src uv run pytest tests/simic/ -q -p no:cacheprovider

# EV-liftoff CI gate (deselected by default; needs -m integration):
PYTHONPATH=src uv run pytest -m integration tests/integration/test_ev_liftoff_k4.py -q

# CUDA AMP grad gate (the load-bearing PR1 safety test):
PYTHONPATH=src uv run pytest tests/simic/test_anchor_reference_pass.py::test_epoch0_per_head_grad_norms_nonzero_under_amp -q

# Real K=4 vs K=1 run via the production entrypoint (NOTE: train_ppo_vectorized REQUIRES slots=["r0c0"]):
#   build a small driver that calls esper.simic.training.vectorized.train_ppo_vectorized(
#       n_episodes=<large>, device="cuda:0", task="cifar_baseline",
#       recurrent_n_epochs=4, ppo_updates_per_batch=1, amp=True, amp_dtype="bfloat16",
#       slots=["r0c0"], use_telemetry=True, telemetry_dir="<dir>", seed=42)
#   The CLI (esper.scripts.train) does NOT expose --recurrent-n-epochs; add a flag or use a driver.
#   EV trend: parse "explained_variance" from <telemetry_dir>/telemetry_*/events.jsonl
```

## Pointers
- Design spec (corrected): `docs/superpowers/specs/2026-06-17-recurrent-ppo-multiepoch-design.md`
- Executable plan: `docs/plans/ready/2026-06-17-recurrent-ppo-multiepoch-plan.md`
- EV gate + bounds: `tests/integration/test_ev_liftoff_k4.py`
- Durable context/confounds: agent memory `recurrent-ppo-p0-confounds.md` (auto-loaded via MEMORY.md)
- Ephemeral EV experiment scripts were in `/tmp/ev_*.py` (may be gone; the CI gate is the reproducible version).
