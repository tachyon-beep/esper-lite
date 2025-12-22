# BUG-022: Bootstrap value uses pre-`step_epoch()` state on truncation

- **Title:** Truncation bootstrap computes `V(s_{t+1})` from post-action slot state *before* the mechanical `step_epoch()` tick
- **Context:** Vectorized PPO uses time-limit truncation at `epoch==max_epochs` and bootstraps from the post-transition state for return estimation
- **Impact:** P2 – biased returns/advantages at episode boundary; can harm PPO stability and credit assignment (especially for BLENDING alpha dynamics and cooldown transitions)
- **Environment:** HEAD @ workspace; multi-slot PPO (`src/esper/simic/training/vectorized.py`)
- **Reproduction Steps:**
  1. Put a slot in `BLENDING` with `alpha_controller.alpha_steps_total=1` (or any config where `step_epoch()` changes alpha/stage).
  2. End an episode via time limit (`epoch==max_epochs`).
  3. Compare the bootstrap state used for `V(s_{t+1})` vs the actual next observed state after `step_epoch()`; they differ.
- **Expected Behavior:** The bootstrap state should match the environment’s true next observation (after action execution + `step_epoch()`), since `step_epoch()` is part of the transition dynamics.
- **Observed Behavior:** Bootstrap features/masks are collected before `step_epoch()` is called later in the env loop.
- **Logs/Telemetry:** Manifests as critic/value inconsistencies near episode ends; no direct log today.
- **Hypotheses:** Bootstrap computation was added early (“post-action”) but wasn’t updated when `step_epoch()` was moved to run after transition storage.
- **Fix Plan:** Defer bootstrap-state collection until after `step_epoch()` runs (still batch it by collecting `post_step_epoch_features/masks` for all truncated envs in a second pass).
- **Validation Plan:** Integration test that constructs a deterministic BLENDING tick and asserts the bootstrap observation equals the next-step observation produced by continuing one step.
- **Status:** Open
- **Links:**
  - Bootstrap pre-tick: `src/esper/simic/training/vectorized.py:2543`
  - Mechanical tick: `src/esper/simic/training/vectorized.py:2641`
  - Tick semantics: `src/esper/kasmina/slot.py:2004`

