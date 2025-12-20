# BUG Template

- **Title:** PPO reward normalizer state not saved/restored, causing resume drift
- **Context:** Simic PPO vectorized training (`src/esper/simic/training/vectorized.py`) uses `RewardNormalizer` for critic stability but historically did not checkpoint its state.
- **Impact:** P1 – resuming from a checkpoint reinitializes the reward normalizer (mean/var/reset), changing value targets and potentially destabilizing training. Checkpoints are not fidelity-preserving for the critic and can regress after resume.
- **Environment:** Main branch; any PPO run that uses `train_ppo_vectorized` and then resumes via `--resume`.
- **Reproduction Steps:**
  1. Start PPO training with nonzero rewards; stop early and save.
  2. Resume from the checkpoint and inspect normalized rewards/critic loss; they jump because the reward normalizer reset to initial stats.
- **Expected Behavior:** Reward normalizer statistics (mean, var, count, clip) are persisted and restored with the checkpoint alongside the agent/network/obs_normalizer.
- **Observed Behavior:** Only `obs_normalizer` was saved in checkpoint metadata; `RewardNormalizer` state was discarded and rebuilt fresh on resume.
- **Hypotheses:** Save metadata includes obs_normalizer but omits reward normalizer; critic relies on normalized rewards so state reset shifts targets.
- **Fix Plan:** Serialize reward normalizer (mean, var, count, clip) into checkpoint metadata and restore on load; add tests to ensure resumed normalized rewards match pre-save behavior.
- **Validation Plan:** Run a short PPO, save, resume, and confirm reward normalizer state continuity (identical normalized rewards for a fixed input); add unit test around save/load.
- **Status:** Closed (Fixed)
- **Resolution:** Reward normalizer state (`mean`, `m2`, `count`) is now persisted in checkpoint metadata on save and restored on resume; a regression test asserts the resume path actually applies the metadata.
- **Links:** `src/esper/simic/training/vectorized.py`, `tests/simic/test_reward_normalizer_checkpoint.py`
