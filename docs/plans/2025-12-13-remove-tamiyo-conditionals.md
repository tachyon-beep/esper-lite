# Remove Tamiyo Mode Conditionals Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all tamiyo mode conditionals and legacy non-vectorized training code, making tamiyo the only training path.

**Architecture:** Delete dual-mode code paths (tamiyo vs non-tamiyo), rename `tamiyo_buffer` to `buffer`, rename `update_tamiyo()` to `update()`, delete `RolloutBuffer`, and remove `--vectorized` flag from CLI.

**Tech Stack:** Python, PyTorch, PPO RL

---

## Task 1: Remove RolloutBuffer from __init__.py exports

**Files:**
- Modify: `src/esper/simic/__init__.py:27-31`

**Step 1: Delete the RolloutBuffer import block**

Delete lines 27-31:
```python
# Buffers
from esper.simic.buffers import (
    RolloutStep,
    RolloutBuffer,
)
```

**Step 2: Remove from __all__ list**

Delete lines 122-124 (the "Buffers" section entries):
```python
    # Buffers
    "RolloutStep",
    "RolloutBuffer",
```

**Step 3: Run tests to verify no import errors**

Run: `python -c "from esper.simic import *; print('OK')"`
Expected: OK (with possible warnings about unused imports elsewhere)

**Step 4: Commit**

```bash
git add src/esper/simic/__init__.py
git commit -m "refactor(simic): remove RolloutBuffer from package exports"
```

---

## Task 2: Remove tamiyo conditionals from PPOAgent.__init__

**Files:**
- Modify: `src/esper/simic/ppo.py:137-276`

**Step 1: Remove tamiyo parameter and flag**

In `__init__` signature (line 171), delete:
```python
        tamiyo: bool = False,  # Use FactoredRecurrentActorCritic + TamiyoRolloutBuffer
```

Delete line 179:
```python
        self.tamiyo = tamiyo
```

**Step 2: Remove tamiyo_buffer initialization guard**

Delete line 207:
```python
        self.tamiyo_buffer = None
```

**Step 3: Remove if tamiyo conditional for network/buffer creation**

Replace lines 209-224 (the `if tamiyo:` block and `else:` block) with just the tamiyo path, renaming `tamiyo_buffer` to `buffer`:

```python
        # Unified factored + recurrent mode
        self.network = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            lstm_hidden_dim=lstm_hidden_dim,
        ).to(device)
        self.buffer = TamiyoRolloutBuffer(
            num_envs=num_envs,
            max_steps_per_env=max_steps_per_env,
            state_dim=state_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            device=torch.device(device),
        )
```

**Step 4: Update weight decay parameter groups**

Replace lines 245-263 (the `if self.tamiyo:` / `else:` blocks for parameter groups) with just the tamiyo path:

```python
            # FactoredRecurrentActorCritic: slot_head, blueprint_head, blend_head, op_head are actors
            actor_params = (
                list(self._base_network.slot_head.parameters()) +
                list(self._base_network.blueprint_head.parameters()) +
                list(self._base_network.blend_head.parameters()) +
                list(self._base_network.op_head.parameters())
            )
            critic_params = list(self._base_network.value_head.parameters())
            shared_params = (
                list(self._base_network.feature_net.parameters()) +
                list(self._base_network.lstm.parameters()) +
                list(self._base_network.lstm_ln.parameters())
            )
```

**Step 5: Remove non-tamiyo buffer initialization**

Delete lines 274-275:
```python
        if not self.tamiyo:
            self.buffer = RolloutBuffer()
```

**Step 6: Run tests**

Run: `pytest tests/simic/test_ppo.py -v -x`
Expected: May fail (we haven't removed RolloutBuffer import yet)

**Step 7: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "refactor(simic): remove tamiyo conditionals from PPOAgent.__init__"
```

---

## Task 3: Remove RolloutBuffer import from ppo.py

**Files:**
- Modify: `src/esper/simic/ppo.py:19`

**Step 1: Delete the import**

Delete line 19:
```python
from esper.simic.buffers import RolloutBuffer
```

**Step 2: Run import check**

Run: `python -c "from esper.simic.ppo import PPOAgent; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "refactor(simic): remove RolloutBuffer import from ppo.py"
```

---

## Task 4: Delete store_transition method

**Files:**
- Modify: `src/esper/simic/ppo.py:375-400`

**Step 1: Delete the entire method**

Delete lines 375-400 (the `store_transition` method):
```python
    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_mask: torch.Tensor,
        truncated: bool = False,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Store transition in buffer.

        Args:
            state: Observation tensor
            action: Action taken
            log_prob: Log probability of action
            value: Value estimate
            reward: Reward received
            done: Whether episode ended
            action_mask: Binary mask of valid actions (stored for PPO update)
            truncated: Whether episode ended due to time limit
            bootstrap_value: Value to bootstrap from if truncated
        """
        self.buffer.add(state, action, log_prob, value, reward, done, action_mask, truncated, bootstrap_value)
```

**Step 2: Run tests**

Run: `pytest tests/simic/test_ppo.py -v -x`
Expected: Tests that use store_transition will fail (expected)

**Step 3: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "refactor(simic): delete store_transition method (non-vectorized path)"
```

---

## Task 5: Rename update_tamiyo to update and delete old update

**Files:**
- Modify: `src/esper/simic/ppo.py:402-877`

**Step 1: Delete the old update() method**

Delete lines 550-877 (the entire non-tamiyo `update()` method).

**Step 2: Rename update_tamiyo to update**

At what is now line 402, change:
```python
    def update_tamiyo(
```
to:
```python
    def update(
```

**Step 3: Update buffer reference in the renamed method**

In the renamed `update()` method, replace all occurrences of `self.tamiyo_buffer` with `self.buffer`:
- Line ~416: `if len(self.tamiyo_buffer) == 0:` → `if len(self.buffer) == 0:`
- Line ~420: `self.tamiyo_buffer.compute_advantages_and_returns(` → `self.buffer.compute_advantages_and_returns(`
- Line ~423: `self.tamiyo_buffer.normalize_advantages()` → `self.buffer.normalize_advantages()`
- Line ~426: `data = self.tamiyo_buffer.get_batched_sequences(device=self.device)` → `data = self.buffer.get_batched_sequences(device=self.device)`
- Line ~541: `self.tamiyo_buffer.reset()` → `self.buffer.reset()`

**Step 4: Run tests**

Run: `pytest tests/simic/test_ppo.py -v -x`
Expected: Some tests may fail if they reference update_tamiyo

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "refactor(simic): rename update_tamiyo() to update(), delete old update()"
```

---

## Task 6: Update save/load to remove tamiyo field

**Files:**
- Modify: `src/esper/simic/ppo.py` (save/load methods, around lines 879-951 after previous deletions)

**Step 1: Remove tamiyo from config dict in save()**

In the `save()` method's config dict, delete:
```python
                'tamiyo': self.tamiyo,
```

**Step 2: Remove tamiyo from architecture dict in save()**

In the `save()` method's architecture dict, delete:
```python
                'tamiyo': self.tamiyo,
```

**Step 3: Update load() to remove tamiyo conditional**

In `load()`, delete these lines:
```python
        is_tamiyo = arch.get('tamiyo', False)
```

Then remove the `if is_tamiyo:` / `else:` conditional entirely - just keep the tamiyo path (FactoredRecurrentActorCritic) and dedent. The non-tamiyo path (ActorCritic) is dead code.

**Step 4: Run tests**

Run: `pytest tests/simic/test_ppo.py::test_save_load -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "refactor(simic): remove tamiyo field from save/load"
```

---

## Task 7: Delete buffers.py entirely

**Files:**
- Delete: `src/esper/simic/buffers.py`

**Step 1: Verify no remaining imports (both patterns)**

Run: `grep -r "from esper.simic.buffers" src/esper/`
AND: `grep -r "import.*buffers" src/esper/simic/`
Expected: No matches for either pattern. If any matches found, fix those files first.

**Step 2: Delete the file**

```bash
rm src/esper/simic/buffers.py
```

**Step 3: Run import check**

Run: `python -c "import esper.simic; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor(simic): delete buffers.py (RolloutBuffer removed)"
```

---

## Task 8: Remove tamiyo field from TrainingConfig

**Files:**
- Modify: `src/esper/simic/config.py:113, 177-205, 207-233, 235-257`

**Step 1: Delete tamiyo field**

Delete line 113:
```python
    tamiyo: bool = False  # Use FactoredRecurrentActorCritic + TamiyoRolloutBuffer
```

**Step 2: Delete for_tamiyo() method**

Delete lines 176-205 (the entire `for_tamiyo()` static method).

**Step 3: Update default gamma and gae_lambda to tamiyo values**

Change lines 53-54:
```python
    gamma: float = 0.99
    gae_lambda: float = 0.95
```
to:
```python
    gamma: float = 0.995  # Tamiyo-optimized for 25-epoch episodes
    gae_lambda: float = 0.97  # Less bias for long delays
```

**Step 4: Remove tamiyo from to_ppo_kwargs()**

In `to_ppo_kwargs()`, delete:
```python
            "tamiyo": self.tamiyo,
```

**Step 5: Remove tamiyo from to_train_kwargs()**

In `to_train_kwargs()`, delete:
```python
            "tamiyo": self.tamiyo,
```

**Step 6: Run tests**

Run: `pytest tests/simic/test_config.py -v`
Expected: Tests that check tamiyo field will fail (expected, need to update tests)

**Step 7: Commit**

```bash
git add src/esper/simic/config.py
git commit -m "refactor(simic): remove tamiyo field from TrainingConfig"
```

---

## Task 9: Update vectorized.py buffer/update references

**Files:**
- Modify: `src/esper/simic/vectorized.py`

**Step 0: Find all tamiyo references first**

Run: `grep -n "tamiyo" src/esper/simic/vectorized.py`
Expected output will show all lines to update. The known lines are 663, 1251, 1277, 1306, 1309 but grep first to catch any others.

**Step 1: Rename tamiyo_buffer to buffer**

Change line 663:
```python
            agent.tamiyo_buffer.start_episode(env_id=env_idx)
```
to:
```python
            agent.buffer.start_episode(env_id=env_idx)
```

Change line 1251:
```python
                agent.tamiyo_buffer.add(
```
to:
```python
                agent.buffer.add(
```

Change line 1277:
```python
                    agent.tamiyo_buffer.end_episode(env_id=env_idx)
```
to:
```python
                    agent.buffer.end_episode(env_id=env_idx)
```

Change line 1306:
```python
            agent.tamiyo_buffer.reset()
```
to:
```python
            agent.buffer.reset()
```

**Step 2: Rename update_tamiyo to update**

Change line 1309:
```python
            update_metrics = agent.update_tamiyo(clear_buffer=True)
```
to:
```python
            update_metrics = agent.update(clear_buffer=True)
```

**Step 3: Run tests**

Run: `pytest tests/simic/test_vectorized.py -v -x`
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "refactor(simic): update vectorized.py to use renamed buffer/update"
```

---

## Task 10: Remove --vectorized flag from CLI

**Files:**
- Modify: `src/esper/scripts/train.py:82, 152-203`

**Step 1: Delete --vectorized argument**

Delete line 82:
```python
    ppo_parser.add_argument("--vectorized", action="store_true")
```

**Step 2: Remove conditional and always use vectorized**

Replace lines 150-203 (the entire `elif args.algorithm == "ppo":` block) with:

```python
    elif args.algorithm == "ppo":
        use_telemetry = not args.no_telemetry
        from esper.simic.vectorized import train_ppo_vectorized
        train_ppo_vectorized(
            n_episodes=args.episodes,
            n_envs=args.n_envs,
            max_epochs=args.max_epochs,
            device=args.device,
            devices=args.devices,
            task=args.task,
            use_telemetry=use_telemetry,
            lr=args.lr,
            clip_ratio=args.clip_ratio,
            entropy_coef=args.entropy_coef,
            entropy_coef_start=args.entropy_coef_start,
            entropy_coef_end=args.entropy_coef_end,
            entropy_coef_min=args.entropy_coef_min,
            entropy_anneal_episodes=args.entropy_anneal_episodes,
            gamma=args.gamma,
            save_path=args.save,
            resume_path=args.resume,
            seed=args.seed,
            num_workers=args.num_workers,
            gpu_preload=args.gpu_preload,
            telemetry_config=telemetry_config,
            slots=args.slots,
            max_seeds=args.max_seeds,
            max_seeds_per_slot=args.max_seeds_per_slot,
        )
```

**Step 3: Update docstring**

Update docstring at top of file (lines 3-12) to remove references to `--vectorized`:

```python
"""Training CLI for Simic RL algorithms.

Usage:
    # Train PPO (vectorized by default)
    PYTHONPATH=src python -m esper.scripts.train ppo --episodes 100 --n-envs 4

    # Multi-GPU PPO
    PYTHONPATH=src python -m esper.scripts.train ppo --n-envs 4 --devices cuda:0 cuda:1

    # Heuristic (h-esper)
    PYTHONPATH=src python -m esper.scripts.train heuristic --max-epochs 75 --max-batches 50
"""
```

**Step 4: Run CLI help check**

Run: `python -m esper.scripts.train ppo --help`
Expected: No --vectorized flag in output

**Step 5: Commit**

```bash
git add src/esper/scripts/train.py
git commit -m "refactor(simic): remove --vectorized flag, vectorized is now default"
```

---

## Task 11: Delete train_ppo from training.py

**Files:**
- Modify: `src/esper/simic/training.py:188-221, 556-559`

**Step 1: Delete train_ppo function**

Delete lines 188-221 (the entire `train_ppo()` function that raises NotImplementedError).

**Step 2: Remove train_ppo from __all__**

In `__all__` (around line 556), delete:
```python
    "train_ppo",
```

Also delete the reference to `run_ppo_episode` if it exists (it's not defined in this file).

**Step 3: Run tests**

Run: `pytest tests/simic/test_training.py -v`
Expected: Tests for train_ppo should fail or be gone

**Step 4: Commit**

```bash
git add src/esper/simic/training.py
git commit -m "refactor(simic): delete train_ppo stub from training.py"
```

---

## Task 12: Update tests

**Files:**
- Modify: `tests/simic/test_ppo.py`
- Modify: `tests/simic/test_config.py`
- Possibly delete: `tests/simic/test_buffers.py` (if only tests RolloutBuffer)

**Step 1: Find all affected tests**

Run these greps to identify scope:
```bash
grep -rn "tamiyo" tests/simic/
grep -rn "RolloutBuffer" tests/simic/
grep -rn "store_transition" tests/simic/
grep -rn "update_tamiyo" tests/simic/
```

**Step 2: Apply changes based on expected patterns**

Expected test changes:
- `test_ppo.py`: Remove `tamiyo=True` from PPOAgent instantiations (~5-10 occurrences)
- `test_config.py`: Delete `test_tamiyo_preset()`, `test_for_tamiyo()` if they exist
- `test_buffers.py`: Delete entirely if it only tests RolloutBuffer; keep if it tests TamiyoRolloutBuffer
- `test_training.py`: Remove any `train_ppo()` tests (the non-vectorized stub)

**Step 3: Specific changes per file**

For `test_ppo.py`:
- Remove `tamiyo=True` parameter (now default)
- Delete any tests that explicitly test `tamiyo=False` behavior
- Change `update_tamiyo()` calls to `update()`
- Change `agent.tamiyo_buffer` to `agent.buffer`

For `test_config.py`:
- Delete `test_for_tamiyo()` test function
- Remove `tamiyo` field assertions from other tests

**Step 4: Run full test suite**

Run: `pytest tests/simic/ -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add tests/
git commit -m "test(simic): update tests for tamiyo-only architecture"
```

---

## Task 13: Final verification

**Step 1: Verify no RolloutBuffer references**

Run: `grep -r "RolloutBuffer" src/esper/`
Expected: No matches

**Step 2: Verify no store_transition references**

Run: `grep -r "store_transition" src/esper/simic/`
Expected: No matches

**Step 3: Verify no tamiyo conditionals**

Run: `grep -r "if.*tamiyo" src/esper/simic/`
Expected: No matches

Run: `grep -r "self\.tamiyo" src/esper/simic/`
Expected: No matches

**Step 4: Verify tamiyo only in file names and docstrings**

Run: `grep -r "tamiyo" src/esper/simic/`
Expected: Only matches in:
- `tamiyo_buffer.py` (filename)
- `tamiyo_network.py` (filename)
- Comments/docstrings

**Step 5: Verify no vectorized flag**

Run: `grep -r "vectorized" src/esper/scripts/train.py`
Expected: No matches

**Step 6: Run full test suite**

Run: `pytest tests/simic/ -v`
Expected: All 170+ tests pass

**Step 7: Run training smoke test**

Run: `python -m esper.scripts.train ppo --episodes 1 --n-envs 2 --max-epochs 5 --slots mid`
Expected: Training completes without errors

**Step 8: Final commit**

```bash
git add -A
git commit -m "refactor(simic): complete tamiyo-only architecture migration"
```

---

## Summary

After completion:
- `tamiyo` parameter removed from PPOAgent, TrainingConfig
- `RolloutBuffer` and `store_transition()` deleted
- `update_tamiyo()` renamed to `update()`
- `tamiyo_buffer` renamed to `buffer`
- `--vectorized` CLI flag removed (vectorized is now default)
- `train_ppo()` stub deleted from training.py
- All tests updated and passing

Remaining tamiyo references (intentional):
- `tamiyo_buffer.py` - file name preserves provenance
- `tamiyo_network.py` - file name preserves provenance
- `TamiyoRolloutBuffer` - class name
- Comments explaining the architecture

---

## Future Work (Out of Scope)

### Remove unused ActorCritic from ppo.py

**Context:** The `ActorCritic` import in `ppo.py:19` is dead code. Investigation shows:

1. The `load()` method has a `is_tamiyo` conditional that uses ActorCritic state dict keys for dimension inference
2. However, ActorCritic is never instantiated - after inferring dims, `cls()` creates FactoredRecurrentActorCritic
3. Loading old non-tamiyo checkpoints is already broken (would fail on `load_state_dict`)

**Action:** After Task 6 removes the tamiyo conditional from load(), also:
- Remove `from esper.simic.networks import ActorCritic` from ppo.py
- Consider removing `ActorCritic` export from `__init__.py` if no external consumers
- Consider deprecating `ActorCritic` class in networks.py (keep for reference/other uses)
