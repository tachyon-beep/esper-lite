# Tamiyo-Centric CLI Reframing

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reframe the training CLI to be Tamiyo-centric, making the meta-learning structure explicit: `--rounds` for training iterations, `--envs` for parallel sample diversity, and exposing key Tamiyo parameters.

**Architecture:** Add new CLI flags that map to existing TrainingConfig fields. Keep internal field names unchanged for JSON config compatibility, but provide Tamiyo-centric CLI surface. Update help text to explain the two-level RL structure.

**Tech Stack:** argparse CLI, Python dataclasses

---

## Task 1: Add Core Tamiyo-Centric CLI Flags

**Files:**
- Modify: `src/esper/scripts/train.py:187-208`
- Test: `tests/scripts/test_train.py` (add to existing file)

**Context:** Currently `n_episodes`, `n_envs`, and `max_epochs` are only configurable via `--preset` or `--config-json`. Add explicit CLI flags with Tamiyo-centric names.

**Step 1: Write failing test for new CLI flags**

```python
# Add to tests/scripts/test_train.py

from esper.scripts.train import build_parser


class TestTamiyoCentricFlags:
    """Tests for Tamiyo-centric CLI flags."""

    def test_rounds_flag_accepted(self):
        """--rounds should set n_episodes in config."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--rounds", "50"])
        assert args.rounds == 50

    def test_envs_flag_accepted(self):
        """--envs should set n_envs in config."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--envs", "8"])
        assert args.envs == 8

    def test_episode_length_flag_accepted(self):
        """--episode-length should set max_epochs in config."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--episode-length", "30"])
        assert args.episode_length == 30
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/scripts/test_train.py::TestTamiyoCentricFlags -v`
Expected: FAIL (flags don't exist)

**Step 3: Add validation helper and CLI flags to ppo_parser**

In `src/esper/scripts/train.py`, add a validation helper near the top of the file (after imports, around line 25):

```python
def _positive_int(value: str) -> int:
    """Argparse type for positive integers (>= 1)."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {ivalue})")
    return ivalue
```

Then add the CLI flags after line 206 (after `--dual-ab`, before `return parser`):

```python
    # === Tamiyo Training Scale ===
    # These control how much and how Tamiyo learns, exposed with Tamiyo-centric names.
    ppo_parser.add_argument(
        "--rounds",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Tamiyo training iterations. Each round = one PPO update using data from all envs. "
             "(Maps to n_episodes in config. Default: 100)",
    )
    ppo_parser.add_argument(
        "--envs",
        type=_positive_int,
        default=None,
        metavar="K",
        help="Parallel CIFAR environments per round. More envs = richer/more diverse data per "
             "Tamiyo update, but same number of updates. (Maps to n_envs. Default: 4)",
    )
    ppo_parser.add_argument(
        "--episode-length",
        type=_positive_int,
        default=None,
        metavar="L",
        help="Timesteps per environment per round. Each round produces envs × episode_length "
             "transitions for Tamiyo. (Maps to max_epochs. Default: 25)",
    )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/scripts/test_train.py::TestTamiyoCentricFlags -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/scripts/train.py tests/scripts/test_train.py
git commit -m "feat(cli): add Tamiyo-centric training scale flags

Adds --rounds, --envs, --episode-length as explicit CLI flags
with help text explaining the meta-learning structure:
- rounds = Tamiyo PPO iterations (was n_episodes)
- envs = parallel sample diversity (was n_envs)
- episode-length = timesteps per env (was max_epochs)

Includes _positive_int() validator for early error messages.

Per DRL specialist review: makes the two-level RL explicit."
```

---

## Task 2: Wire CLI Flags to TrainingConfig

**Files:**
- Modify: `src/esper/scripts/train.py:496-501` (after existing overrides, before A/B test handling)
- Test: `tests/scripts/test_train.py`

**Context:** The new flags need to override TrainingConfig values when provided. Insert after line 500 (after `gradient_telemetry_stride` handling) and before line 502 (A/B test handling).

**Step 1: Write test for flag-to-config wiring**

```python
# Add to TestTamiyoCentricFlags class in tests/scripts/test_train.py

    def test_rounds_overrides_config(self):
        """--rounds should override n_episodes from preset."""
        from esper.simic.training import TrainingConfig

        # Simulate what main() does: start with preset, apply CLI overrides
        config = TrainingConfig.for_cifar10()
        assert config.n_episodes == 100  # Default

        # CLI would set rounds=50, which maps to n_episodes
        config.n_episodes = 50  # Simulating the override
        assert config.n_episodes == 50

    def test_envs_overrides_config(self):
        """--envs should override n_envs from preset."""
        from esper.simic.training import TrainingConfig

        config = TrainingConfig.for_cifar10()
        config.n_envs = 8
        assert config.n_envs == 8
```

**Step 2: Run test to verify baseline passes**

Run: `PYTHONPATH=src uv run pytest tests/scripts/test_train.py::TestTamiyoCentricFlags::test_rounds_overrides_config -v`
Expected: PASS (this tests our intent, not the wiring yet)

**Step 3: Add override logic in main()**

In `main()`, add after line 500 (after `config.gradient_telemetry_stride = 1`), before line 502 (`# Handle A/B testing`):

```python
                # === Tamiyo-centric CLI overrides ===
                if args.rounds is not None:
                    config.n_episodes = args.rounds
                if args.envs is not None:
                    config.n_envs = args.envs
                if args.episode_length is not None:
                    # chunk_length MUST equal max_epochs per TrainingConfig validation
                    config.max_epochs = args.episode_length
                    config.chunk_length = args.episode_length
```

**Note:** We always update both `max_epochs` and `chunk_length` because `TrainingConfig._validate()` enforces `chunk_length == max_epochs` (see config.py:376-378). Attempting to set them differently would fail validation.

**Step 4: Run train.py tests**

Run: `PYTHONPATH=src uv run pytest tests/scripts/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/esper/scripts/train.py tests/scripts/test_train.py
git commit -m "feat(cli): wire Tamiyo-centric flags to TrainingConfig

--rounds overrides n_episodes
--envs overrides n_envs
--episode-length overrides max_epochs and chunk_length (must match)"
```

---

## Task 3: Add Advanced Tamiyo Parameters

**Files:**
- Modify: `src/esper/scripts/train.py`
- Test: `tests/scripts/test_train.py`

**Context:** Per DRL specialist, expose `ppo_updates_per_batch` and `lstm_hidden_dim` as CLI flags.

**Step 1: Write failing test**

```python
# Add to TestTamiyoCentricFlags class in tests/scripts/test_train.py

    def test_ppo_epochs_flag_accepted(self):
        """--ppo-epochs should set ppo_updates_per_batch."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--ppo-epochs", "3"])
        assert args.ppo_epochs == 3

    def test_memory_size_flag_accepted(self):
        """--memory-size should set lstm_hidden_dim."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--memory-size", "256"])
        assert args.memory_size == 256
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/scripts/test_train.py::TestTamiyoCentricFlags::test_ppo_epochs_flag_accepted -v`
Expected: FAIL

**Step 3: Add advanced CLI flags**

Add after the `--episode-length` flag (still using `_positive_int` for validation):

```python
    ppo_parser.add_argument(
        "--ppo-epochs",
        type=_positive_int,
        default=None,
        metavar="E",
        help="Gradient steps per round (passes over rollout data). Higher = more sample-efficient "
             "but risks overfitting. (Maps to ppo_updates_per_batch. Default: 1)",
    )
    ppo_parser.add_argument(
        "--memory-size",
        type=_positive_int,
        default=None,
        metavar="H",
        help="Tamiyo's LSTM hidden dimension (temporal reasoning capacity). "
             "Smaller = faster but less temporal memory. (Maps to lstm_hidden_dim. Default: 128)",
    )
```

**Step 4: Wire to config in main()**

Add to the Tamiyo-centric CLI overrides section (after the `--episode-length` wiring):

```python
                if args.ppo_epochs is not None:
                    config.ppo_updates_per_batch = args.ppo_epochs
                if args.memory_size is not None:
                    config.lstm_hidden_dim = args.memory_size
```

**Step 5: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/scripts/test_train.py::TestTamiyoCentricFlags -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/esper/scripts/train.py tests/scripts/test_train.py
git commit -m "feat(cli): add --ppo-epochs and --memory-size flags

Per DRL specialist:
- --ppo-epochs controls gradient steps per round (sample efficiency)
- --memory-size controls Tamiyo's LSTM capacity"
```

---

## Task 4: Add entropy_anneal_rounds Alias for Consistency

**Files:**
- Modify: `src/esper/simic/training/config.py:243-257` (`from_dict()` method)
- Modify: `src/esper/scripts/train.py`
- Test: `tests/simic/test_config.py`

**Context:** DRL specialist noted `entropy_anneal_episodes` should have an alias `entropy_anneal_rounds` for consistency with `--rounds`. We'll add an alias in `from_dict()` for JSON config compatibility.

**Step 1: Write test for alias (including conflict detection)**

```python
# Add to tests/simic/test_config.py

def test_entropy_anneal_rounds_alias():
    """entropy_anneal_rounds should be an alias for entropy_anneal_episodes."""
    from esper.simic.training import TrainingConfig

    # Old name still works
    config = TrainingConfig(entropy_anneal_episodes=50)
    assert config.entropy_anneal_episodes == 50

    # New name also works (via from_dict for JSON compat)
    config2 = TrainingConfig.from_dict({"entropy_anneal_rounds": 75})
    assert config2.entropy_anneal_episodes == 75


def test_entropy_anneal_alias_conflict_rejected():
    """Specifying both entropy_anneal_rounds and entropy_anneal_episodes should fail."""
    import pytest
    from esper.simic.training import TrainingConfig

    with pytest.raises(ValueError, match="Cannot specify both"):
        TrainingConfig.from_dict({
            "entropy_anneal_rounds": 50,
            "entropy_anneal_episodes": 100,
        })
```

Also add to `TestTamiyoCentricFlags` class in `tests/scripts/test_train.py`:

```python
    def test_entropy_anneal_rounds_flag_accepted(self):
        """--entropy-anneal-rounds should parse correctly."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--entropy-anneal-rounds", "50"])
        assert args.entropy_anneal_rounds == 50

    def test_entropy_anneal_rounds_accepts_zero(self):
        """--entropy-anneal-rounds should accept 0 (no annealing)."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--entropy-anneal-rounds", "0"])
        assert args.entropy_anneal_rounds == 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_config.py::test_entropy_anneal_rounds_alias -v`
Expected: FAIL (alias doesn't exist)

**Step 3: Add alias handling in from_dict with conflict detection**

In `src/esper/simic/training/config.py`, update `from_dict()` at line 243. Add alias handling BEFORE `_validate_known_keys()`:

```python
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary, rejecting unknown keys.

        Supports aliases:
        - entropy_anneal_rounds -> entropy_anneal_episodes (Tamiyo-centric naming)
        """
        # Handle aliases before validation (detect conflicts)
        if "entropy_anneal_rounds" in data:
            if "entropy_anneal_episodes" in data:
                raise ValueError(
                    "Cannot specify both 'entropy_anneal_rounds' and 'entropy_anneal_episodes' "
                    "(they are aliases for the same field)"
                )
            data = dict(data)  # Don't mutate input
            data["entropy_anneal_episodes"] = data.pop("entropy_anneal_rounds")

        cls._validate_known_keys(data)
        # ... rest unchanged (lines 246-257)
```

**Step 4: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_config.py::test_entropy_anneal_rounds_alias tests/simic/test_config.py::test_entropy_anneal_alias_conflict_rejected -v`
Expected: PASS

**Step 5: Add CLI flag**

In train.py, add after `--memory-size`:

```python
    # Note: Uses type=int (not _positive_int) since 0 is valid (no annealing)
    ppo_parser.add_argument(
        "--entropy-anneal-rounds",
        type=int,
        default=None,
        metavar="R",
        help="Rounds over which to anneal entropy coefficient. 0 = no annealing. "
             "(Maps to entropy_anneal_episodes. Default: 0)",
    )
```

And wire it in the Tamiyo-centric CLI overrides section:

```python
                if args.entropy_anneal_rounds is not None:
                    config.entropy_anneal_episodes = args.entropy_anneal_rounds
```

**Step 6: Run full config tests**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_config.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/esper/simic/training/config.py src/esper/scripts/train.py tests/simic/test_config.py
git commit -m "feat(config): add entropy_anneal_rounds alias with conflict detection

Adds Tamiyo-centric alias for entropy_anneal_episodes.
Both JSON key and CLI flag supported for consistency with --rounds.
Rejects configs that specify both aliases (undefined behavior prevention)."
```

---

## Task 5: Update Help Text and Documentation

**Files:**
- Modify: `src/esper/scripts/train.py` (add epilog with `textwrap.dedent`)
- Modify: `README.md:166-182` (replace "Config-first workflow" table)

**Context:** Add clear documentation explaining the two-level RL structure and what each parameter controls.

**Step 1: Add epilog to PPO parser**

In `build_parser()`, add to the imports at top of train.py (around line 6, with other stdlib imports):

```python
from textwrap import dedent
```

Then after creating ppo_parser (after the last `add_argument` call, before `return parser`), add:

```python
    ppo_parser.epilog = dedent("""
        Tamiyo Training Parameters:
          Esper has a two-level RL structure:

          Inner loop: CIFAR environments train small neural networks
          Outer loop: Tamiyo (policy network) learns to control them via PPO

          Each "round" runs all --envs environments for --episode-length timesteps,
          then performs one PPO update on Tamiyo using the collected experience.

          Total Tamiyo transitions per round: envs × episode_length

          Example:
            --rounds 100 --envs 4 --episode-length 25
            = 100 PPO updates, each using 4 × 25 = 100 transitions
    """).strip()
    ppo_parser.formatter_class = argparse.RawDescriptionHelpFormatter
```

**Note:** We use `dedent().strip()` to remove leading/trailing whitespace and `RawDescriptionHelpFormatter` to preserve the manual formatting in the epilog.

**Step 2: Update README CLI section**

In `README.md`, **replace** lines 166-182 (the "Config-first workflow" section) with:

```markdown
#### Training Scale (Tamiyo-Centric)

These flags control Tamiyo's training directly. All are optional - presets provide sensible defaults.

| Flag | Default | Description |
|------|---------|-------------|
| `--rounds N` | 100 | Tamiyo PPO training iterations |
| `--envs K` | 4 | Parallel CIFAR environments (sample diversity per round) |
| `--episode-length L` | 25 | Timesteps per environment per round |
| `--ppo-epochs E` | 1 | Gradient steps per round (passes over rollout data) |
| `--memory-size H` | 128 | Tamiyo LSTM hidden dimension |
| `--entropy-anneal-rounds R` | 0 | Rounds over which to anneal entropy (0 = no annealing) |

Each round produces `K × L` transitions for Tamiyo's PPO update.
Doubling `--rounds` = 2× training time. Doubling `--envs` = richer data per round, same training time.

#### Config & Presets

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | `cifar10` | Hyperparameter preset: `cifar10`, `cifar10_stable`, `cifar10_deep`, `cifar10_blind`, `tinystories` |
| `--config-json` | (none) | Path to JSON config (strict: unknown keys fail) |
| `--task` | `cifar10` | Task preset for dataloaders/topology |
| `--seed` | (config default) | Override run seed |
```

**Step 3: Verify help output**

Run: `PYTHONPATH=src python -m esper.scripts.train ppo --help`
Expected: New flags and epilog visible, formatting preserved

**Step 4: Commit**

```bash
git add src/esper/scripts/train.py README.md
git commit -m "docs(cli): add Tamiyo training parameter documentation

Adds epilog explaining the two-level RL structure and
updates README with the new Tamiyo-centric CLI flags.
Uses RawDescriptionHelpFormatter to preserve epilog formatting."
```

---

## Task 6: Update Leyline Constants Documentation

**Files:**
- Modify: `src/esper/leyline/__init__.py`

**Context:** Update the comments on DEFAULT_N_ENVS and DEFAULT_EPISODE_LENGTH to explain their Tamiyo-centric meaning.

**Step 1: Update constant comments**

Find and update:

```python
# Episode length for CIFAR environments.
# This is the "rollout length" for Tamiyo - how many timesteps each env
# contributes to one Tamiyo training batch. Longer = more temporal context
# but slower iteration.
# Used by: config.py, vectorized.py, ppo.py (chunk_length, max_steps_per_env)
DEFAULT_EPISODE_LENGTH = 25

# Parallel environments for vectorized training.
# This controls sample DIVERSITY per Tamiyo update, not training quantity.
# More envs = richer/more varied experience per PPO batch, but same number
# of Tamiyo gradient updates. Affects GPU memory usage.
# Used by: config.py, vectorized.py, ppo.py, train.py CLI
DEFAULT_N_ENVS = 4
```

**Step 2: Commit**

```bash
git add src/esper/leyline/__init__.py
git commit -m "docs(leyline): clarify Tamiyo-centric meaning of defaults

Updates DEFAULT_N_ENVS and DEFAULT_EPISODE_LENGTH comments to
explain their role in Tamiyo's training, not just CIFAR mechanics."
```

---

## Task 7: Integration Test

**Files:**
- Test: `tests/scripts/test_train.py`

**Step 1: Write integration test**

```python
# Add to TestTamiyoCentricFlags class in tests/scripts/test_train.py

    def test_full_tamiyo_cli_integration(self):
        """All Tamiyo-centric flags should work together."""
        parser = build_parser()
        args = parser.parse_args([
            "ppo",
            "--rounds", "50",
            "--envs", "8",
            "--episode-length", "30",
            "--ppo-epochs", "2",
            "--memory-size", "256",
            "--entropy-anneal-rounds", "25",
        ])

        assert args.rounds == 50
        assert args.envs == 8
        assert args.episode_length == 30
        assert args.ppo_epochs == 2
        assert args.memory_size == 256
        assert args.entropy_anneal_rounds == 25

    def test_invalid_rounds_rejected(self):
        """--rounds 0 should fail with clear error at parse time."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ppo", "--rounds", "0"])

    def test_negative_envs_rejected(self):
        """--envs -1 should fail with clear error at parse time."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ppo", "--envs", "-1"])
```

**Step 2: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/scripts/test_train.py::TestTamiyoCentricFlags -v`
Expected: All PASS

**Step 3: Run existing simic tests to verify no regressions**

Run: `PYTHONPATH=src uv run pytest tests/simic/ -v --tb=short`
Expected: All PASS

**Step 4: Final commit**

```bash
git add tests/scripts/test_train.py
git commit -m "test(cli): add Tamiyo-centric CLI integration tests

Verifies all new flags parse correctly, work together, and
validates input at parse time (rejects <= 0 values early)."
```

---

## Summary of Changes

| Old Name | New CLI Flag | Internal Field (unchanged) |
|----------|--------------|---------------------------|
| n_episodes | `--rounds` | `n_episodes` |
| n_envs | `--envs` | `n_envs` |
| max_epochs | `--episode-length` | `max_epochs` |
| ppo_updates_per_batch | `--ppo-epochs` | `ppo_updates_per_batch` |
| lstm_hidden_dim | `--memory-size` | `lstm_hidden_dim` |
| entropy_anneal_episodes | `--entropy-anneal-rounds` | `entropy_anneal_episodes` |

**Backward Compatibility:**
- JSON configs continue to use internal field names
- `entropy_anneal_rounds` added as JSON alias for consistency
- All changes are additive (no breaking changes)
