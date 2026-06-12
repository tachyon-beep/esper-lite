"""Tests for vectorized environment construction helpers."""

from __future__ import annotations

import pytest

from esper.simic.training.env_factory import make_env_seed


def test_make_env_seed_separates_batches_and_envs() -> None:
    """Batch and env dimensions should not collide in the seed stream."""
    root_seed = 1234
    envs_per_batch = 12

    seeds = {
        make_env_seed(
            root_seed=root_seed,
            batch_idx=batch_idx,
            env_idx=env_idx,
            envs_per_batch=envs_per_batch,
        )
        for batch_idx in range(2)
        for env_idx in range(envs_per_batch)
    }

    assert len(seeds) == 2 * envs_per_batch
    assert make_env_seed(
        root_seed=root_seed,
        batch_idx=0,
        env_idx=10,
        envs_per_batch=envs_per_batch,
    ) != make_env_seed(
        root_seed=root_seed,
        batch_idx=1,
        env_idx=0,
        envs_per_batch=envs_per_batch,
    )


def test_make_env_seed_rejects_env_index_outside_batch() -> None:
    """The seed helper should fail when the caller gives an impossible env index."""
    with pytest.raises(ValueError, match="env_idx"):
        make_env_seed(root_seed=1234, batch_idx=0, env_idx=4, envs_per_batch=4)


def test_make_env_seed_locks_deterministic_vectorized_trace_sequence() -> None:
    """A vectorized rollout's batch/env coordinate trace maps to stable seeds."""
    root_seed = 9001
    envs_per_batch = 4

    trace = [
        make_env_seed(
            root_seed=root_seed,
            batch_idx=batch_idx,
            env_idx=env_idx,
            envs_per_batch=envs_per_batch,
        )
        for batch_idx in range(3)
        for env_idx in range(envs_per_batch)
    ]

    assert trace == [
        9001,
        10010,
        11019,
        12028,
        13037,
        14046,
        15055,
        16064,
        17073,
        18082,
        19091,
        20100,
    ]
    assert trace == sorted(trace)
    assert len(trace) == len(set(trace))
