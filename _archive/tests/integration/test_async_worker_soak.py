"""Soak test for the shared async worker implementation.

Run with ``RUN_SOAK_TESTS=1 pytest -m soak`` to execute the full harness.
"""

from __future__ import annotations

import os

import pytest

from esper.core.async_runner import AsyncWorker
from tests.helpers.async_worker_harness import SoakConfig, run_soak

pytestmark = pytest.mark.soak


def _should_run_soak() -> bool:
    return os.environ.get("RUN_SOAK_TESTS") == "1"


def test_async_worker_soak_harness() -> None:
    if not _should_run_soak():
        pytest.skip("Set RUN_SOAK_TESTS=1 to enable soak harness")

    config = SoakConfig(iterations=3, jobs_per_iteration=48)
    result = run_soak(lambda c: AsyncWorker(max_concurrency=c), seed=7, config=config)

    assert result.jobs_submitted > 0
    # Validate accounting matches the worker stats snapshot.
    total_outcomes = (
        result.jobs_completed
        + result.jobs_cancelled
        + result.jobs_failed
        + result.jobs_timed_out
    )
    assert total_outcomes == result.jobs_submitted

    # Ensure scenarios exercised as expected.
    assert result.jobs_cancelled > 0
    assert result.jobs_timed_out > 0
    assert result.jobs_failed > 0

