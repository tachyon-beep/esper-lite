"""Pytest configuration for property-based tests."""

import os

import pytest
from hypothesis import settings, Verbosity

# Register profile for CI (more examples, deterministic)
settings.register_profile(
    "ci",
    max_examples=1000,
    verbosity=Verbosity.verbose,
    deadline=None,  # No timeout in CI
)

# Register profile for development (fewer examples, faster)
settings.register_profile(
    "dev",
    max_examples=100,
    verbosity=Verbosity.normal,
    deadline=5000,  # 5 second timeout
)

# Load profile from env or default to dev
# Usage: HYPOTHESIS_PROFILE=ci pytest tests/simic/properties/
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
