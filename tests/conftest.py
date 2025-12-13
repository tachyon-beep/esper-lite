"""Shared pytest fixtures and configuration.

This file is automatically loaded by pytest and provides fixtures
accessible to all tests.
"""

import pytest
import json
import torch
from pathlib import Path
from hypothesis import settings, HealthCheck

# =============================================================================
# Hypothesis Configuration
# =============================================================================

# Define profiles for different environments
settings.register_profile(
    "ci",
    max_examples=50,  # Faster for CI
    deadline=None,  # No deadlines for slow tests
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "dev",
    max_examples=10,  # Very fast for local development
    deadline=500,  # 500ms deadline for local tests
)

settings.register_profile(
    "thorough",
    max_examples=1000,  # Comprehensive for nightly runs
    deadline=None,
)

# Load profile based on environment variable
import os
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))

# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def snapshots_dir(fixtures_dir):
    """Return path to snapshots fixtures."""
    return fixtures_dir / "snapshots"


@pytest.fixture(scope="session")
def telemetry_dir(fixtures_dir):
    """Return path to telemetry fixtures."""
    return fixtures_dir / "telemetry"


# =============================================================================
# JSON Fixture Loaders
# =============================================================================

@pytest.fixture
def healthy_gradients_telemetry(telemetry_dir):
    """Load healthy gradients telemetry."""
    with open(telemetry_dir / "healthy_gradients.json") as f:
        data = json.load(f)
    from esper.leyline import SeedTelemetry
    return SeedTelemetry(**data)


# =============================================================================
# Model Fixtures (Generated On-Demand)
# =============================================================================

@pytest.fixture(scope="session")
def small_ppo_model_deterministic(tmp_path_factory):
    """Create a PPO model with deterministic weights (no training required).

    This fixture creates a valid PPO agent with fixed, deterministic weights
    instead of training a model. Benefits:
    - No training time (instant setup)
    - Deterministic (no flaky tests from training variance)
    - If PPO breaks, test setup doesn't crash (fail fast at the right place)

    Use this for testing the "plumbing" (loading, inference, etc.), not
    for testing that the model is "smart."
    """
    from esper.simic.ppo import PPOAgent

    # Create agent with deterministic weights
    agent = PPOAgent(state_dim=30, action_dim=7)

    # Initialize all weights deterministically
    for param in agent.parameters():
        if param.dim() >= 2:
            # Weights: constant initialization
            torch.nn.init.constant_(param, 0.1)
        else:
            # Biases: zeros
            torch.nn.init.zeros_(param)

    # Note: This agent is "dumb" (not trained), but it's valid for testing
    # model loading, inference, feature extraction, etc.
    return agent

@pytest.fixture(scope="session")
def small_ppo_model_checkpoint(tmp_path_factory, small_ppo_model_deterministic):
    """Save deterministic PPO model to checkpoint file.

    Use this to test checkpoint loading/saving without training overhead.
    """
    cache_dir = tmp_path_factory.mktemp("models")
    checkpoint_path = cache_dir / "small_ppo_deterministic.pt"

    # Save deterministic model
    torch.save(small_ppo_model_deterministic.state_dict(), checkpoint_path)

    return checkpoint_path


# =============================================================================
# Temporary Workspace Fixtures
# =============================================================================

@pytest.fixture
def temp_workspace(tmp_path):
    """Provide isolated temporary directory for test.

    Automatically cleaned up after test.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


# =============================================================================
# Seed-Based RNG for Reproducibility
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility.

    This fixture runs automatically for all tests.
    """
    import random
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    yield  # Test runs here

    # Cleanup after test (if needed)
