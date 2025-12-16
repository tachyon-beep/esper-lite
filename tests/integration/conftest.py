"""Integration test configuration.

Auto-marks all tests in this directory as integration tests.
These tests are excluded by default (run with: pytest -m integration).
"""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-mark all tests in integration directory."""
    for item in items:
        # Check if test is in integration directory
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
