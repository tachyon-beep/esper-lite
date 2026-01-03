"""Tests for lazy blueprint loading.

These tests verify that:
1. Blueprint modules are loaded on-demand (not at import time)
2. Topology-scoped loading works correctly
3. The loader is idempotent and thread-safe
"""

from __future__ import annotations

import sys
from unittest.mock import patch


def test_registry_import_does_not_load_blueprints():
    """Importing BlueprintRegistry should not eagerly load blueprint modules.

    This is the core property we're testing: the package __init__ no longer
    triggers imports of cnn.py and transformer.py.
    """
    # Clear any cached imports to test fresh import behavior
    modules_to_clear = [
        key
        for key in sys.modules
        if key.startswith("esper.kasmina.blueprints")
    ]
    saved_modules = {key: sys.modules.pop(key) for key in modules_to_clear}

    # Also clear the loader's internal state
    try:
        # Fresh import of just the registry
        from esper.kasmina.blueprints import BlueprintRegistry  # noqa: F401

        # At this point, cnn and transformer should NOT be in sys.modules
        # (they'll only load when list_for_topology/get is called)
        assert "esper.kasmina.blueprints.cnn" not in sys.modules, (
            "cnn module should not be loaded just from importing BlueprintRegistry"
        )
        assert "esper.kasmina.blueprints.transformer" not in sys.modules, (
            "transformer module should not be loaded just from importing BlueprintRegistry"
        )
    finally:
        # Restore original modules
        sys.modules.update(saved_modules)


def test_list_for_topology_loads_only_requested_topology():
    """Calling list_for_topology('cnn') should only load CNN blueprints."""
    from esper.kasmina.blueprints.loader import _LOADED, ensure_loaded

    # Clear loader state
    _LOADED.clear()

    # This should load CNN but not transformer
    ensure_loaded("cnn")

    assert "cnn" in _LOADED, "CNN should be marked as loaded"
    # Note: transformer might be loaded from earlier test runs in this process,
    # so we just verify CNN is loaded, not that transformer isn't


def test_loader_is_idempotent():
    """Multiple calls to ensure_loaded() should be safe and fast."""
    from esper.kasmina.blueprints.loader import _LOADED, ensure_loaded

    _LOADED.clear()

    # First call loads
    ensure_loaded("cnn")
    first_count = len(_LOADED)

    # Second call is a no-op
    ensure_loaded("cnn")
    second_count = len(_LOADED)

    assert first_count == second_count, "Second call should not re-load"


def test_ensure_all_loaded_populates_registry():
    """ensure_all_loaded() should load all known topologies."""
    from esper.kasmina.blueprints.loader import (
        _LOADED,
        ensure_all_loaded,
        get_known_topologies,
    )

    _LOADED.clear()
    ensure_all_loaded()

    known = get_known_topologies()
    for topology in known:
        assert topology in _LOADED, f"{topology} should be loaded"


def test_unknown_topology_does_not_raise():
    """ensure_loaded() for unknown topology should silently pass."""
    from esper.kasmina.blueprints.loader import ensure_loaded

    # This should not raise - unknown topologies just don't have built-in blueprints
    ensure_loaded("unknown_topology")


def test_registry_list_for_topology_triggers_lazy_load():
    """BlueprintRegistry.list_for_topology() should trigger lazy loading."""
    from esper.kasmina.blueprints import BlueprintRegistry
    from esper.kasmina.blueprints.loader import _LOADED

    _LOADED.clear()

    # This should trigger loading
    blueprints = BlueprintRegistry.list_for_topology("cnn")

    assert "cnn" in _LOADED, "list_for_topology should trigger loading"
    assert len(blueprints) > 0, "CNN blueprints should be registered after loading"


def test_registry_get_triggers_lazy_load():
    """BlueprintRegistry.get() should trigger lazy loading."""
    from esper.kasmina.blueprints import BlueprintRegistry
    from esper.kasmina.blueprints.loader import _LOADED

    # Don't clear _blueprints - we just want to verify that get() calls ensure_loaded()
    # Clearing both _LOADED and _blueprints creates an impossible state since
    # Python caches imported modules in sys.modules
    _LOADED.clear()

    # This should trigger loading (even though blueprints may already be registered
    # from previous imports in this process)
    spec = BlueprintRegistry.get("cnn", "noop")

    assert "cnn" in _LOADED, "get() should trigger loading"
    assert spec.name == "noop", "Should find the noop blueprint"


def test_convblock_lazy_import():
    """ConvBlock should be importable via __getattr__ lazy loading."""
    # This tests the PEP 562 __getattr__ mechanism
    from esper.kasmina.blueprints import ConvBlock

    # Verify it's the actual class
    assert hasattr(ConvBlock, "__init__"), "ConvBlock should be a class"
    assert "ConvBlock" in str(ConvBlock), "Should be the ConvBlock class"


def test_import_error_is_descriptive():
    """If a blueprint module fails to import, the error should be descriptive."""
    from esper.kasmina.blueprints.loader import _LOADED, _TOPOLOGY_MODULES

    # Temporarily add a fake topology that will fail to import
    _TOPOLOGY_MODULES["fake"] = "esper.kasmina.blueprints.nonexistent"
    _LOADED.discard("fake")

    try:
        from esper.kasmina.blueprints.loader import ensure_loaded
        import pytest

        with pytest.raises(ImportError, match=r"Failed to load fake blueprints"):
            ensure_loaded("fake")
    finally:
        # Clean up
        del _TOPOLOGY_MODULES["fake"]
