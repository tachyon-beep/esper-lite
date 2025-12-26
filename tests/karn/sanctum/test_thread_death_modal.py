"""Tests for ThreadDeathModal widget."""
from textual.screen import ModalScreen


def test_thread_death_modal_is_modal():
    """ThreadDeathModal should be a ModalScreen."""
    from esper.karn.sanctum.widgets.thread_death_modal import ThreadDeathModal

    modal = ThreadDeathModal()
    assert isinstance(modal, ModalScreen)


def test_thread_death_modal_has_dismiss_binding():
    """ThreadDeathModal should be dismissable with Escape."""
    from esper.karn.sanctum.widgets.thread_death_modal import ThreadDeathModal

    modal = ThreadDeathModal()
    binding_keys = [b.key for b in modal.BINDINGS]
    assert "escape" in binding_keys
