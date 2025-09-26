from esper.core.dependency_guard import (
    DependencyContext,
    DependencyViolationError,
    ensure_present,
    verify_registry_entry,
)


def test_ensure_present_passes_when_condition_true() -> None:
    ensure_present(True, DependencyContext("kasmina", "blueprint"), reason="ok")


def test_ensure_present_raises_with_context() -> None:
    ctx = DependencyContext("tamiyo", "policy", identifier="policy-1", details={"hint": "missing"})
    try:
        ensure_present(False, ctx, reason="missing policy")
    except DependencyViolationError as exc:
        assert exc.subsystem == "tamiyo"
        assert "missing policy" in str(exc)
        assert exc.context["hint"] == "missing"
    else:
        raise AssertionError("Violation was not raised")


def test_verify_registry_entry_returns_value() -> None:
    registry = {"bp-1": object()}

    def lookup(identifier: str):
        return registry.get(identifier)

    value = verify_registry_entry(
        "bp-1",
        lookup,
        DependencyContext("tolaria", "blueprint"),
    )
    assert value is registry["bp-1"]


def test_verify_registry_entry_raises_for_missing() -> None:
    def lookup(identifier: str):
        return None

    try:
        verify_registry_entry(
            "missing",
            lookup,
            DependencyContext("kasmina", "blueprint"),
        )
    except DependencyViolationError as exc:
        assert exc.subsystem == "kasmina"
        assert "missing blueprint missing" in str(exc)
    else:
        raise AssertionError("Violation not raised")
