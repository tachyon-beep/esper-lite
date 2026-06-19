"""TGV-001: All-payload telemetry contract round-trip coverage.

Evidence / motivation
---------------------
A missing test allowed ``EpochCompletedPayload.episode_idx`` to be declared and
required by ``from_dict()`` while being silently omitted by ``to_dict()`` (the
LN-001 bug). There was no test asserting that EVERY active telemetry payload
dataclass round-trips ALL of its non-default fields.

What this module does
---------------------
For every active telemetry payload dataclass that exposes a ``from_dict()``
classmethod (and the nested typed payloads they embed), it:

1. Constructs an instance with EVERY field set to a DISTINCTIVE, NON-DEFAULT
   value (Optional fields get a non-None value; each field gets a distinct
   value so a field-swap is also caught -- a field left at its default would
   pass even if the serializer dropped it).
2. Serializes via the path the payload actually uses on the wire:
   - ``payload.to_dict()`` when defined, else
   - the generic Karn serializer (``_payload_to_dict`` -> ``dataclasses.asdict``),
     which is the same path ``nissa.output`` / ``karn.serialization`` take for
     payloads without a hand-written ``to_dict``.
3. Reconstructs via ``cls.from_dict(serialized)`` and asserts the reconstructed
   instance equals the original on EVERY field.

Completeness guard
------------------
``test_round_trip_covers_every_payload_with_from_dict`` discovers payload
classes DYNAMICALLY from ``esper.leyline.telemetry`` (every dataclass with a
``from_dict``) and asserts the parametrized set covers all of them. Adding a new
payload dataclass with a ``from_dict`` that is not round-trip-safe therefore
FAILS this suite -- which is the entire point: prevent the next silent
field-drop.

History
-------
This suite originally caught two LN-001-class field drops:
  - ``AnalyticsSnapshotPayload.from_dict`` dropped 11 declared fields
    (reward-component breakdown + decision-card context + max_seeds/reward_mode).
  - ``PPOUpdatePayload.from_dict`` dropped the value-function quality fields
    (TELE-220..228) plus value_target_scale.
Both production ``from_dict`` methods were fixed to read every declared field,
so all cases below now assert a clean round-trip with no xfails.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timezone
from typing import Any

import pytest

from esper.karn.serialization import _payload_to_dict
from esper.leyline import telemetry as T
from esper.leyline.factored_actions import NUM_OPS


# ---------------------------------------------------------------------------
# Distinctive-value construction.
#
# Each field gets a value that is:
#   - non-default (so a dropped field is detectable),
#   - non-None for Optional fields,
#   - distinct from its neighbours (so a field-swap is detectable).
#
# Fields with validation constraints (enums, fixed-length tuples, Literals,
# nested dataclasses) are special-cased; everything else is derived from the
# field's type annotation.
# ---------------------------------------------------------------------------

_LITERAL_VALUES: dict[str, Any] = {
    "phase": "verdict",  # MorphologyCausalLogPhase
    "manifest_role": "static_final_replay",  # TopologyManifestRole
    "pattern": "ransomware_signature",  # RewardHackingPattern
    "panic_reason": "governor_nan",  # GovernorPanicReason
}


def _make_head_telemetry() -> "T.HeadTelemetry":
    fields = dataclasses.fields(T.HeadTelemetry)
    return T.HeadTelemetry(**{f.name: float(50 + i) for i, f in enumerate(fields)})


def _make_reward_components() -> Any:
    from esper.leyline.telemetry_contracts import RewardComponentsTelemetry

    # Distinctive, non-default values across a representative spread of fields.
    return RewardComponentsTelemetry(
        base_acc_delta=1.5,
        bounded_attribution=2.5,
        compute_rent=-3.5,
        stage_bonus=4.5,
        seed_stage=3,
        action_name="GERMINATE",
        epoch=12,
        total_reward=9.25,
    )


def _make_observation_stats() -> Any:
    from esper.leyline.telemetry_contracts import ObservationStatsTelemetry

    return ObservationStatsTelemetry(
        slot_features_mean=1.1,
        slot_features_std=2.2,
        host_features_mean=3.3,
        host_features_std=4.4,
        outlier_pct=0.25,
        nan_count=2,
        inf_count=3,
        batch_size=33,
    )


def _make_field_value(cls: type, f: "dataclasses.Field[Any]", idx: int) -> Any:
    """Return a distinctive, non-default value for one dataclass field."""
    name = f.name
    ann = str(f.type)

    # --- Class-specific validated fields -------------------------------------
    if cls is T.SeedTelemetry:
        if name == "stage":
            return 7  # valid SeedStage.value (skips the value-5 gap)
        if name == "alpha_mode":
            return 2  # valid AlphaMode.value
        if name == "alpha_algorithm":
            return 3  # valid AlphaAlgorithm.value
        if name == "captured_at":
            return datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    if cls is T.PPOUpdatePayload:
        if name == "op_q_values":
            return tuple(float(100 + idx + j) for j in range(NUM_OPS))
        if name == "op_valid_mask":
            # Distinct, mostly-True mask (must be length NUM_OPS).
            return tuple(j != 0 for j in range(NUM_OPS))

    # --- Structured fields with no generic annotation rule -------------------
    if cls is T.AnalyticsSnapshotPayload and name == "alternatives":
        # Declared list[tuple[str, float]] (top-2 (action, prob) pairs). JSON
        # has no tuple type, so from_dict re-tuples the inner pairs; this value
        # exercises that path.
        return [("WAIT", 0.6), ("GERMINATE", 0.3)]

    # --- Literals / discriminators -------------------------------------------
    if name in _LITERAL_VALUES:
        return _LITERAL_VALUES[name]

    # --- Nested typed payloads -----------------------------------------------
    if name == "head_telemetry":
        return _make_head_telemetry()
    if name == "reward_components":
        return _make_reward_components()
    if name == "observation_stats":
        return _make_observation_stats()

    # --- Generic, annotation-driven ------------------------------------------
    base = idx + 1

    # Order matters: check container types before scalar substrings.
    if "tuple[str" in ann:
        return (f"{name}_a", f"{name}_b")
    if "tuple[float" in ann:
        return (1.0 + base, 2.0 + base)
    if "tuple[bool" in ann:
        return (True, False)
    if "tuple[int" in ann:
        return (10 + base, 20 + base)
    if "tuple[dict" in ann:
        return ({f"{name}_k": base},)
    if "list[str]" in ann:
        return [f"{name}_x", f"{name}_y"]
    if "dict[int" in ann:
        return {base: f"{name}_v"}
    if "dict[str, dict" in ann:
        return {f"{name}_outer": {f"{name}_inner": float(base)}}
    if "dict[str, float]" in ann:
        return {f"{name}_k": float(base)}
    if "dict[str, int]" in ann:
        return {f"{name}_k": base}
    if "dict[str, bool]" in ann:
        return {f"{name}_k": True}
    if "dict" in ann:
        return {f"{name}_k": f"{name}_v_{base}"}

    # Scalars. bool must be checked before int (bool is a subclass of int).
    if "bool" in ann:
        return True
    if "str" in ann:
        return f"{name}_val_{base}"
    if "float" in ann:
        return 0.5 + base
    if "int" in ann:
        return 1000 + base

    raise AssertionError(
        f"No distinctive value rule for {cls.__name__}.{name}: {ann!r}. "
        "Add a rule to _make_field_value so this field is exercised."
    )


def _build_instance(cls: type) -> Any:
    kwargs: dict[str, Any] = {}
    for idx, f in enumerate(dataclasses.fields(cls)):
        kwargs[f.name] = _make_field_value(cls, f, idx)
    return cls(**kwargs)


def _serialize(inst: Any) -> dict[str, Any]:
    """Serialize via the payload's real wire path."""
    to_dict = getattr(inst, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    # Generic path used by nissa.output / karn.serialization for payloads
    # without a hand-written to_dict (dataclasses.asdict under the hood).
    return _payload_to_dict(inst)


def _values_equal(a: Any, b: Any) -> bool:
    """Equality that treats NaN==NaN as equal and list/tuple as interchangeable.

    JSON does not distinguish tuple from list; the from_dict() helpers normalize
    back to tuples where it matters, so we compare structurally.
    """
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------
# Dynamic discovery of payload classes with a from_dict classmethod.
# ---------------------------------------------------------------------------

def _discover_payload_classes() -> list[type]:
    classes: list[type] = []
    for attr in dir(T):
        obj = getattr(T, attr)
        if not isinstance(obj, type):
            continue
        if not dataclasses.is_dataclass(obj):
            continue
        if obj.__module__ != T.__name__:
            continue
        from_dict = getattr(obj, "from_dict", None)
        if callable(from_dict):
            classes.append(obj)
    return sorted(classes, key=lambda c: c.__name__)


PAYLOAD_CLASSES = _discover_payload_classes()
PAYLOAD_CLASS_NAMES = [c.__name__ for c in PAYLOAD_CLASSES]


@pytest.mark.parametrize("cls", PAYLOAD_CLASSES, ids=PAYLOAD_CLASS_NAMES)
def test_payload_round_trips_all_fields(cls: type) -> None:
    """from_dict(to_dict(instance)) preserves every non-default field."""
    original = _build_instance(cls)
    serialized = _serialize(original)
    reconstructed = cls.from_dict(serialized)

    dropped: list[str] = []
    for f in dataclasses.fields(cls):
        ov = getattr(original, f.name)
        rv = getattr(reconstructed, f.name)
        if not _values_equal(ov, rv):
            dropped.append(f.name)

    assert not dropped, (
        f"{cls.__name__} did NOT round-trip these declared fields "
        f"(to_dict emitted them but from_dict dropped or mangled them): {dropped}. "
        "This is an LN-001-class serialization bug: a declared field that does "
        "not survive to_dict -> from_dict. Fix the production serializer."
    )


def test_round_trip_covers_every_payload_with_from_dict() -> None:
    """Completeness guard: every telemetry dataclass with from_dict is covered.

    Discovered dynamically, so adding a NEW payload dataclass automatically
    enrolls it in the parametrized round-trip test above. This test exists to
    fail loudly if discovery ever silently misses a class.
    """
    discovered = {c.__name__ for c in _discover_payload_classes()}
    covered = set(PAYLOAD_CLASS_NAMES)
    assert discovered == covered, (
        "Payload discovery drifted from the parametrized set. "
        f"Discovered-but-not-covered: {sorted(discovered - covered)}; "
        f"Covered-but-not-discovered: {sorted(covered - discovered)}."
    )

    # Sanity: the well-known payloads the P2 burn-down touched MUST be present,
    # so a refactor that accidentally strips from_dict from one of them trips
    # here rather than silently shrinking coverage.
    must_have = {
        "EpochCompletedPayload",
        "HeadTelemetry",
        "BatchEpochCompletedPayload",
        "PPOUpdatePayload",
        "SeedPrunedPayload",
        "SeedFossilizedPayload",
        "SeedGerminatedPayload",
        "SeedStageChangedPayload",
        "EpisodeOutcomePayload",
        "GovernorRollbackPayload",
        "MorphologyCausalLogPayload",
        "TopologyManifestPayload",
        "AnomalyDetectedPayload",
        "TrainingStartedPayload",
    }
    missing = must_have - discovered
    assert not missing, f"Expected payload(s) missing a from_dict: {sorted(missing)}"


def test_every_union_payload_has_a_serialization_path() -> None:
    """Every member of the TelemetryPayload union must serialize losslessly.

    Most union members carry a from_dict (covered above). The lone exception is
    CheckpointLoadedPayload, which has no hand-written from_dict/to_dict and
    flows through the generic asdict serializer. Assert that generic path is
    lossless against its declared fields, so this union member is not a blind
    spot.
    """
    union_members = [
        T.TrainingStartedPayload,
        T.CheckpointLoadedPayload,
        T.EpochCompletedPayload,
        T.BatchEpochCompletedPayload,
        T.TrendDetectedPayload,
        T.PPOUpdatePayload,
        T.MemoryWarningPayload,
        T.RewardHackingSuspectedPayload,
        T.TamiyoInitiatedPayload,
        T.SeedGerminatedPayload,
        T.SeedStageChangedPayload,
        T.SeedGateEvaluatedPayload,
        T.SeedFossilizedPayload,
        T.SeedPrunedPayload,
        T.CounterfactualMatrixPayload,
        T.AnalyticsSnapshotPayload,
        T.AnomalyDetectedPayload,
        T.PerformanceDegradationPayload,
        T.EpisodeOutcomePayload,
        T.GovernorRollbackPayload,
        T.MorphologyCausalLogPayload,
        T.TopologyManifestPayload,
    ]
    for cls in union_members:
        if hasattr(cls, "from_dict"):
            continue  # covered by the parametrized round-trip
        # No from_dict: verify the generic serializer preserves every field
        # against a re-parse via dataclass field reconstruction.
        inst = _build_instance(cls)
        serialized = _serialize(inst)
        rebuilt = cls(**serialized)
        for f in dataclasses.fields(cls):
            assert _values_equal(getattr(inst, f.name), getattr(rebuilt, f.name)), (
                f"{cls.__name__}.{f.name} not preserved by the generic asdict "
                "serialization path."
            )
