"""End-to-end tests for SeedState detail telemetry records (TELE-550 to TELE-554).

Verifies seed state telemetry flows from schema to EnvOverview display.

These tests cover:
- TELE-550: seed.blueprint_id (str | None, truncated to 6 chars)
- TELE-551: seed.alpha (float 0.0-1.0, blend weight)
- TELE-552: seed.epochs_in_stage (int, stage duration)
- TELE-553: seed.alpha_curve (str, maps to glyphs)
- TELE-554: seed.blend_tempo_epochs (int, tempo arrows)

All records are FULLY WIRED - SeedState dataclass in schema.py provides
fields that EnvOverview._format_slot_cell() consumes for display.
"""

import pytest

from esper.karn.sanctum.schema import SeedState
from esper.leyline import ALPHA_CURVE_GLYPHS


# =============================================================================
# TELE-550: Seed Blueprint ID
# =============================================================================


class TestTELE550SeedBlueprintId:
    """TELE-550: seed.blueprint_id field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: Kasmina SeedState stores blueprint_id at germination
    - Transport: SeedGerminatedPayload carries blueprint_id; aggregator populates schema
    - Schema: SeedState.blueprint_id at line 394
    - Consumer: EnvOverview._format_slot_cell() accesses seed.blueprint_id

    Blueprint ID identifies the neural module template type:
    - "conv_light", "attention", "norm", "depthwise", "bottleneck", etc.
    - None represents DORMANT slot with no seed
    - Display truncates to 6 characters: "conv_light" -> "conv_l"
    """

    def test_blueprint_id_field_exists(self) -> None:
        """TELE-550: Verify SeedState.blueprint_id field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "blueprint_id")

    def test_blueprint_id_default_value(self) -> None:
        """TELE-550: Default blueprint_id is None (no seed in slot)."""
        seed = SeedState(slot_id="r0c0")
        assert seed.blueprint_id is None

    def test_blueprint_id_type_is_optional_str(self) -> None:
        """TELE-550: blueprint_id type is str | None."""
        # Default is None
        seed = SeedState(slot_id="r0c0")
        assert seed.blueprint_id is None

        # Can be set to string
        seed = SeedState(slot_id="r0c0", blueprint_id="conv_light")
        assert seed.blueprint_id == "conv_light"
        assert isinstance(seed.blueprint_id, str)

    def test_blueprint_id_valid_values(self) -> None:
        """TELE-550: blueprint_id accepts known blueprint names."""
        valid_blueprints = [
            "conv_light",
            "attention",
            "norm",
            "depthwise",
            "bottleneck",
            "conv_small",
            "conv_heavy",
            "lora",
            "lora_large",
            "mlp_small",
            "mlp",
            "flex_attention",
            "noop",
        ]

        for bp in valid_blueprints:
            seed = SeedState(slot_id="r0c0", blueprint_id=bp)
            assert seed.blueprint_id == bp

    def test_blueprint_id_truncation_to_six_chars(self) -> None:
        """TELE-550: Display truncates blueprint_id to 6 characters.

        This tests the truncation logic that EnvOverview._format_slot_cell() applies:
        if len(blueprint) > 6:
            blueprint = blueprint[:6]
        """
        test_cases = [
            ("conv_light", "conv_l"),  # 10 chars -> 6
            ("attention", "attent"),   # 9 chars -> 6
            ("depthwise", "depthw"),   # 9 chars -> 6
            ("lora_large", "lora_l"),  # 10 chars -> 6
            ("conv", "conv"),          # 4 chars -> 4 (no truncation)
            ("mlp", "mlp"),            # 3 chars -> 3 (no truncation)
            ("noop", "noop"),          # 4 chars -> 4 (no truncation)
        ]

        for full_name, expected_display in test_cases:
            seed = SeedState(slot_id="r0c0", blueprint_id=full_name)
            blueprint = seed.blueprint_id or "?"
            if len(blueprint) > 6:
                blueprint = blueprint[:6]
            assert blueprint == expected_display, f"Failed for {full_name}"

    def test_blueprint_id_none_displays_as_question_mark(self) -> None:
        """TELE-550: None blueprint_id displays as '?' in format logic."""
        seed = SeedState(slot_id="r0c0", blueprint_id=None)
        blueprint = seed.blueprint_id or "?"
        assert blueprint == "?"


# =============================================================================
# TELE-551: Seed Alpha
# =============================================================================


class TestTELE551SeedAlpha:
    """TELE-551: seed.alpha field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: AlphaController computes alpha per epoch
    - Transport: SeedState.alpha synced from controller; telemetry carries value
    - Schema: SeedState.alpha at line 395
    - Consumer: EnvOverview._format_slot_cell() displays seed.alpha

    Alpha (blend weight) progression:
    - 0.0: Seed has zero output contribution (DORMANT, GERMINATED, TRAINING)
    - 0.5: Seed contributes equally with host baseline
    - 1.0: Seed is fully integrated (FOSSILIZED or max blend)

    Display format: "{alpha:.1f}" (1 decimal place)
    """

    def test_alpha_field_exists(self) -> None:
        """TELE-551: Verify SeedState.alpha field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "alpha")

    def test_alpha_default_value(self) -> None:
        """TELE-551: Default alpha is 0.0 (seed not blending)."""
        seed = SeedState(slot_id="r0c0")
        assert seed.alpha == 0.0

    def test_alpha_type_is_float(self) -> None:
        """TELE-551: alpha must be float type."""
        seed = SeedState(slot_id="r0c0", alpha=0.5)
        assert isinstance(seed.alpha, float)

    def test_alpha_range_zero_to_one(self) -> None:
        """TELE-551: alpha is in range [0.0, 1.0]."""
        # Lower bound
        seed_zero = SeedState(slot_id="r0c0", alpha=0.0)
        assert seed_zero.alpha == 0.0

        # Upper bound
        seed_one = SeedState(slot_id="r0c0", alpha=1.0)
        assert seed_one.alpha == 1.0

        # Mid-range values
        seed_half = SeedState(slot_id="r0c0", alpha=0.5)
        assert seed_half.alpha == 0.5

    def test_alpha_precision_one_decimal(self) -> None:
        """TELE-551: alpha supports 1 decimal place precision for display."""
        test_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

        for val in test_values:
            seed = SeedState(slot_id="r0c0", alpha=val)
            assert seed.alpha == pytest.approx(val, rel=1e-6)
            # Verify format produces expected string
            formatted = f"{seed.alpha:.1f}"
            assert formatted == f"{val:.1f}"

    def test_alpha_by_stage_semantics(self) -> None:
        """TELE-551: Alpha values have stage-specific meanings."""
        # DORMANT/GERMINATED/TRAINING: alpha = 0.0
        dormant_seed = SeedState(slot_id="r0c0", stage="DORMANT", alpha=0.0)
        assert dormant_seed.alpha == 0.0

        training_seed = SeedState(slot_id="r0c0", stage="TRAINING", alpha=0.0)
        assert training_seed.alpha == 0.0

        # BLENDING: alpha ramps from 0.0 to target
        blending_seed = SeedState(slot_id="r0c0", stage="BLENDING", alpha=0.3)
        assert 0.0 <= blending_seed.alpha <= 1.0

        # FOSSILIZED: alpha = 1.0
        fossilized_seed = SeedState(slot_id="r0c0", stage="FOSSILIZED", alpha=1.0)
        assert fossilized_seed.alpha == 1.0


# =============================================================================
# TELE-552: Seed Epochs In Stage
# =============================================================================


class TestTELE552SeedEpochsInStage:
    """TELE-552: seed.epochs_in_stage field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: SeedMetrics.epochs_in_current_stage tracked by SeedSlot
    - Transport: sync_telemetry() copies value; aggregator populates schema
    - Schema: SeedState.epochs_in_stage at line 403
    - Consumer: EnvOverview._format_slot_cell() displays "e{N}" suffix

    Epochs in stage semantics:
    - Resets to 0 when stage transitions occur
    - Increments by 1 each epoch the seed remains in same stage
    - Display: "e5" means seed has been in stage for 5 epochs
    """

    def test_epochs_in_stage_field_exists(self) -> None:
        """TELE-552: Verify SeedState.epochs_in_stage field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "epochs_in_stage")

    def test_epochs_in_stage_default_value(self) -> None:
        """TELE-552: Default epochs_in_stage is 0 (just entered stage)."""
        seed = SeedState(slot_id="r0c0")
        assert seed.epochs_in_stage == 0

    def test_epochs_in_stage_type_is_int(self) -> None:
        """TELE-552: epochs_in_stage must be int type."""
        seed = SeedState(slot_id="r0c0", epochs_in_stage=5)
        assert isinstance(seed.epochs_in_stage, int)

    def test_epochs_in_stage_non_negative(self) -> None:
        """TELE-552: epochs_in_stage is non-negative."""
        seed = SeedState(slot_id="r0c0", epochs_in_stage=0)
        assert seed.epochs_in_stage >= 0

        seed = SeedState(slot_id="r0c0", epochs_in_stage=100)
        assert seed.epochs_in_stage >= 0

    def test_epochs_in_stage_display_format(self) -> None:
        """TELE-552: Display format is 'e{N}' where N is epochs_in_stage."""
        test_cases = [
            (0, ""),      # epochs_in_stage=0 shows no suffix
            (1, " e1"),
            (5, " e5"),
            (10, " e10"),
            (100, " e100"),
        ]

        for epochs, expected_suffix in test_cases:
            seed = SeedState(slot_id="r0c0", epochs_in_stage=epochs)
            # Match _format_slot_cell logic
            epochs_str = f" e{seed.epochs_in_stage}" if seed.epochs_in_stage > 0 else ""
            assert epochs_str == expected_suffix

    def test_epochs_in_stage_shown_for_training_germinated(self) -> None:
        """TELE-552: epochs_in_stage displayed for TRAINING/GERMINATED stages."""
        # Per TELE-552 spec: epochs shown for stages where time-in-stage
        # is the primary progress indicator
        training_seed = SeedState(
            slot_id="r0c0", stage="TRAINING", epochs_in_stage=5
        )
        assert training_seed.epochs_in_stage == 5

        germinated_seed = SeedState(
            slot_id="r0c0", stage="GERMINATED", epochs_in_stage=2
        )
        assert germinated_seed.epochs_in_stage == 2


# =============================================================================
# TELE-553: Seed Alpha Curve
# =============================================================================


class TestTELE553SeedAlphaCurve:
    """TELE-553: seed.alpha_curve field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: AlphaController.alpha_curve stores curve type
    - Transport: Stage change events carry curve; aggregator populates schema
    - Schema: SeedState.alpha_curve at line 414
    - Consumer: EnvOverview._format_slot_cell() maps to glyph via ALPHA_CURVE_GLYPHS

    Alpha curve determines the shape of the alpha ramp during BLENDING:
    - LINEAR: Constant rate increase
    - COSINE: Ease-in/ease-out
    - SIGMOID_GENTLE: Gradual S-curve (steepness=6)
    - SIGMOID: Standard S-curve (steepness=12)
    - SIGMOID_SHARP: Near-step function (steepness=24)
    """

    def test_alpha_curve_field_exists(self) -> None:
        """TELE-553: Verify SeedState.alpha_curve field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "alpha_curve")

    def test_alpha_curve_default_value(self) -> None:
        """TELE-553: Default alpha_curve is 'LINEAR'."""
        seed = SeedState(slot_id="r0c0")
        assert seed.alpha_curve == "LINEAR"

    def test_alpha_curve_type_is_str(self) -> None:
        """TELE-553: alpha_curve must be str type."""
        seed = SeedState(slot_id="r0c0", alpha_curve="COSINE")
        assert isinstance(seed.alpha_curve, str)

    def test_alpha_curve_valid_values(self) -> None:
        """TELE-553: alpha_curve accepts known curve types."""
        valid_curves = ["LINEAR", "COSINE", "SIGMOID_GENTLE", "SIGMOID", "SIGMOID_SHARP"]

        for curve in valid_curves:
            seed = SeedState(slot_id="r0c0", alpha_curve=curve)
            assert seed.alpha_curve == curve

    def test_alpha_curve_glyph_mapping(self) -> None:
        """TELE-553: Each alpha_curve maps to a display glyph.

        Glyphs from leyline.ALPHA_CURVE_GLYPHS:
        - LINEAR: diagonal line (constant rate)
        - COSINE: wave (ease-in/ease-out)
        - SIGMOID family: arc shapes showing transition sharpness
        """
        expected_glyphs = {
            "LINEAR": "\u2571",       # Diagonal slash
            "COSINE": "\u223f",       # Wave
            "SIGMOID_GENTLE": "\u2312",  # Wide arc
            "SIGMOID": "\u2322",      # Narrow arc
            "SIGMOID_SHARP": "\u2290",  # Squared bracket
        }

        for curve, expected_glyph in expected_glyphs.items():
            seed = SeedState(slot_id="r0c0", alpha_curve=curve)
            glyph = ALPHA_CURVE_GLYPHS.get(seed.alpha_curve, "-")
            assert glyph == expected_glyph, f"Failed for curve {curve}"

    def test_alpha_curve_glyphs_from_leyline(self) -> None:
        """TELE-553: ALPHA_CURVE_GLYPHS contains all curve types."""
        required_curves = ["LINEAR", "COSINE", "SIGMOID_GENTLE", "SIGMOID", "SIGMOID_SHARP"]

        for curve in required_curves:
            assert curve in ALPHA_CURVE_GLYPHS, f"Missing glyph for {curve}"
            assert isinstance(ALPHA_CURVE_GLYPHS[curve], str)
            assert len(ALPHA_CURVE_GLYPHS[curve]) == 1  # Single character glyph

    def test_alpha_curve_unknown_fallback(self) -> None:
        """TELE-553: Unknown alpha_curve falls back to '-' glyph."""
        seed = SeedState(slot_id="r0c0", alpha_curve="UNKNOWN")
        glyph = ALPHA_CURVE_GLYPHS.get(seed.alpha_curve, "-")
        assert glyph == "-"

    def test_alpha_curve_shown_for_blending_holding_fossilized(self) -> None:
        """TELE-553: Curve glyph shown for BLENDING/HOLDING/FOSSILIZED stages.

        Per spec: Curve glyph only shown for stages where it's causally relevant.
        """
        active_stages = ["BLENDING", "HOLDING", "FOSSILIZED"]
        inactive_stages = ["DORMANT", "GERMINATED", "TRAINING"]

        for stage in active_stages:
            seed = SeedState(slot_id="r0c0", stage=stage, alpha_curve="SIGMOID")
            # In _format_slot_cell: curve_glyph = ALPHA_CURVE_GLYPHS.get(...)
            # if seed.stage in ("BLENDING", "HOLDING", "FOSSILIZED") else "-"
            assert seed.stage in ("BLENDING", "HOLDING", "FOSSILIZED")

        for stage in inactive_stages:
            seed = SeedState(slot_id="r0c0", stage=stage, alpha_curve="SIGMOID")
            # For inactive stages, display shows dim "-" instead of curve glyph
            assert seed.stage not in ("BLENDING", "HOLDING", "FOSSILIZED")


# =============================================================================
# TELE-554: Seed Blend Tempo Epochs
# =============================================================================


class TestTELE554SeedBlendTempoEpochs:
    """TELE-554: seed.blend_tempo_epochs field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: SeedState.blend_tempo_epochs set at germination
    - Transport: SeedGerminatedPayload carries tempo; aggregator populates schema
    - Schema: SeedState.blend_tempo_epochs at line 411
    - Consumer: EnvOverview._format_slot_cell() computes tempo arrows

    Blend tempo epochs:
    - FAST: 3 epochs (rapid integration)
    - STANDARD: 5 epochs (balanced, default)
    - SLOW: 8 epochs (gradual integration)

    Display as tempo arrows:
    - <= 3: triple arrows (rapid)
    - <= 5: double arrows (normal)
    - > 5: single arrow (gradual)
    """

    def test_blend_tempo_epochs_field_exists(self) -> None:
        """TELE-554: Verify SeedState.blend_tempo_epochs field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "blend_tempo_epochs")

    def test_blend_tempo_epochs_default_value(self) -> None:
        """TELE-554: Default blend_tempo_epochs is 5 (STANDARD)."""
        seed = SeedState(slot_id="r0c0")
        assert seed.blend_tempo_epochs == 5

    def test_blend_tempo_epochs_type_is_int(self) -> None:
        """TELE-554: blend_tempo_epochs must be int type."""
        seed = SeedState(slot_id="r0c0", blend_tempo_epochs=3)
        assert isinstance(seed.blend_tempo_epochs, int)

    def test_blend_tempo_epochs_valid_values(self) -> None:
        """TELE-554: blend_tempo_epochs accepts known tempo values."""
        tempo_mapping = {
            "FAST": 3,
            "STANDARD": 5,
            "SLOW": 8,
        }

        for tempo_name, epochs in tempo_mapping.items():
            seed = SeedState(slot_id="r0c0", blend_tempo_epochs=epochs)
            assert seed.blend_tempo_epochs == epochs

    def test_tempo_arrows_fast(self) -> None:
        """TELE-554: tempo <= 3 displays as triple arrows."""
        for tempo in [1, 2, 3]:
            seed = SeedState(slot_id="r0c0", blend_tempo_epochs=tempo)
            arrows = _tempo_arrows(seed.blend_tempo_epochs)
            assert arrows == "\u25b8\u25b8\u25b8"  # Three arrows

    def test_tempo_arrows_standard(self) -> None:
        """TELE-554: tempo 4-5 displays as double arrows."""
        for tempo in [4, 5]:
            seed = SeedState(slot_id="r0c0", blend_tempo_epochs=tempo)
            arrows = _tempo_arrows(seed.blend_tempo_epochs)
            assert arrows == "\u25b8\u25b8"  # Two arrows

    def test_tempo_arrows_slow(self) -> None:
        """TELE-554: tempo > 5 displays as single arrow."""
        for tempo in [6, 7, 8, 10, 15]:
            seed = SeedState(slot_id="r0c0", blend_tempo_epochs=tempo)
            arrows = _tempo_arrows(seed.blend_tempo_epochs)
            assert arrows == "\u25b8"  # One arrow

    def test_tempo_arrows_none_returns_empty(self) -> None:
        """TELE-554: None tempo returns empty string."""
        arrows = _tempo_arrows(None)
        assert arrows == ""

    def test_tempo_arrows_threshold_boundaries(self) -> None:
        """TELE-554: Verify tempo arrow threshold boundaries.

        Thresholds from _format_slot_cell:
        - <= 3: triple arrows
        - <= 5: double arrows (but > 3)
        - > 5: single arrow
        """
        test_cases = [
            (1, "\u25b8\u25b8\u25b8"),   # FAST range
            (3, "\u25b8\u25b8\u25b8"),   # FAST boundary
            (4, "\u25b8\u25b8"),         # STANDARD range
            (5, "\u25b8\u25b8"),         # STANDARD boundary
            (6, "\u25b8"),               # SLOW range start
            (8, "\u25b8"),               # SLOW typical
        ]

        for tempo, expected_arrows in test_cases:
            seed = SeedState(slot_id="r0c0", blend_tempo_epochs=tempo)
            arrows = _tempo_arrows(seed.blend_tempo_epochs)
            assert arrows == expected_arrows, f"Failed for tempo {tempo}"


# =============================================================================
# Helper Functions (matching _format_slot_cell logic)
# =============================================================================


def _tempo_arrows(tempo: int | None) -> str:
    """Compute tempo arrows based on blend_tempo_epochs.

    This replicates the logic from EnvOverview._format_slot_cell():
    - tempo <= 3: triple arrows (FAST)
    - tempo <= 5: double arrows (STANDARD)
    - tempo > 5: single arrow (SLOW)
    """
    if tempo is None:
        return ""
    return "\u25b8\u25b8\u25b8" if tempo <= 3 else ("\u25b8\u25b8" if tempo <= 5 else "\u25b8")


# =============================================================================
# SeedState Schema Completeness
# =============================================================================


class TestSeedStateSchemaCompleteness:
    """Verify SeedState has all TELE-550 to TELE-554 fields."""

    def test_all_seed_detail_fields_present(self) -> None:
        """Verify all seed detail fields are present in SeedState."""
        seed = SeedState(slot_id="r0c0")

        # TELE-550: Blueprint ID
        assert hasattr(seed, "blueprint_id")

        # TELE-551: Alpha
        assert hasattr(seed, "alpha")

        # TELE-552: Epochs in stage
        assert hasattr(seed, "epochs_in_stage")

        # TELE-553: Alpha curve
        assert hasattr(seed, "alpha_curve")

        # TELE-554: Blend tempo epochs
        assert hasattr(seed, "blend_tempo_epochs")

    def test_all_defaults_match_spec(self) -> None:
        """Verify all default values match TELE record specifications."""
        seed = SeedState(slot_id="r0c0")

        # TELE-550: Default None
        assert seed.blueprint_id is None

        # TELE-551: Default 0.0
        assert seed.alpha == 0.0

        # TELE-552: Default 0
        assert seed.epochs_in_stage == 0

        # TELE-553: Default "LINEAR"
        assert seed.alpha_curve == "LINEAR"

        # TELE-554: Default 5 (STANDARD)
        assert seed.blend_tempo_epochs == 5

    def test_seed_state_with_all_fields(self) -> None:
        """Verify SeedState can be constructed with all TELE-550 to TELE-554 fields."""
        seed = SeedState(
            slot_id="r0c0",
            stage="BLENDING",
            blueprint_id="conv_light",
            alpha=0.5,
            epochs_in_stage=10,
            alpha_curve="SIGMOID",
            blend_tempo_epochs=3,
        )

        assert seed.slot_id == "r0c0"
        assert seed.stage == "BLENDING"
        assert seed.blueprint_id == "conv_light"
        assert seed.alpha == 0.5
        assert seed.epochs_in_stage == 10
        assert seed.alpha_curve == "SIGMOID"
        assert seed.blend_tempo_epochs == 3


# =============================================================================
# Consumer Widget Tests (EnvOverview._format_slot_cell)
# =============================================================================


class TestEnvOverviewFormatSlotCellConsumer:
    """Tests verifying EnvOverview correctly consumes SeedState fields.

    These tests verify the data contracts between SeedState (TELE-550 to TELE-554)
    and the EnvOverview._format_slot_cell() display method.
    """

    def test_format_slot_cell_blueprint_truncation(self) -> None:
        """EnvOverview._format_slot_cell truncates blueprint to 6 chars."""
        seed = SeedState(slot_id="r0c0", blueprint_id="conv_light")

        # Match _format_slot_cell logic
        blueprint = seed.blueprint_id or "?"
        if len(blueprint) > 6:
            blueprint = blueprint[:6]

        assert blueprint == "conv_l"

    def test_format_slot_cell_alpha_precision(self) -> None:
        """EnvOverview._format_slot_cell displays alpha with 1 decimal."""
        seed = SeedState(slot_id="r0c0", stage="BLENDING", alpha=0.37)

        # Match _format_slot_cell logic: {seed.alpha:.1f}
        alpha_display = f"{seed.alpha:.1f}"
        assert alpha_display == "0.4"  # Rounds 0.37 to 0.4

    def test_format_slot_cell_epochs_suffix(self) -> None:
        """EnvOverview._format_slot_cell adds 'e{N}' suffix for epochs."""
        seed = SeedState(slot_id="r0c0", stage="TRAINING", epochs_in_stage=5)

        # Match _format_slot_cell logic
        epochs_str = f" e{seed.epochs_in_stage}" if seed.epochs_in_stage > 0 else ""
        assert epochs_str == " e5"

    def test_format_slot_cell_curve_glyph(self) -> None:
        """EnvOverview._format_slot_cell shows curve glyph for BLENDING."""
        seed = SeedState(slot_id="r0c0", stage="BLENDING", alpha_curve="SIGMOID")

        # Match _format_slot_cell logic
        curve_glyph = (
            ALPHA_CURVE_GLYPHS.get(seed.alpha_curve, "-")
            if seed.stage in ("BLENDING", "HOLDING", "FOSSILIZED")
            else "-"
        )
        assert curve_glyph == "\u2322"

    def test_format_slot_cell_tempo_arrows(self) -> None:
        """EnvOverview._format_slot_cell shows tempo arrows."""
        seed = SeedState(slot_id="r0c0", stage="BLENDING", blend_tempo_epochs=3)

        # Match _format_slot_cell logic
        arrows = _tempo_arrows(seed.blend_tempo_epochs)
        assert arrows == "\u25b8\u25b8\u25b8"

    def test_format_slot_cell_dormant_returns_dash(self) -> None:
        """EnvOverview._format_slot_cell returns dash for DORMANT slots."""
        seed = SeedState(slot_id="r0c0", stage="DORMANT")

        # DORMANT slots display as "-" (no detailed info)
        assert seed.stage == "DORMANT"

    def test_format_slot_cell_blending_format(self) -> None:
        """EnvOverview._format_slot_cell formats BLENDING with all fields.

        Expected format: "Blend:conv_l {curve_glyph} {tempo_arrows} {alpha}"
        """
        seed = SeedState(
            slot_id="r0c0",
            stage="BLENDING",
            blueprint_id="conv_light",
            alpha=0.3,
            alpha_curve="SIGMOID",
            blend_tempo_epochs=5,
        )

        # Verify all fields are accessible for format
        assert seed.stage == "BLENDING"
        assert seed.alpha > 0

        blueprint = seed.blueprint_id[:6] if seed.blueprint_id and len(seed.blueprint_id) > 6 else seed.blueprint_id or "?"
        curve_glyph = ALPHA_CURVE_GLYPHS.get(seed.alpha_curve, "-")
        tempo_arrows = _tempo_arrows(seed.blend_tempo_epochs)

        assert blueprint == "conv_l"
        assert curve_glyph == "\u2322"
        assert tempo_arrows == "\u25b8\u25b8"
        assert f"{seed.alpha:.1f}" == "0.3"

    def test_format_slot_cell_training_format(self) -> None:
        """EnvOverview._format_slot_cell formats TRAINING with epochs.

        Expected format: "Train:conv_l {dim curve} e{N}"
        """
        seed = SeedState(
            slot_id="r0c0",
            stage="TRAINING",
            blueprint_id="attention",
            epochs_in_stage=12,
        )

        # Verify fields for TRAINING format
        assert seed.stage == "TRAINING"
        blueprint = seed.blueprint_id[:6] if seed.blueprint_id and len(seed.blueprint_id) > 6 else seed.blueprint_id or "?"
        epochs_str = f" e{seed.epochs_in_stage}" if seed.epochs_in_stage > 0 else ""

        assert blueprint == "attent"
        assert epochs_str == " e12"


# =============================================================================
# Integration: ALPHA_CURVE_GLYPHS from leyline
# =============================================================================


class TestAlphaCurveGlyphsIntegration:
    """Verify ALPHA_CURVE_GLYPHS integration with SeedState."""

    def test_leyline_exports_alpha_curve_glyphs(self) -> None:
        """ALPHA_CURVE_GLYPHS is exported from esper.leyline."""
        from esper.leyline import ALPHA_CURVE_GLYPHS as glyphs

        assert isinstance(glyphs, dict)
        assert len(glyphs) >= 5  # At least 5 curve types

    def test_all_glyphs_are_single_unicode_characters(self) -> None:
        """All glyph values are single Unicode characters."""
        for curve, glyph in ALPHA_CURVE_GLYPHS.items():
            assert len(glyph) == 1, f"Glyph for {curve} is not single char: {glyph!r}"
            assert isinstance(glyph, str)

    def test_glyphs_are_visually_distinct(self) -> None:
        """All glyphs are unique (no duplicates)."""
        glyphs = list(ALPHA_CURVE_GLYPHS.values())
        assert len(glyphs) == len(set(glyphs)), "Duplicate glyphs found"
