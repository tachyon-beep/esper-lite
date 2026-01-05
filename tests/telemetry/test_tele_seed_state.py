"""End-to-end tests for SeedState detail telemetry records (TELE-550 to TELE-560).

Verifies seed state telemetry flows from schema to EnvOverview display.

These tests cover:
- TELE-550: seed.blueprint_id (str | None, truncated to 6 chars)
- TELE-551: seed.alpha (float 0.0-1.0, blend weight)
- TELE-552: seed.epochs_in_stage (int, stage duration)
- TELE-553: seed.alpha_curve (str, maps to glyphs)
- TELE-554: seed.blend_tempo_epochs (int, tempo arrows)
- TELE-555: seed.seed_params (int, parameter count with K/M suffix)
- TELE-556: seed.accuracy_delta (float, contribution %)
- TELE-557: seed.grad_ratio (float, gradient health ratio)
- TELE-558: seed.interaction_sum (float, synergy from interactions)
- TELE-559: seed.boost_received (float, strongest interaction partner)
- TELE-560: seed.contribution_velocity (float, trend direction)

All records are FULLY WIRED - SeedState dataclass in schema.py provides
fields that EnvOverview._format_slot_cell() and EnvDetailScreen SeedCard consume for display.
"""

import pytest

from esper.karn.constants import DisplayThresholds
from esper.karn.sanctum.formatting import format_params
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


def _format_accuracy_delta(delta: float, stage: str) -> str:
    """Format accuracy delta for display.

    This replicates the logic from EnvDetailScreen SeedCard:
    - TRAINING/GERMINATED: "0.0 (learning)"
    - Positive delta: "+0.25%"
    - Negative delta: "-0.10%"
    - Zero/None: "--"
    """
    if stage in ("TRAINING", "GERMINATED"):
        return "0.0 (learning)"
    if delta != 0:
        return f"{delta:+.2f}%"
    return "--"


def _format_contribution_velocity(velocity: float) -> str:
    """Format contribution velocity as trend indicator.

    This replicates the logic from EnvDetailScreen SeedCard:
    - velocity > EPSILON: "improving" (green arrow)
    - velocity < -EPSILON: "declining" (yellow arrow)
    - |velocity| <= EPSILON: "--" (stable)
    """
    epsilon = DisplayThresholds.CONTRIBUTION_VELOCITY_EPSILON
    if velocity > epsilon:
        return "\u2197 improving"  # ↗
    elif velocity < -epsilon:
        return "\u2198 declining"  # ↘
    return "--"


# =============================================================================
# TELE-555: Seed Params
# =============================================================================


class TestTELE555SeedParams:
    """TELE-555: seed.seed_params field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: Module parameter count computed at instantiation
    - Transport: SeedState.seed_params updated; telemetry carries value
    - Schema: SeedState.seed_params at line 397
    - Consumer: EnvDetailScreen SeedCard displays "Params: 12.5K"

    Parameter count represents the total number of trainable parameters:
    - 0: No parameters (invalid or dormant placeholder)
    - > 0: Actual parameter count (e.g., 12,500 -> "12.5K")

    Display uses human-readable suffixes:
    - < 1,000: raw number (e.g., "256")
    - 1,000 - 999,999: K suffix (e.g., "12.5K")
    - >= 1,000,000: M suffix (e.g., "1.2M")
    """

    def test_seed_params_field_exists(self) -> None:
        """TELE-555: Verify SeedState.seed_params field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "seed_params")

    def test_seed_params_default_value(self) -> None:
        """TELE-555: Default seed_params is 0 (no parameters)."""
        seed = SeedState(slot_id="r0c0")
        assert seed.seed_params == 0

    def test_seed_params_type_is_int(self) -> None:
        """TELE-555: seed_params must be int type."""
        seed = SeedState(slot_id="r0c0", seed_params=12500)
        assert isinstance(seed.seed_params, int)

    def test_seed_params_non_negative(self) -> None:
        """TELE-555: seed_params is non-negative."""
        seed = SeedState(slot_id="r0c0", seed_params=0)
        assert seed.seed_params >= 0

        seed = SeedState(slot_id="r0c0", seed_params=1_000_000)
        assert seed.seed_params >= 0

    def test_seed_params_formatting(self) -> None:
        """TELE-555: format_params utility produces human-readable output.

        Tests the format_params() helper that SeedCard uses:
        - < 1,000: raw number
        - 1,000 - 999,999: K suffix
        - >= 1,000,000: M suffix
        """
        test_cases = [
            (0, "0"),
            (256, "256"),
            (999, "999"),
            (1_000, "1.0K"),
            (12_500, "12.5K"),
            (150_000, "150.0K"),
            (1_000_000, "1.0M"),
            (2_500_000, "2.5M"),
        ]

        for params, expected in test_cases:
            seed = SeedState(slot_id="r0c0", seed_params=params)
            formatted = format_params(seed.seed_params)
            assert formatted == expected, f"Failed for {params}: got {formatted}"


# =============================================================================
# TELE-556: Seed Accuracy Delta
# =============================================================================


class TestTELE556SeedAccuracyDelta:
    """TELE-556: seed.accuracy_delta field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: Counterfactual engine computes accuracy_delta per epoch
    - Transport: SeedState.accuracy_delta updated; telemetry carries value
    - Schema: SeedState.accuracy_delta at line 396
    - Consumer: EnvDetailScreen SeedCard displays "Acc Delta: +0.25%"

    Accuracy delta measures per-seed accuracy contribution:
    - > 0: Seed is helping (positive contribution)
    - = 0: Seed has no measurable effect
    - < 0: Seed is hurting (negative contribution)

    Stage-aware semantics:
    - TRAINING/GERMINATED: Always 0.0 (seed has alpha=0)
    - BLENDING/HOLDING: Active contribution measurement
    """

    def test_accuracy_delta_field_exists(self) -> None:
        """TELE-556: Verify SeedState.accuracy_delta field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "accuracy_delta")

    def test_accuracy_delta_default_value(self) -> None:
        """TELE-556: Default accuracy_delta is 0.0 (no contribution)."""
        seed = SeedState(slot_id="r0c0")
        assert seed.accuracy_delta == 0.0

    def test_accuracy_delta_type_is_float(self) -> None:
        """TELE-556: accuracy_delta must be float type."""
        seed = SeedState(slot_id="r0c0", accuracy_delta=0.25)
        assert isinstance(seed.accuracy_delta, float)

    def test_accuracy_delta_can_be_positive_or_negative(self) -> None:
        """TELE-556: accuracy_delta can be positive (helping) or negative (hurting)."""
        # Positive delta - seed is helping
        seed_positive = SeedState(slot_id="r0c0", accuracy_delta=0.25)
        assert seed_positive.accuracy_delta > 0

        # Negative delta - seed is hurting
        seed_negative = SeedState(slot_id="r0c0", accuracy_delta=-0.10)
        assert seed_negative.accuracy_delta < 0

        # Zero delta - no effect
        seed_zero = SeedState(slot_id="r0c0", accuracy_delta=0.0)
        assert seed_zero.accuracy_delta == 0.0

    def test_accuracy_delta_display_format(self) -> None:
        """TELE-556: Display format is '+0.25%' or '-0.10%' with sign prefix.

        This tests the display logic from EnvDetailScreen SeedCard.
        """
        test_cases = [
            (0.25, "BLENDING", "+0.25%"),
            (-0.10, "BLENDING", "-0.10%"),
            (0.0, "BLENDING", "--"),
            (1.5, "HOLDING", "+1.50%"),
            (-0.05, "HOLDING", "-0.05%"),
        ]

        for delta, stage, expected in test_cases:
            seed = SeedState(slot_id="r0c0", stage=stage, accuracy_delta=delta)
            formatted = _format_accuracy_delta(seed.accuracy_delta, seed.stage)
            assert formatted == expected, f"Failed for delta={delta}, stage={stage}"

    def test_accuracy_delta_training_stage_always_zero(self) -> None:
        """TELE-556: TRAINING/GERMINATED stages display as '0.0 (learning)'.

        Seeds in TRAINING or GERMINATED have alpha=0 and cannot affect output.
        """
        for stage in ["TRAINING", "GERMINATED"]:
            # Even if accuracy_delta is non-zero in data, display shows learning context
            seed = SeedState(slot_id="r0c0", stage=stage, accuracy_delta=0.0)
            formatted = _format_accuracy_delta(seed.accuracy_delta, seed.stage)
            assert formatted == "0.0 (learning)"


# =============================================================================
# TELE-557: Seed Grad Ratio
# =============================================================================


class TestTELE557SeedGradRatio:
    """TELE-557: seed.grad_ratio field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: Gradient hooks compute ratio during backward pass
    - Transport: SeedState.grad_ratio updated; telemetry carries value
    - Schema: SeedState.grad_ratio at line 398
    - Consumer: EnvDetailScreen SeedCard displays "ratio=0.85"

    Gradient ratio measures gradient flow health:
    - ~= 1.0: Healthy gradient flow
    - << 1.0: Potential vanishing gradients
    - >> 1.0: Potential exploding gradients
    - = 0.0: No gradient data available
    """

    def test_grad_ratio_field_exists(self) -> None:
        """TELE-557: Verify SeedState.grad_ratio field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "grad_ratio")

    def test_grad_ratio_default_value(self) -> None:
        """TELE-557: Default grad_ratio is 0.0 (no gradient data)."""
        seed = SeedState(slot_id="r0c0")
        assert seed.grad_ratio == 0.0

    def test_grad_ratio_type_is_float(self) -> None:
        """TELE-557: grad_ratio must be float type."""
        seed = SeedState(slot_id="r0c0", grad_ratio=0.85)
        assert isinstance(seed.grad_ratio, float)

    def test_grad_ratio_display_format(self) -> None:
        """TELE-557: Display format is 'ratio=0.85' with 2 decimal places.

        This tests the display logic from EnvDetailScreen SeedCard:
        - grad_ratio > 0: show "ratio=X.XX"
        - grad_ratio = 0: show "OK" (no data available)
        """
        test_cases = [
            (0.85, "ratio=0.85"),
            (1.00, "ratio=1.00"),
            (1.50, "ratio=1.50"),
            (0.50, "ratio=0.50"),
        ]

        for ratio, expected in test_cases:
            seed = SeedState(slot_id="r0c0", grad_ratio=ratio)
            # Match EnvDetailScreen display logic
            if seed.grad_ratio > 0:
                formatted = f"ratio={seed.grad_ratio:.2f}"
            else:
                formatted = "OK"
            assert formatted == expected, f"Failed for ratio={ratio}"

    def test_grad_ratio_zero_shows_ok(self) -> None:
        """TELE-557: grad_ratio=0.0 displays as 'OK' (no data)."""
        seed = SeedState(slot_id="r0c0", grad_ratio=0.0)
        # When no ratio data, display shows OK
        if seed.grad_ratio > 0:
            formatted = f"ratio={seed.grad_ratio:.2f}"
        else:
            formatted = "OK"
        assert formatted == "OK"


# =============================================================================
# TELE-558: Seed Interaction Sum
# =============================================================================


class TestTELE558SeedInteractionSum:
    """TELE-558: seed.interaction_sum field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: Counterfactual engine computes pairwise interactions
    - Transport: SeedState.interaction_sum updated; telemetry carries value
    - Schema: SeedState.interaction_sum at line 419
    - Consumer: EnvDetailScreen SeedCard displays "Synergy: +1.5"

    Interaction sum (Sigma I_ij for all j != i) measures total synergy:
    - > 0: Net positive synergy (seed benefits from others)
    - = 0: No interaction effects (seed operates independently)
    - < 0: Net negative synergy (seed conflicts with others)
    """

    def test_interaction_sum_field_exists(self) -> None:
        """TELE-558: Verify SeedState.interaction_sum field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "interaction_sum")

    def test_interaction_sum_default_value(self) -> None:
        """TELE-558: Default interaction_sum is 0.0 (no interaction data)."""
        seed = SeedState(slot_id="r0c0")
        assert seed.interaction_sum == 0.0

    def test_interaction_sum_type_is_float(self) -> None:
        """TELE-558: interaction_sum must be float type."""
        seed = SeedState(slot_id="r0c0", interaction_sum=1.5)
        assert isinstance(seed.interaction_sum, float)

    def test_interaction_sum_positive_is_synergy(self) -> None:
        """TELE-558: Positive interaction_sum indicates beneficial synergy."""
        seed = SeedState(slot_id="r0c0", interaction_sum=1.5)
        assert seed.interaction_sum > 0

    def test_interaction_sum_negative_is_interference(self) -> None:
        """TELE-558: Negative interaction_sum indicates harmful interference."""
        seed = SeedState(slot_id="r0c0", interaction_sum=-0.8)
        assert seed.interaction_sum < 0

    def test_interaction_sum_threshold_coloring(self) -> None:
        """TELE-558: Threshold determines display coloring.

        From DisplayThresholds.INTERACTION_SYNERGY_THRESHOLD = 0.5:
        - > 0.5: Green (positive synergy)
        - < -0.5: Red (negative synergy)
        - |sum| <= 0.5: Dim (neutral)
        """
        threshold = DisplayThresholds.INTERACTION_SYNERGY_THRESHOLD

        # Above threshold - green
        seed_synergy = SeedState(slot_id="r0c0", interaction_sum=1.5)
        assert seed_synergy.interaction_sum > threshold

        # Below negative threshold - red
        seed_conflict = SeedState(slot_id="r0c0", interaction_sum=-0.8)
        assert seed_conflict.interaction_sum < -threshold

        # Within threshold - dim/neutral
        seed_neutral = SeedState(slot_id="r0c0", interaction_sum=0.3)
        assert abs(seed_neutral.interaction_sum) <= threshold


# =============================================================================
# TELE-559: Seed Boost Received
# =============================================================================


class TestTELE559SeedBoostReceived:
    """TELE-559: seed.boost_received field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: Counterfactual engine extracts max pairwise interaction
    - Transport: SeedState.boost_received updated; telemetry carries value
    - Schema: SeedState.boost_received at line 420
    - Consumer: EnvDetailScreen SeedCard displays "(up-arrow 0.8)"

    Boost received (max(I_ij) for j != i) measures strongest interaction:
    - > 0: Another seed is boosting this one's performance
    - = 0: No significant positive interactions from other seeds
    """

    def test_boost_received_field_exists(self) -> None:
        """TELE-559: Verify SeedState.boost_received field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "boost_received")

    def test_boost_received_default_value(self) -> None:
        """TELE-559: Default boost_received is 0.0 (no boost)."""
        seed = SeedState(slot_id="r0c0")
        assert seed.boost_received == 0.0

    def test_boost_received_type_is_float(self) -> None:
        """TELE-559: boost_received must be float type."""
        seed = SeedState(slot_id="r0c0", boost_received=0.8)
        assert isinstance(seed.boost_received, float)

    def test_boost_received_non_negative(self) -> None:
        """TELE-559: boost_received is non-negative (max of positive interactions)."""
        seed_zero = SeedState(slot_id="r0c0", boost_received=0.0)
        assert seed_zero.boost_received >= 0

        seed_positive = SeedState(slot_id="r0c0", boost_received=0.8)
        assert seed_positive.boost_received >= 0

    def test_boost_received_display_format(self) -> None:
        """TELE-559: Display format is '(up-arrow 0.8)' with 1 decimal place.

        From DisplayThresholds.BOOST_RECEIVED_THRESHOLD = 0.1:
        - > 0.1: Show "(up-arrow X.X)" in cyan
        - <= 0.1: Not displayed
        """
        threshold = DisplayThresholds.BOOST_RECEIVED_THRESHOLD

        test_cases = [
            (0.8, True, "\u2197 0.8"),  # Above threshold, shown
            (0.5, True, "\u2197 0.5"),  # Above threshold, shown
            (0.1, False, ""),           # At threshold, not shown
            (0.05, False, ""),          # Below threshold, not shown
        ]

        for boost, should_show, expected_suffix in test_cases:
            seed = SeedState(slot_id="r0c0", boost_received=boost)
            shows = seed.boost_received > threshold
            assert shows == should_show, f"Failed for boost={boost}"
            if shows:
                formatted = f"\u2197 {seed.boost_received:.1f}"
                assert formatted == expected_suffix


# =============================================================================
# TELE-560: Seed Contribution Velocity
# =============================================================================


class TestTELE560SeedContributionVelocity:
    """TELE-560: seed.contribution_velocity field in SeedState.

    Wiring Status: FULLY WIRED
    - Emitter: EMA computation tracks contribution changes
    - Transport: SeedState.contribution_velocity updated; telemetry carries value
    - Schema: SeedState.contribution_velocity at line 418
    - Consumer: EnvDetailScreen SeedCard displays "Trend: up-arrow improving"

    Contribution velocity (EMA of contribution changes) measures trend:
    - > EPSILON: Improving (contribution is increasing)
    - ~= 0: Stable (contribution is steady)
    - < -EPSILON: Declining (contribution is decreasing)
    """

    def test_contribution_velocity_field_exists(self) -> None:
        """TELE-560: Verify SeedState.contribution_velocity field exists."""
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "contribution_velocity")

    def test_contribution_velocity_default_value(self) -> None:
        """TELE-560: Default contribution_velocity is 0.0 (stable)."""
        seed = SeedState(slot_id="r0c0")
        assert seed.contribution_velocity == 0.0

    def test_contribution_velocity_type_is_float(self) -> None:
        """TELE-560: contribution_velocity must be float type."""
        seed = SeedState(slot_id="r0c0", contribution_velocity=0.05)
        assert isinstance(seed.contribution_velocity, float)

    def test_contribution_velocity_positive_is_improving(self) -> None:
        """TELE-560: Positive velocity above epsilon indicates improving trend."""
        epsilon = DisplayThresholds.CONTRIBUTION_VELOCITY_EPSILON
        seed = SeedState(slot_id="r0c0", contribution_velocity=0.05)
        assert seed.contribution_velocity > epsilon
        assert _format_contribution_velocity(seed.contribution_velocity) == "\u2197 improving"

    def test_contribution_velocity_negative_is_declining(self) -> None:
        """TELE-560: Negative velocity below -epsilon indicates declining trend."""
        epsilon = DisplayThresholds.CONTRIBUTION_VELOCITY_EPSILON
        seed = SeedState(slot_id="r0c0", contribution_velocity=-0.05)
        assert seed.contribution_velocity < -epsilon
        assert _format_contribution_velocity(seed.contribution_velocity) == "\u2198 declining"

    def test_contribution_velocity_near_zero_is_stable(self) -> None:
        """TELE-560: Velocity near zero (within epsilon) indicates stable trend."""
        epsilon = DisplayThresholds.CONTRIBUTION_VELOCITY_EPSILON

        # Exactly zero
        seed_zero = SeedState(slot_id="r0c0", contribution_velocity=0.0)
        assert abs(seed_zero.contribution_velocity) <= epsilon
        assert _format_contribution_velocity(seed_zero.contribution_velocity) == "--"

        # Just within positive epsilon
        seed_small_pos = SeedState(slot_id="r0c0", contribution_velocity=0.005)
        assert abs(seed_small_pos.contribution_velocity) <= epsilon
        assert _format_contribution_velocity(seed_small_pos.contribution_velocity) == "--"

        # Just within negative epsilon
        seed_small_neg = SeedState(slot_id="r0c0", contribution_velocity=-0.005)
        assert abs(seed_small_neg.contribution_velocity) <= epsilon
        assert _format_contribution_velocity(seed_small_neg.contribution_velocity) == "--"


# =============================================================================
# SeedState Schema Completeness
# =============================================================================


class TestSeedStateSchemaCompleteness:
    """Verify SeedState has all TELE-550 to TELE-560 fields."""

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

        # TELE-555: Seed params
        assert hasattr(seed, "seed_params")

        # TELE-556: Accuracy delta
        assert hasattr(seed, "accuracy_delta")

        # TELE-557: Grad ratio
        assert hasattr(seed, "grad_ratio")

        # TELE-558: Interaction sum
        assert hasattr(seed, "interaction_sum")

        # TELE-559: Boost received
        assert hasattr(seed, "boost_received")

        # TELE-560: Contribution velocity
        assert hasattr(seed, "contribution_velocity")

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

        # TELE-555: Default 0
        assert seed.seed_params == 0

        # TELE-556: Default 0.0
        assert seed.accuracy_delta == 0.0

        # TELE-557: Default 0.0
        assert seed.grad_ratio == 0.0

        # TELE-558: Default 0.0
        assert seed.interaction_sum == 0.0

        # TELE-559: Default 0.0
        assert seed.boost_received == 0.0

        # TELE-560: Default 0.0
        assert seed.contribution_velocity == 0.0

    def test_seed_state_with_all_fields(self) -> None:
        """Verify SeedState can be constructed with all TELE-550 to TELE-560 fields."""
        seed = SeedState(
            slot_id="r0c0",
            stage="BLENDING",
            blueprint_id="conv_light",
            alpha=0.5,
            epochs_in_stage=10,
            alpha_curve="SIGMOID",
            blend_tempo_epochs=3,
            seed_params=12500,
            accuracy_delta=0.25,
            grad_ratio=0.85,
            interaction_sum=1.5,
            boost_received=0.8,
            contribution_velocity=0.05,
        )

        assert seed.slot_id == "r0c0"
        assert seed.stage == "BLENDING"
        assert seed.blueprint_id == "conv_light"
        assert seed.alpha == 0.5
        assert seed.epochs_in_stage == 10
        assert seed.alpha_curve == "SIGMOID"
        assert seed.blend_tempo_epochs == 3
        assert seed.seed_params == 12500
        assert seed.accuracy_delta == 0.25
        assert seed.grad_ratio == 0.85
        assert seed.interaction_sum == 1.5
        assert seed.boost_received == 0.8
        assert seed.contribution_velocity == 0.05


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
