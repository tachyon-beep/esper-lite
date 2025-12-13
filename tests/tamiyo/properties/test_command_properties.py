"""Property-based tests for TamiyoDecision command conversion.

Tier 5 properties verify correct conversion to AdaptationCommand:
1. Round-trip preservation - Key info preserved in conversion
2. Command type mapping - Each action maps to correct CommandType
3. Risk level assignment - Appropriate risk levels for each action
4. Blueprint extraction - GERMINATE correctly extracts blueprint_id
5. Field propagation - All fields correctly transferred

These properties ensure TamiyoDecision.to_command() is a reliable transformation.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from esper.leyline import CommandType, RiskLevel, SeedStage
from esper.leyline.actions import build_action_enum
from esper.tamiyo.decisions import TamiyoDecision

# Import shared strategies
from tests.strategies import bounded_floats


# Build action enum once for tests
ActionEnum = build_action_enum("cnn")


# =============================================================================
# Strategies for Command Testing
# =============================================================================

@st.composite
def tamiyo_decisions(draw, action_type: str | None = None):
    """Generate TamiyoDecision instances.

    Args:
        action_type: If specified, generate only this action type.
                     Options: "WAIT", "FOSSILIZE", "CULL", "GERMINATE"
    """
    if action_type == "WAIT":
        action = ActionEnum.WAIT
    elif action_type == "FOSSILIZE":
        action = ActionEnum.FOSSILIZE
    elif action_type == "CULL":
        action = ActionEnum.CULL
    elif action_type == "GERMINATE":
        blueprint = draw(st.sampled_from(["conv_light", "conv_heavy", "attention"]))
        action = getattr(ActionEnum, f"GERMINATE_{blueprint.upper()}")
    else:
        # Any action
        action = draw(st.sampled_from([
            ActionEnum.WAIT,
            ActionEnum.FOSSILIZE,
            ActionEnum.CULL,
            ActionEnum.GERMINATE_CONV_LIGHT,
            ActionEnum.GERMINATE_CONV_HEAVY,
            ActionEnum.GERMINATE_ATTENTION,
        ]))

    target_seed_id = None
    if action.name in ("FOSSILIZE", "CULL"):
        target_seed_id = draw(st.text(min_size=1, max_size=16, alphabet="abcdefgh0123456789"))
    elif action.name == "WAIT":
        # WAIT can optionally have a target
        target_seed_id = draw(st.one_of(
            st.none(),
            st.text(min_size=1, max_size=16, alphabet="abcdefgh0123456789")
        ))

    return TamiyoDecision(
        action=action,
        target_seed_id=target_seed_id,
        reason=draw(st.text(min_size=1, max_size=100)),
        confidence=draw(bounded_floats(0.0, 1.0)),
    )


# =============================================================================
# Property: Command Type Mapping
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestCommandTypeMapping:
    """Action → CommandType mapping is correct and deterministic."""

    @given(decision=tamiyo_decisions(action_type="WAIT"))
    @settings(max_examples=100)
    def test_wait_maps_to_request_state(self, decision):
        """Property: WAIT → CommandType.REQUEST_STATE."""
        command = decision.to_command()
        assert command.command_type == CommandType.REQUEST_STATE

    @given(decision=tamiyo_decisions(action_type="GERMINATE"))
    @settings(max_examples=100)
    def test_germinate_maps_to_germinate(self, decision):
        """Property: GERMINATE_* → CommandType.GERMINATE."""
        command = decision.to_command()
        assert command.command_type == CommandType.GERMINATE

    @given(decision=tamiyo_decisions(action_type="FOSSILIZE"))
    @settings(max_examples=100)
    def test_fossilize_maps_to_advance_stage(self, decision):
        """Property: FOSSILIZE → CommandType.ADVANCE_STAGE."""
        command = decision.to_command()
        assert command.command_type == CommandType.ADVANCE_STAGE

    @given(decision=tamiyo_decisions(action_type="CULL"))
    @settings(max_examples=100)
    def test_cull_maps_to_cull(self, decision):
        """Property: CULL → CommandType.CULL."""
        command = decision.to_command()
        assert command.command_type == CommandType.CULL

    @given(decision=tamiyo_decisions())
    @settings(max_examples=200)
    def test_command_type_always_valid(self, decision):
        """Property: to_command() always produces valid CommandType."""
        command = decision.to_command()
        assert command.command_type is not None
        assert isinstance(command.command_type, CommandType)


# =============================================================================
# Property: Risk Level Assignment
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestRiskLevelAssignment:
    """Appropriate risk levels for each action type."""

    @given(decision=tamiyo_decisions(action_type="WAIT"))
    @settings(max_examples=50)
    def test_wait_is_green(self, decision):
        """Property: WAIT is low risk (GREEN)."""
        command = decision.to_command()
        assert command.risk_level == RiskLevel.GREEN

    @given(decision=tamiyo_decisions(action_type="GERMINATE"))
    @settings(max_examples=50)
    def test_germinate_is_yellow(self, decision):
        """Property: GERMINATE is medium risk (YELLOW)."""
        command = decision.to_command()
        assert command.risk_level == RiskLevel.YELLOW

    @given(decision=tamiyo_decisions(action_type="FOSSILIZE"))
    @settings(max_examples=50)
    def test_fossilize_is_yellow(self, decision):
        """Property: FOSSILIZE is medium risk (YELLOW)."""
        command = decision.to_command()
        assert command.risk_level == RiskLevel.YELLOW

    @given(decision=tamiyo_decisions(action_type="CULL"))
    @settings(max_examples=50)
    def test_cull_is_orange(self, decision):
        """Property: CULL is higher risk (ORANGE)."""
        command = decision.to_command()
        assert command.risk_level == RiskLevel.ORANGE

    @given(decision=tamiyo_decisions())
    @settings(max_examples=200)
    def test_risk_level_always_valid(self, decision):
        """Property: Risk level is always a valid RiskLevel."""
        command = decision.to_command()
        assert command.risk_level is not None
        assert isinstance(command.risk_level, RiskLevel)


# =============================================================================
# Property: Blueprint Extraction
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestBlueprintExtraction:
    """GERMINATE correctly extracts and propagates blueprint_id."""

    @given(blueprint=st.sampled_from(["conv_light", "conv_heavy", "attention"]))
    @settings(max_examples=50)
    def test_germinate_extracts_blueprint(self, blueprint):
        """Property: GERMINATE_{BLUEPRINT} extracts correct blueprint_id."""
        action = getattr(ActionEnum, f"GERMINATE_{blueprint.upper()}")
        decision = TamiyoDecision(action=action, reason="test")

        command = decision.to_command()

        assert command.blueprint_id == blueprint, \
            f"Expected blueprint '{blueprint}', got '{command.blueprint_id}'"

    @given(decision=tamiyo_decisions(action_type="GERMINATE"))
    @settings(max_examples=100)
    def test_germinate_blueprint_not_none(self, decision):
        """Property: GERMINATE commands always have blueprint_id."""
        command = decision.to_command()
        assert command.blueprint_id is not None
        assert len(command.blueprint_id) > 0

    @given(decision=tamiyo_decisions(action_type="WAIT"))
    @settings(max_examples=50)
    def test_wait_no_blueprint(self, decision):
        """Property: WAIT commands have no blueprint_id."""
        command = decision.to_command()
        assert command.blueprint_id is None

    @given(decision=tamiyo_decisions(action_type="CULL"))
    @settings(max_examples=50)
    def test_cull_no_blueprint(self, decision):
        """Property: CULL commands have no blueprint_id."""
        command = decision.to_command()
        assert command.blueprint_id is None


# =============================================================================
# Property: Field Propagation
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestFieldPropagation:
    """All fields correctly transferred from Decision to Command."""

    @given(decision=tamiyo_decisions())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_target_seed_id_preserved(self, decision):
        """Property: target_seed_id is preserved in conversion."""
        command = decision.to_command()
        assert command.target_seed_id == decision.target_seed_id

    @given(decision=tamiyo_decisions())
    @settings(max_examples=200)
    def test_reason_preserved(self, decision):
        """Property: reason is preserved in conversion."""
        command = decision.to_command()
        assert command.reason == decision.reason

    @given(decision=tamiyo_decisions())
    @settings(max_examples=200)
    def test_confidence_preserved(self, decision):
        """Property: confidence is preserved in conversion."""
        command = decision.to_command()
        assert command.confidence == decision.confidence

    @given(
        target_id=st.text(min_size=1, max_size=20, alphabet="abcdefgh0123456789"),
        reason=st.text(min_size=1, max_size=100),
        confidence=bounded_floats(0.0, 1.0),
    )
    @settings(max_examples=100)
    def test_all_fields_preserved(self, target_id, reason, confidence):
        """Property: All configurable fields preserved together."""
        decision = TamiyoDecision(
            action=ActionEnum.FOSSILIZE,
            target_seed_id=target_id,
            reason=reason,
            confidence=confidence,
        )

        command = decision.to_command()

        assert command.target_seed_id == target_id
        assert command.reason == reason
        assert command.confidence == confidence


# =============================================================================
# Property: Target Stage Assignment
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestTargetStageAssignment:
    """Correct target_stage for stage-changing commands."""

    @given(decision=tamiyo_decisions(action_type="GERMINATE"))
    @settings(max_examples=50)
    def test_germinate_targets_germinated(self, decision):
        """Property: GERMINATE targets GERMINATED stage."""
        command = decision.to_command()
        assert command.target_stage == SeedStage.GERMINATED

    @given(decision=tamiyo_decisions(action_type="FOSSILIZE"))
    @settings(max_examples=50)
    def test_fossilize_targets_fossilized(self, decision):
        """Property: FOSSILIZE targets FOSSILIZED stage."""
        command = decision.to_command()
        assert command.target_stage == SeedStage.FOSSILIZED

    @given(decision=tamiyo_decisions(action_type="CULL"))
    @settings(max_examples=50)
    def test_cull_targets_culled(self, decision):
        """Property: CULL targets CULLED stage."""
        command = decision.to_command()
        assert command.target_stage == SeedStage.CULLED

    @given(decision=tamiyo_decisions(action_type="WAIT"))
    @settings(max_examples=50)
    def test_wait_no_target_stage(self, decision):
        """Property: WAIT has no target_stage."""
        command = decision.to_command()
        assert command.target_stage is None


# =============================================================================
# Property: Conversion Determinism
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestConversionDeterminism:
    """to_command() is deterministic and idempotent."""

    @given(decision=tamiyo_decisions())
    @settings(max_examples=100)
    def test_conversion_deterministic(self, decision):
        """Property: Same decision always produces identical command."""
        command1 = decision.to_command()
        command2 = decision.to_command()

        assert command1.command_type == command2.command_type
        assert command1.target_seed_id == command2.target_seed_id
        assert command1.blueprint_id == command2.blueprint_id
        assert command1.target_stage == command2.target_stage
        assert command1.reason == command2.reason
        assert command1.confidence == command2.confidence
        assert command1.risk_level == command2.risk_level


# =============================================================================
# Property: Decision.blueprint_id Property
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestDecisionBlueprintProperty:
    """TamiyoDecision.blueprint_id property works correctly."""

    @given(blueprint=st.sampled_from(["conv_light", "conv_heavy", "attention"]))
    @settings(max_examples=50)
    def test_blueprint_property_extracts_correctly(self, blueprint):
        """Property: blueprint_id property extracts from action name."""
        action = getattr(ActionEnum, f"GERMINATE_{blueprint.upper()}")
        decision = TamiyoDecision(action=action, reason="test")

        assert decision.blueprint_id == blueprint

    @given(decision=tamiyo_decisions(action_type="WAIT"))
    @settings(max_examples=50)
    def test_non_germinate_returns_none(self, decision):
        """Property: Non-GERMINATE actions return None for blueprint_id."""
        assert decision.blueprint_id is None

    @given(decision=tamiyo_decisions(action_type="CULL"))
    @settings(max_examples=50)
    def test_cull_returns_none(self, decision):
        """Property: CULL returns None for blueprint_id."""
        assert decision.blueprint_id is None


# =============================================================================
# Property: Decision __str__ Representation
# =============================================================================

@pytest.mark.property
@pytest.mark.tamiyo
class TestDecisionStringRepresentation:
    """TamiyoDecision.__str__ produces valid representations."""

    @given(decision=tamiyo_decisions())
    @settings(max_examples=100)
    def test_str_not_empty(self, decision):
        """Property: __str__ always returns non-empty string."""
        s = str(decision)
        assert s
        assert len(s) > 0

    @given(decision=tamiyo_decisions())
    @settings(max_examples=100)
    def test_str_contains_action(self, decision):
        """Property: __str__ includes the action name."""
        s = str(decision)
        assert decision.action.name in s

    @given(decision=tamiyo_decisions(action_type="FOSSILIZE"))
    @settings(max_examples=50)
    def test_str_contains_target_when_present(self, decision):
        """Property: __str__ includes target_seed_id when present."""
        if decision.target_seed_id:
            s = str(decision)
            assert decision.target_seed_id in s
