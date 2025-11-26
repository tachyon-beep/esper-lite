from esper.simic import PolicyValidator, ValidationConfig


def test_validator_passes_within_thresholds() -> None:
    validator = PolicyValidator(
        ValidationConfig(
            min_average_reward=0.1,
            max_policy_loss=1.0,
            max_value_loss=1.0,
            max_param_loss=1.0,
            min_entropy=0.01,
        )
    )
    metrics = {
        "average_reward": 0.2,
        "policy_loss": 0.5,
        "value_loss": 0.7,
        "param_loss": 0.5,
        "policy_entropy": 0.05,
    }
    result = validator.validate(metrics)
    assert result.passed
    assert not result.reasons


def test_validator_collects_failures() -> None:
    validator = PolicyValidator(
        ValidationConfig(
            min_average_reward=0.2,
            max_policy_loss=0.3,
            max_value_loss=0.4,
            max_param_loss=0.1,
            min_entropy=0.05,
        )
    )
    metrics = {
        "average_reward": 0.1,
        "policy_loss": 0.6,
        "value_loss": 0.7,
        "param_loss": 0.3,
        "policy_entropy": 0.01,
    }
    result = validator.validate(metrics)
    assert not result.passed
    assert len(result.reasons) == 5
    assert "average_reward" in result.reasons[0]
