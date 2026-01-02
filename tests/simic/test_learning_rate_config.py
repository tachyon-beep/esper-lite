"""Tests for configurable learning rates in TaskSpec.

Learning rates were previously hardcoded throughout the training code.
This parameterization allows task-specific tuning without code changes.
"""



class TestTaskSpecLearningRates:
    """Tests for learning rate configuration in TaskSpec."""

    def test_taskspec_has_host_lr_field(self):
        """TaskSpec should have host_lr field with default 0.01."""
        from esper.runtime.tasks import get_task_spec

        spec = get_task_spec("cifar_baseline")

        # Direct attribute access tests existence and value
        # (AttributeError raised if field doesn't exist)
        assert spec.host_lr == 0.01

    def test_taskspec_has_seed_lr_field(self):
        """TaskSpec should have seed_lr field with default 0.01."""
        from esper.runtime.tasks import get_task_spec

        spec = get_task_spec("cifar_baseline")

        # Direct attribute access tests existence and value
        assert spec.seed_lr == 0.01

    def test_tinystories_has_lr_fields(self):
        """TinyStories spec should also have LR fields."""
        from esper.runtime.tasks import get_task_spec

        spec = get_task_spec("tinystories")

        assert spec.host_lr == 0.01
        assert spec.seed_lr == 0.01

    def test_custom_learning_rates(self):
        """Should be able to create TaskSpec with custom LRs."""
        from esper.runtime.tasks import TaskSpec

        # Minimal TaskSpec with custom LRs
        spec = TaskSpec(
            name="custom",
            topology="cnn",
            task_type="classification",
            model_factory=lambda device: None,  # Dummy
            dataloader_factory=lambda: None,  # Dummy
            host_lr=0.001,
            seed_lr=0.005,
        )

        assert spec.host_lr == 0.001
        assert spec.seed_lr == 0.005

    def test_lr_fields_are_floats(self):
        """LR fields should be float type."""
        from esper.runtime.tasks import get_task_spec

        spec = get_task_spec("cifar_baseline")

        assert isinstance(spec.host_lr, float)
        assert isinstance(spec.seed_lr, float)

    def test_lr_defaults_are_positive(self):
        """Default LRs should be positive values."""
        from esper.runtime.tasks import get_task_spec

        spec = get_task_spec("cifar_baseline")

        assert spec.host_lr > 0
        assert spec.seed_lr > 0


class TestLearningRateUseCases:
    """Tests for common learning rate configuration patterns."""

    def test_different_host_and_seed_lr(self):
        """Common pattern: lower seed LR for fine-tuning."""
        from esper.runtime.tasks import TaskSpec

        spec = TaskSpec(
            name="fine_tune",
            topology="cnn",
            task_type="classification",
            model_factory=lambda device: None,
            dataloader_factory=lambda: None,
            host_lr=0.01,
            seed_lr=0.001,  # 10x smaller for stability
        )

        assert spec.seed_lr < spec.host_lr

    def test_transformer_lower_lr(self):
        """Transformers typically need lower LRs."""
        from esper.runtime.tasks import TaskSpec

        spec = TaskSpec(
            name="transformer_task",
            topology="transformer",
            task_type="lm",
            model_factory=lambda device: None,
            dataloader_factory=lambda: None,
            host_lr=0.0001,  # Lower for transformers
            seed_lr=0.0001,
        )

        assert spec.host_lr == 0.0001
        assert spec.seed_lr == 0.0001
