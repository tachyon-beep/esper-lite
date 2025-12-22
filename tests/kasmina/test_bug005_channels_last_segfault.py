"""Regression test for BUG-005: channels_last + isolate_gradients segfault.

This bug caused a segfault during backward() when:
1. Using channels_last memory format (CNNHost default)
2. With isolate_gradients=True (detach on seed input)
3. In ANY active stage (TRAINING, BLENDING, etc.)

The root cause is a PyTorch bug with non-contiguous (channels_last)
tensors combined with detach() in the autograd graph.

The fix is in SeedSlot.forward(): feed the seed a contiguous_format
detached copy under isolation (avoid channels_last + detach in backward)
without coercing the host_features stream.
"""

import pytest
import torch

from esper.kasmina.host import CNNHost, MorphogeneticModel
from esper.leyline import SeedStage


class TestBug005ChannelsLastSegfault:
    """Regression tests for BUG-005."""

    def test_training_stage_with_channels_last_no_crash(self):
        """The original crash scenario: TRAINING + isolate_gradients + channels_last."""
        x = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        # CNNHost defaults to channels_last memory format
        model = MorphogeneticModel(CNNHost(num_classes=10), device="cpu", slots=["r0c1"])
        model.germinate_seed("norm", "seed-1", slot="r0c1")
        slot = model.seed_slots["r0c1"]
        slot.state.stage = SeedStage.TRAINING
        slot.isolate_gradients = True

        # This used to segfault during backward()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, labels)
        loss.backward()

        # Verify gradients were computed
        assert any(p.grad is not None for p in model.parameters())

    def test_training_stage_with_channels_last_gradient_telemetry(self):
        """Gradient telemetry should work after backward() with channels_last."""
        x = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        model = MorphogeneticModel(CNNHost(num_classes=10), device="cpu", slots=["r0c1"])
        model.germinate_seed("norm", "seed-1", slot="r0c1")
        slot = model.seed_slots["r0c1"]
        slot.state.stage = SeedStage.TRAINING
        slot.isolate_gradients = True

        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, labels)
        loss.backward()

        # This is what the bug report originally claimed crashed (it was actually backward)
        slot.capture_gradient_telemetry()

        # Verify gradient metrics were captured
        assert slot.state.metrics.seed_gradient_norm_ratio > 0

    def test_training_stage_explicit_channels_last(self):
        """Explicitly test with channels_last memory format."""
        x = torch.randn(4, 3, 32, 32)

        model = MorphogeneticModel(
            CNNHost(num_classes=10, memory_format=torch.channels_last),
            device="cpu",
            slots=["r0c1"],
        )
        model.germinate_seed("norm", "seed-1", slot="r0c1")
        slot = model.seed_slots["r0c1"]
        slot.state.stage = SeedStage.TRAINING
        slot.isolate_gradients = True

        out = model(x)
        loss = out.mean()
        loss.backward()  # Should not segfault

    def test_training_stage_contiguous_format_still_works(self):
        """Contiguous format should continue to work (was always fine)."""
        x = torch.randn(4, 3, 32, 32)

        model = MorphogeneticModel(
            CNNHost(num_classes=10, memory_format=torch.contiguous_format),
            device="cpu",
            slots=["r0c1"],
        )
        model.germinate_seed("norm", "seed-1", slot="r0c1")
        slot = model.seed_slots["r0c1"]
        slot.state.stage = SeedStage.TRAINING
        slot.isolate_gradients = True

        out = model(x)
        loss = out.mean()
        loss.backward()

    def test_blending_stage_channels_last_with_fix(self):
        """BLENDING stage with channels_last also had this issue (isolate_gradients)."""
        x = torch.randn(4, 3, 32, 32)

        model = MorphogeneticModel(
            CNNHost(num_classes=10, memory_format=torch.channels_last),
            device="cpu",
            slots=["r0c1"],
        )
        model.germinate_seed("norm", "seed-1", slot="r0c1")
        slot = model.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        slot.set_alpha(0.5)
        slot.isolate_gradients = True

        out = model(x)
        loss = out.mean()
        loss.backward()  # Used to segfault, now fixed

    def test_training_without_isolate_gradients_no_issue(self):
        """TRAINING without isolate_gradients never had this issue."""
        x = torch.randn(4, 3, 32, 32)

        model = MorphogeneticModel(
            CNNHost(num_classes=10, memory_format=torch.channels_last),
            device="cpu",
            slots=["r0c1"],
        )
        model.germinate_seed("norm", "seed-1", slot="r0c1")
        slot = model.seed_slots["r0c1"]
        slot.state.stage = SeedStage.TRAINING
        slot.isolate_gradients = False  # No detach

        out = model(x)
        loss = out.mean()
        loss.backward()
