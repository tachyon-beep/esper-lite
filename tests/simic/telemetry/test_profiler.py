"""Tests for torch.profiler integration."""

import os
import tempfile

import torch

from esper.simic.telemetry.profiler import training_profiler


class TestTrainingProfiler:
    """Tests for training_profiler context manager."""

    def test_disabled_returns_none(self):
        """When disabled=False, profiler yields None."""
        with training_profiler(enabled=False) as prof:
            assert prof is None

    def test_enabled_returns_profiler(self):
        """When enabled=True, profiler yields a torch.profiler.profile instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with training_profiler(
                output_dir=tmpdir,
                enabled=True,
                wait=0,
                warmup=0,
                active=1,
                repeat=1,
            ) as prof:
                assert prof is not None
                assert isinstance(prof, torch.profiler.profile)
                # Step the profiler once (required to complete a cycle)
                prof.step()

    def test_creates_output_directory(self):
        """Profiler creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nested", "profiler_output")
            assert not os.path.exists(output_dir)

            with training_profiler(
                output_dir=output_dir,
                enabled=True,
                wait=0,
                warmup=0,
                active=1,
                repeat=1,
            ) as prof:
                prof.step()

            # Directory should now exist
            assert os.path.isdir(output_dir)

    def test_disabled_does_not_create_directory(self):
        """When disabled, output directory is not created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "should_not_exist")

            with training_profiler(output_dir=output_dir, enabled=False) as prof:
                assert prof is None

            # Directory should NOT be created
            assert not os.path.exists(output_dir)

    def test_schedule_parameters_accepted(self):
        """Schedule parameters (wait, warmup, active, repeat) are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with training_profiler(
                output_dir=tmpdir,
                enabled=True,
                wait=2,
                warmup=1,
                active=3,
                repeat=2,
            ) as prof:
                # Step through the schedule (wait=2, warmup=1, active=3, repeat=2)
                # Total: (2 + 1 + 3) * 2 = 12 steps for full cycle
                for _ in range(12):
                    prof.step()

    def test_optional_flags_accepted(self):
        """Optional profiling flags (record_shapes, profile_memory, with_stack) are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with training_profiler(
                output_dir=tmpdir,
                enabled=True,
                wait=0,
                warmup=0,
                active=1,
                repeat=1,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,  # Keep False to reduce overhead in tests
            ) as prof:
                # Basic tensor op to profile
                x = torch.randn(10, 10)
                _ = x @ x.T
                prof.step()
