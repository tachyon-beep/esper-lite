"""Tests for normalizer checkpoint save/load in vectorized training.

B5-PT-02: Normalizer state must be saved in checkpoint metadata to prevent
policy collapse on training resume.
"""

import pytest
import torch


class _StopAfterCheckpointLoaded(RuntimeError):
    pass


def _obs_normalizer_metadata_for_slots(slot_ids: tuple[str, ...]) -> dict[str, object]:
    from esper.leyline import SlotConfig
    from esper.simic.control.normalization import RunningMeanStd
    from esper.simic.training.normalizer_checkpoint import obs_normalizer_metadata
    from esper.tamiyo.policy.features import get_feature_size

    slot_config = SlotConfig(slot_ids=slot_ids)
    state_dim = get_feature_size(slot_config)
    obs_normalizer = RunningMeanStd(shape=(state_dim,), device="cpu", momentum=0.99)
    return obs_normalizer_metadata(obs_normalizer, slot_config=slot_config)


class TestNormalizerCheckpointSave:
    """B5-PT-02 regression tests: verify save path populates normalizer metadata."""

    def test_checkpoint_metadata_format_obs_normalizer(self):
        """Observation normalizer state serializes correctly for checkpoint."""
        from esper.leyline import SlotConfig
        from esper.simic.control.normalization import RunningMeanStd
        from esper.simic.training.normalizer_checkpoint import obs_normalizer_metadata
        from esper.tamiyo.policy.features import get_feature_size

        # Create normalizer with some accumulated state
        slot_config = SlotConfig(slot_ids=("s0", "s1"))
        state_dim = get_feature_size(slot_config)
        obs_normalizer = RunningMeanStd(shape=(state_dim,), device="cpu", momentum=0.99)
        obs_normalizer.update(torch.randn(32, state_dim))
        obs_normalizer.update(torch.randn(32, state_dim))

        # Build metadata dict using the same helper as the checkpoint save path.
        metadata = obs_normalizer_metadata(
            obs_normalizer,
            slot_config=slot_config,
        )

        # Verify round-trip: metadata -> torch.tensor (as load code does)
        restored_mean = torch.tensor(metadata["obs_normalizer_mean"], device="cpu")
        restored_var = torch.tensor(metadata["obs_normalizer_var"], device="cpu")
        restored_count = torch.tensor(metadata["obs_normalizer_count"], device="cpu")

        assert torch.allclose(obs_normalizer.mean, restored_mean)
        assert torch.allclose(obs_normalizer.var, restored_var)
        assert torch.isclose(obs_normalizer.count, restored_count)
        assert metadata["obs_normalizer_momentum"] == 0.99
        assert metadata["obs_normalizer_contract"]["state_dim"] == state_dim

    def test_checkpoint_metadata_format_reward_normalizer(self):
        """Reward normalizer state serializes correctly for checkpoint."""
        from esper.simic.control.normalization import RewardNormalizer

        # Create normalizer with some accumulated state
        reward_normalizer = RewardNormalizer(clip=10.0)
        for _ in range(50):
            reward_normalizer.update_and_normalize(torch.randn(1).item())

        # Build metadata dict using same format as vectorized.py:3775-3778
        metadata = {
            "reward_normalizer_mean": reward_normalizer.mean,
            "reward_normalizer_m2": reward_normalizer.m2,
            "reward_normalizer_count": reward_normalizer.count,
        }

        # Verify round-trip works (load code assigns directly)
        assert metadata["reward_normalizer_mean"] == reward_normalizer.mean
        assert metadata["reward_normalizer_m2"] == reward_normalizer.m2
        assert metadata["reward_normalizer_count"] == reward_normalizer.count

    def test_full_checkpoint_roundtrip(self, tmp_path):
        """B5-PT-02: Full checkpoint save/load preserves normalizer state."""
        from esper.leyline import SlotConfig
        from esper.simic.control.normalization import RunningMeanStd, RewardNormalizer
        from esper.simic.training.normalizer_checkpoint import (
            obs_normalizer_metadata,
            restore_obs_normalizer_from_metadata,
        )
        from esper.tamiyo.policy.features import get_feature_size

        # Create normalizers with accumulated state
        slot_config = SlotConfig(slot_ids=("s0",))
        state_dim = get_feature_size(slot_config)
        obs_normalizer = RunningMeanStd(shape=(state_dim,), device="cpu", momentum=0.99)
        obs_normalizer.update(torch.randn(16, state_dim))
        reward_normalizer = RewardNormalizer(clip=10.0)
        for _ in range(30):
            reward_normalizer.update_and_normalize(torch.randn(1).item())

        # Build checkpoint metadata (exactly as vectorized.py does)
        checkpoint_metadata = {
            **obs_normalizer_metadata(obs_normalizer, slot_config=slot_config),
            "reward_normalizer_mean": reward_normalizer.mean,
            "reward_normalizer_m2": reward_normalizer.m2,
            "reward_normalizer_count": reward_normalizer.count,
            "n_episodes": 42,
        }

        # Simulate torch.save/load
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save({"metadata": checkpoint_metadata}, checkpoint_path)
        loaded = torch.load(checkpoint_path, weights_only=True)
        metadata = loaded["metadata"]

        # Simulate load path (vectorized.py:859-880)
        obs_normalizer_restored = RunningMeanStd(
            shape=(state_dim,),
            device="cpu",
            momentum=0.99,
        )
        restore_obs_normalizer_from_metadata(
            metadata,
            obs_normalizer=obs_normalizer_restored,
            slot_config=slot_config,
            device="cpu",
        )

        reward_normalizer_restored = RewardNormalizer(clip=10.0)
        reward_normalizer_restored.mean = metadata["reward_normalizer_mean"]
        reward_normalizer_restored.m2 = metadata["reward_normalizer_m2"]
        reward_normalizer_restored.count = metadata["reward_normalizer_count"]

        # Verify state was preserved
        assert torch.allclose(obs_normalizer.mean, obs_normalizer_restored.mean)
        assert torch.allclose(obs_normalizer.var, obs_normalizer_restored.var)
        assert torch.isclose(obs_normalizer.count, obs_normalizer_restored.count)
        assert obs_normalizer.momentum == obs_normalizer_restored.momentum

        assert reward_normalizer.mean == reward_normalizer_restored.mean
        assert reward_normalizer.m2 == reward_normalizer_restored.m2
        assert reward_normalizer.count == reward_normalizer_restored.count

    def test_obs_normalizer_resume_requires_contract(self):
        """Resume fails closed when old checkpoints lack an obs contract."""
        from esper.leyline import SlotConfig
        from esper.simic.control.normalization import RunningMeanStd
        from esper.simic.training.normalizer_checkpoint import (
            restore_obs_normalizer_from_metadata,
        )
        from esper.tamiyo.policy.features import get_feature_size

        slot_config = SlotConfig(slot_ids=("s0",))
        state_dim = get_feature_size(slot_config)

        metadata = {
            "obs_normalizer_mean": [0.0] * state_dim,
            "obs_normalizer_var": [1.0] * state_dim,
            "obs_normalizer_count": 10.0,
            "obs_normalizer_momentum": 0.99,
        }
        obs_normalizer = RunningMeanStd(shape=(state_dim,), device="cpu", momentum=0.99)

        with pytest.raises(RuntimeError, match="obs_normalizer_contract"):
            restore_obs_normalizer_from_metadata(
                metadata,
                obs_normalizer=obs_normalizer,
                slot_config=slot_config,
                device="cpu",
            )

    def test_obs_normalizer_resume_rejects_stale_contract(self):
        """Resume fails closed when normalizer stats fit a different obs contract."""
        from esper.leyline import SlotConfig
        from esper.simic.control.normalization import RunningMeanStd
        from esper.simic.training.normalizer_checkpoint import (
            obs_normalizer_metadata,
            restore_obs_normalizer_from_metadata,
        )
        from esper.tamiyo.policy.features import get_feature_size

        checkpoint_slot_config = SlotConfig(slot_ids=("s0",))
        runtime_slot_config = SlotConfig(slot_ids=("s0", "s1"))
        checkpoint_state_dim = get_feature_size(checkpoint_slot_config)
        runtime_state_dim = get_feature_size(runtime_slot_config)
        obs_normalizer = RunningMeanStd(
            shape=(checkpoint_state_dim,),
            device="cpu",
            momentum=0.99,
        )
        metadata = obs_normalizer_metadata(
            obs_normalizer,
            slot_config=checkpoint_slot_config,
        )

        with pytest.raises(RuntimeError, match="Observation normalizer contract mismatch"):
            restore_obs_normalizer_from_metadata(
                metadata,
                obs_normalizer=RunningMeanStd(
                    shape=(runtime_state_dim,),
                    device="cpu",
                    momentum=0.99,
                ),
                slot_config=runtime_slot_config,
                device="cpu",
            )


def test_resume_restores_reward_normalizer_state(monkeypatch, tmp_path):
    import esper.runtime as runtime
    import esper.simic.training.vectorized as vectorized
    from esper.leyline import TelemetryEventType

    original_get_task_spec = runtime.get_task_spec

    def get_task_spec_mock(name: str):
        spec = original_get_task_spec(name)
        spec.dataloader_defaults["mock"] = True
        return spec

    class StubHub:
        def add_backend(self, _backend) -> None:
            return None

        def emit(self, event) -> None:
            if event.event_type == TelemetryEventType.CHECKPOINT_LOADED:
                raise _StopAfterCheckpointLoaded()
            return None

    class StubRewardNormalizer:
        last_instance = None

        def __init__(self, clip: float):
            self.clip = clip
            self.mean = None
            self.m2 = None
            self.count = None
            StubRewardNormalizer.last_instance = self

    class StubSharedBatchIterator:
        def __init__(self, *args, **kwargs):
            return None

        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(runtime, "get_task_spec", get_task_spec_mock)
    monkeypatch.setattr(vectorized, "get_hub", lambda: StubHub())
    monkeypatch.setattr(vectorized, "RewardNormalizer", StubRewardNormalizer)
    monkeypatch.setattr(vectorized, "SharedBatchIterator", StubSharedBatchIterator)
    monkeypatch.setattr(
        vectorized.PPOAgent,
        "load_from_checkpoint_dict",
        lambda *args, **kwargs: object(),
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "metadata": {
                "n_episodes": 0,
                "batches_completed": 0,
                "n_envs": 1,
                **_obs_normalizer_metadata_for_slots(("r0c1",)),
                "reward_normalizer_mean": 1.23,
                "reward_normalizer_m2": 4.56,
                "reward_normalizer_count": 7,
            }
        },
        checkpoint_path,
    )

    with pytest.raises(_StopAfterCheckpointLoaded):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=1,
            chunk_length=1,
            device="cpu",
            devices=["cpu"],
            task="cifar_baseline",
            slots=["r0c1"],
            use_telemetry=False,
            num_workers=0,
            resume_path=str(checkpoint_path),
            quiet_analytics=True,
        )

    reward_normalizer = StubRewardNormalizer.last_instance
    assert reward_normalizer is not None
    assert reward_normalizer.clip == 10.0
    assert reward_normalizer.mean == 1.23
    assert reward_normalizer.m2 == 4.56
    assert reward_normalizer.count == 7


def test_resume_uses_single_preloaded_checkpoint_for_agent_and_metadata(
    monkeypatch, tmp_path
):
    import esper.runtime as runtime
    import esper.simic.training.vectorized as vectorized
    from esper.leyline import TelemetryEventType

    original_get_task_spec = runtime.get_task_spec
    original_torch_load = torch.load
    load_calls = []
    agent_load_calls = []

    def get_task_spec_mock(name: str):
        spec = original_get_task_spec(name)
        spec.dataloader_defaults["mock"] = True
        return spec

    def torch_load_mock(*args, **kwargs):
        load_calls.append((args, kwargs))
        return original_torch_load(*args, **kwargs)

    def load_from_checkpoint_dict_mock(*args, **kwargs):
        agent_load_calls.append((args, kwargs))
        return object()

    class StubHub:
        def add_backend(self, _backend) -> None:
            return None

        def emit(self, event) -> None:
            if event.event_type == TelemetryEventType.CHECKPOINT_LOADED:
                raise _StopAfterCheckpointLoaded()
            return None

    class StubSharedBatchIterator:
        def __init__(self, *args, **kwargs):
            return None

        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(runtime, "get_task_spec", get_task_spec_mock)
    monkeypatch.setattr(vectorized, "get_hub", lambda: StubHub())
    monkeypatch.setattr(vectorized, "SharedBatchIterator", StubSharedBatchIterator)
    monkeypatch.setattr(vectorized.torch, "load", torch_load_mock)
    monkeypatch.setattr(
        vectorized.PPOAgent,
        "load_from_checkpoint_dict",
        load_from_checkpoint_dict_mock,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "metadata": {
                "n_episodes": 0,
                "batches_completed": 0,
                "n_envs": 1,
                **_obs_normalizer_metadata_for_slots(("r0c1",)),
            }
        },
        checkpoint_path,
    )

    with pytest.raises(_StopAfterCheckpointLoaded):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=1,
            chunk_length=1,
            device="cpu",
            devices=["cpu"],
            task="cifar_baseline",
            slots=["r0c1"],
            use_telemetry=False,
            num_workers=0,
            resume_path=str(checkpoint_path),
            quiet_analytics=True,
        )

    assert len(load_calls) == 1
    assert load_calls[0][0][0] == str(checkpoint_path)
    assert load_calls[0][1]["map_location"] == "cpu"
    assert len(agent_load_calls) == 1
    assert agent_load_calls[0][1]["compile_mode"] == "off"


def test_resume_force_compile_overrides_quiet_analytics(monkeypatch, tmp_path):
    import esper.runtime as runtime
    import esper.simic.training.vectorized as vectorized
    from esper.leyline import TelemetryEventType

    original_get_task_spec = runtime.get_task_spec
    agent_load_calls = []

    def get_task_spec_mock(name: str):
        spec = original_get_task_spec(name)
        spec.dataloader_defaults["mock"] = True
        return spec

    def load_from_checkpoint_dict_mock(*args, **kwargs):
        agent_load_calls.append((args, kwargs))
        return object()

    class StubHub:
        def add_backend(self, _backend) -> None:
            return None

        def emit(self, event) -> None:
            if event.event_type == TelemetryEventType.CHECKPOINT_LOADED:
                raise _StopAfterCheckpointLoaded()
            return None

    class StubSharedBatchIterator:
        def __init__(self, *args, **kwargs):
            return None

        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(runtime, "get_task_spec", get_task_spec_mock)
    monkeypatch.setattr(vectorized, "get_hub", lambda: StubHub())
    monkeypatch.setattr(vectorized, "SharedBatchIterator", StubSharedBatchIterator)
    monkeypatch.setattr(
        vectorized.PPOAgent,
        "load_from_checkpoint_dict",
        load_from_checkpoint_dict_mock,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "metadata": {
                "n_episodes": 0,
                "batches_completed": 0,
                "n_envs": 1,
                **_obs_normalizer_metadata_for_slots(("r0c1",)),
            }
        },
        checkpoint_path,
    )

    with pytest.raises(_StopAfterCheckpointLoaded):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=1,
            chunk_length=1,
            device="cpu",
            devices=["cpu"],
            task="cifar_baseline",
            slots=["r0c1"],
            use_telemetry=False,
            num_workers=0,
            resume_path=str(checkpoint_path),
            quiet_analytics=True,
            force_compile=True,
            compile_mode="reduce-overhead",
        )

    assert len(agent_load_calls) == 1
    assert agent_load_calls[0][1]["compile_mode"] == "reduce-overhead"


def test_resume_rejects_checkpoint_slot_ids_that_do_not_match_runtime_slots(
    tmp_path,
):
    import esper.simic.training.vectorized as vectorized
    from esper.leyline.slot_config import SlotConfig
    from esper.simic.agent import PPOAgent
    from esper.tamiyo.policy.factory import create_policy
    from esper.tamiyo.policy.features import get_feature_size

    checkpoint_slot_config = SlotConfig(slot_ids=("r0c0",))
    policy = create_policy(
        policy_type="lstm",
        state_dim=get_feature_size(checkpoint_slot_config),
        slot_config=checkpoint_slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(policy=policy, slot_config=checkpoint_slot_config, device="cpu")
    checkpoint_path = tmp_path / "checkpoint.pt"
    agent.save(
        checkpoint_path,
        metadata={
            "n_episodes": 0,
            "batches_completed": 0,
            "n_envs": 1,
        },
    )

    with pytest.raises(RuntimeError, match="checkpoint=.*r0c0.*runtime=.*r0c1"):
        vectorized.train_ppo_vectorized(
            n_episodes=1,
            n_envs=1,
            max_epochs=1,
            chunk_length=1,
            device="cpu",
            devices=["cpu"],
            task="cifar_baseline",
            slots=["r0c1"],
            use_telemetry=False,
            num_workers=0,
            resume_path=str(checkpoint_path),
            quiet_analytics=True,
        )
