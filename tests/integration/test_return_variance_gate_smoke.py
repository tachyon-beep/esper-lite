"""Integration smoke: the value-free Stage-0 gate survives the REAL train_ppo_vectorized
path (the strict ``_PPO_MEAN_REDUCED_METRICS`` reducer whitelist).

Every other Stage-0 test is per-layer or ``update()``-level and BYPASSES the reducer (the
returned metrics dict is inspected directly). The reducer's handling of the three new gate
keys is the one seam a real (GPU) run hits first — an emitted key with no declared reducer
is a hard KeyError. This drives a tiny CPU training run to close that gap for pennies.
"""

from dataclasses import replace

import pytest


@pytest.fixture
def mock_cifar_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the smoke onto a hermetic mock dataset (mirrors test_sparse_training)."""
    import esper.runtime as runtime

    original_get_task_spec = runtime.get_task_spec

    def get_mock_task_spec(name: str):
        spec = original_get_task_spec(name)
        return replace(
            spec,
            dataloader_defaults={**spec.dataloader_defaults, "mock": True},
        )

    monkeypatch.setattr(runtime, "get_task_spec", get_mock_task_spec)


@pytest.mark.slow
def test_return_variance_gate_survives_vectorized_reducer(mock_cifar_task: None) -> None:
    """return_variance_telemetry=True through the full loop: the gate keys reach the
    aggregated history without a reducer KeyError, on the HRA-OFF (control) leg."""
    pytest.importorskip("torch")

    from esper.simic.rewards import RewardMode
    from esper.simic.training.vectorized import train_ppo_vectorized

    _, history = train_ppo_vectorized(
        n_episodes=2,
        n_envs=2,
        max_epochs=3,
        task="cifar_minimal",
        device="cpu",
        devices=["cpu"],
        num_workers=0,
        batch_size_per_env=8,
        compile_mode="off",
        reward_mode=RewardMode.SHAPED,  # CONTRIBUTION family — the gate's scope
        slots=["r0c1"],
        use_telemetry=False,
        hra_value_decomposition=False,  # Stage-2 OFF: the control-run posture
        return_variance_telemetry=True,
    )

    assert len(history) > 0
    all_keys = set().union(*(dict(h).keys() for h in history))
    assert "return_var_cf_share" in all_keys, (
        f"value-free gate key missing from aggregated history — reducer dropped it. "
        f"keys={sorted(all_keys)}"
    )
