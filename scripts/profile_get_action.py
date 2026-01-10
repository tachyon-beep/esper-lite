#!/usr/bin/env python3
"""Profile get_action() to identify 16x slowdown after op/value consistency fix.

Hypothesis areas:
1. Exposing lstm_out in forward() preventing graph optimization
2. Tensor operations (unsqueeze/squeeze) in deterministic mode
3. torch.compile cache invalidation
4. Memory bandwidth from returning extra tensor
5. MaskedCategorical validation causing sync in inference_mode

Run: uv run python scripts/profile_get_action.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Any

import torch
from torch.profiler import profile, ProfilerActivity

from esper.tamiyo.networks import FactoredRecurrentActorCritic
from esper.tamiyo.policy.action_masks import MaskedCategorical
from esper.leyline import NUM_OPS
from esper.leyline.slot_config import SlotConfig


def create_test_inputs(
    batch_size: int, device: torch.device, slot_config: SlotConfig
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Create realistic test inputs for get_action()."""
    from esper.tamiyo.policy.features import get_feature_size

    state_dim = get_feature_size(slot_config)
    num_slots = slot_config.num_slots

    state = torch.randn(batch_size, state_dim, device=device)
    blueprint_indices = torch.randint(0, 13, (batch_size, num_slots), device=device)

    from esper.leyline import (
        NUM_ALPHA_TARGETS,
        NUM_ALPHA_SPEEDS,
        NUM_ALPHA_CURVES,
        NUM_STYLES,
        NUM_TEMPO,
        NUM_BLUEPRINTS,
    )

    # Realistic masks with correct sizes from leyline
    masks = {
        "slot": torch.ones(batch_size, num_slots, dtype=torch.bool, device=device),
        "blueprint": torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
        "style": torch.ones(batch_size, NUM_STYLES, dtype=torch.bool, device=device),
        "tempo": torch.ones(batch_size, NUM_TEMPO, dtype=torch.bool, device=device),
        "alpha_target": torch.ones(batch_size, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
        "alpha_speed": torch.ones(batch_size, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
        "alpha_curve": torch.ones(batch_size, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
        "op": torch.ones(batch_size, NUM_OPS, dtype=torch.bool, device=device),
    }

    return state, blueprint_indices, masks


def benchmark_get_action(
    network: FactoredRecurrentActorCritic,
    state: torch.Tensor,
    blueprint_indices: torch.Tensor,
    masks: dict[str, torch.Tensor],
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
    deterministic: bool,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark get_action() with proper GPU timing."""
    device = state.device

    # Warmup
    for _ in range(num_warmup):
        _ = network.get_action(
            state, blueprint_indices, hidden,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            style_mask=masks["style"],
            tempo_mask=masks["tempo"],
            alpha_target_mask=masks["alpha_target"],
            alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"],
            op_mask=masks["op"],
            deterministic=deterministic,
            return_op_logits=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()

        _result = network.get_action(
            state, blueprint_indices, hidden,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            style_mask=masks["style"],
            tempo_mask=masks["tempo"],
            alpha_target_mask=masks["alpha_target"],
            alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"],
            op_mask=masks["op"],
            deterministic=deterministic,
            return_op_logits=True,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5 * 1000,
    }


def profile_get_action_detailed(
    network: FactoredRecurrentActorCritic,
    state: torch.Tensor,
    blueprint_indices: torch.Tensor,
    masks: dict[str, torch.Tensor],
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
    deterministic: bool,
) -> profile:
    """Profile get_action() with stack traces."""
    device = state.device

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Warmup
    for _ in range(5):
        _ = network.get_action(
            state, blueprint_indices, hidden,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            style_mask=masks["style"],
            tempo_mask=masks["tempo"],
            alpha_target_mask=masks["alpha_target"],
            alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"],
            op_mask=masks["op"],
            deterministic=deterministic,
            return_op_logits=True,
        )

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(10):
            _ = network.get_action(
                state, blueprint_indices, hidden,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
                deterministic=deterministic,
                return_op_logits=True,
            )

    result: profile = prof
    return result


def main() -> None:
    print("=" * 70)
    print("Profiling get_action() - Investigating 16x Slowdown")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    slot_config = SlotConfig.default()
    batch_size = 8  # Typical vectorized env batch

    # Create network
    from esper.tamiyo.policy.features import get_feature_size
    state_dim = get_feature_size(slot_config)

    network = FactoredRecurrentActorCritic(
        state_dim=state_dim,
        slot_config=slot_config,
    ).to(device)
    network.eval()

    # Create inputs
    state, blueprint_indices, masks = create_test_inputs(batch_size, device, slot_config)
    hidden = network.get_initial_hidden(batch_size, device)

    print(f"\nBatch size: {batch_size}")
    print(f"State dim: {state_dim}")
    print(f"Device: {device}")

    # Test 1: Baseline with MaskedCategorical.validate = True (current default)
    print("\n" + "=" * 70)
    print("TEST 1: Validation ENABLED (current default)")
    print("=" * 70)

    MaskedCategorical.validate = True

    print("\nStochastic mode (rollout hot path):")
    result = benchmark_get_action(network, state, blueprint_indices, masks, hidden, deterministic=False)
    print(f"  Mean: {result['mean_ms']:.3f}ms, Std: {result['std_ms']:.3f}ms")
    stochastic_validate_on = result['mean_ms']

    print("\nDeterministic mode (bootstrap):")
    result = benchmark_get_action(network, state, blueprint_indices, masks, hidden, deterministic=True)
    print(f"  Mean: {result['mean_ms']:.3f}ms, Std: {result['std_ms']:.3f}ms")
    deterministic_validate_on = result['mean_ms']

    # Test 2: With validation disabled
    print("\n" + "=" * 70)
    print("TEST 2: Validation DISABLED")
    print("=" * 70)

    MaskedCategorical.validate = False

    print("\nStochastic mode (rollout hot path):")
    result = benchmark_get_action(network, state, blueprint_indices, masks, hidden, deterministic=False)
    print(f"  Mean: {result['mean_ms']:.3f}ms, Std: {result['std_ms']:.3f}ms")
    stochastic_validate_off = result['mean_ms']

    print("\nDeterministic mode (bootstrap):")
    result = benchmark_get_action(network, state, blueprint_indices, masks, hidden, deterministic=True)
    print(f"  Mean: {result['mean_ms']:.3f}ms, Std: {result['std_ms']:.3f}ms")
    deterministic_validate_off = result['mean_ms']

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Validation Impact")
    print("=" * 70)
    print("\nStochastic mode:")
    print(f"  Validate ON:  {stochastic_validate_on:.3f}ms")
    print(f"  Validate OFF: {stochastic_validate_off:.3f}ms")
    print(f"  Speedup:      {stochastic_validate_on / stochastic_validate_off:.2f}x")

    print("\nDeterministic mode:")
    print(f"  Validate ON:  {deterministic_validate_on:.3f}ms")
    print(f"  Validate OFF: {deterministic_validate_off:.3f}ms")
    print(f"  Speedup:      {deterministic_validate_on / deterministic_validate_off:.2f}x")

    # Test 3: Detailed profiling to find specific bottlenecks
    print("\n" + "=" * 70)
    print("TEST 3: Detailed Profiling (Validation ON, Stochastic)")
    print("=" * 70)

    MaskedCategorical.validate = True
    prof = profile_get_action_detailed(network, state, blueprint_indices, masks, hidden, deterministic=False)

    print("\n=== CPU Time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    if device.type == "cuda":
        print("\n=== CUDA Time ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # Test 4: Profile the forward() sampling path specifically
    print("\n" + "=" * 70)
    print("TEST 4: Isolate forward() op sampling overhead")
    print("=" * 70)

    # Time just forward() with and without the op sampling
    state_3d = state.unsqueeze(1)
    bp_3d = blueprint_indices.unsqueeze(1)
    masks_3d = {k: v.unsqueeze(1) for k, v in masks.items()}

    MaskedCategorical.validate = False

    # Warmup
    for _ in range(10):
        with torch.inference_mode():
            _ = network.forward(
                state_3d, bp_3d, hidden,
                slot_mask=masks_3d["slot"],
                blueprint_mask=masks_3d["blueprint"],
                style_mask=masks_3d["style"],
                tempo_mask=masks_3d["tempo"],
                alpha_target_mask=masks_3d["alpha_target"],
                alpha_speed_mask=masks_3d["alpha_speed"],
                alpha_curve_mask=masks_3d["alpha_curve"],
                op_mask=masks_3d["op"],
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.inference_mode():
            _output = network.forward(
                state_3d, bp_3d, hidden,
                slot_mask=masks_3d["slot"],
                blueprint_mask=masks_3d["blueprint"],
                style_mask=masks_3d["style"],
                tempo_mask=masks_3d["tempo"],
                alpha_target_mask=masks_3d["alpha_target"],
                alpha_speed_mask=masks_3d["alpha_speed"],
                alpha_curve_mask=masks_3d["alpha_curve"],
                op_mask=masks_3d["op"],
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    forward_mean = sum(times) / len(times) * 1000
    print(f"\nforward() only: {forward_mean:.3f}ms")
    print(f"get_action() overhead: {stochastic_validate_off - forward_mean:.3f}ms")

    # Restore validation
    MaskedCategorical.validate = True


if __name__ == "__main__":
    main()
