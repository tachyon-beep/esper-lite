"""Phase 5: rent + alpha-shock wiring invariants.

These tests lock the high-risk reward inputs that come from runtime Kasmina state:
- BaseSlotRent must stop after physical prune (cooldown pays no rent).
- Convex alpha shock must remain well-defined during BLEND_OUT freeze
  (requires_grad toggles must not zero-out param scaling mid-ramp).
"""

from __future__ import annotations

import torch

from esper.kasmina.host import CNNHost, MorphogeneticModel
from esper.leyline.alpha import AlphaMode
from esper.leyline.stages import SeedStage
from esper.simic.training.helpers import compute_rent_and_shock_inputs


def _host_param_count(host: torch.nn.Module) -> int:
    return sum(p.numel() for p in host.parameters() if p.requires_grad)


def test_prune_ramp_shock_invariant_to_blend_out_freeze_and_rent_stops_after_prune() -> None:
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0"])
    model.germinate_seed("conv_light", "seed0", slot="r0c0")

    slot = model.seed_slots["r0c0"]
    assert slot.state is not None

    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)

    # Start from a stable full-amplitude hold.
    slot.set_alpha(1.0)
    slot.state.alpha_controller.alpha = 1.0
    slot.state.alpha_controller.alpha_target = 1.0
    slot.state.alpha_controller.alpha_mode = AlphaMode.HOLD

    host_params = _host_param_count(host)
    prev_alphas = {"r0c0": 0.0}
    prev_params = {"r0c0": 0}

    # Prime prev state (no shock on first observation, since prev_params==0).
    _, shock0 = compute_rent_and_shock_inputs(
        model=model,
        slot_ids=["r0c0"],
        host_params=host_params,
        base_slot_rent_ratio=0.0039,
        prev_slot_alphas=prev_alphas,
        prev_slot_params=prev_params,
    )
    assert shock0 == 0.0
    assert prev_alphas["r0c0"] == 1.0
    assert prev_params["r0c0"] > 0

    # Schedule a 2-tick prune ramp: BLEND_OUT freeze should make active_seed_params==0,
    # but shock scaling must still use stable param counts (not requires_grad).
    ok = slot.schedule_prune(steps=2, reason="policy_prune")
    assert ok is True
    assert slot.active_seed_params == 0

    # Tick 1: alpha decreases (shock must be non-zero).
    slot.step_epoch()
    effective1, shock1 = compute_rent_and_shock_inputs(
        model=model,
        slot_ids=["r0c0"],
        host_params=host_params,
        base_slot_rent_ratio=0.0039,
        prev_slot_alphas=prev_alphas,
        prev_slot_params=prev_params,
    )
    assert effective1 > 0.0
    assert shock1 > 0.0
    assert prev_params["r0c0"] > 0, "Param scaling must not be zeroed during BLEND_OUT"

    # Tick 2: prune completes (seed removed, state retained in PRUNED stage).
    slot.step_epoch()
    assert slot.seed is None
    assert slot.state is not None
    assert slot.state.stage == SeedStage.PRUNED

    # Shock must still apply for the final alpha drop, but rent must be zero post-prune.
    effective2, shock2 = compute_rent_and_shock_inputs(
        model=model,
        slot_ids=["r0c0"],
        host_params=host_params,
        base_slot_rent_ratio=0.0039,
        prev_slot_alphas=prev_alphas,
        prev_slot_params=prev_params,
    )
    assert shock2 > 0.0
    assert effective2 == 0.0

