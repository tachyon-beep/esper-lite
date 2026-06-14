"""Parity gates for P1-VEC (forced-step vectorization) and P1-QLOOP (batched Q-values).

Both are sync-removal optimizations on telemetry-feeding code; they must be numerically
identical to the serial loops they replace.
"""

import torch

from esper.leyline import NUM_OPS
from esper.leyline.slot_config import SlotConfig
from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic


def _forced_count_old(forced_actions: torch.Tensor, step_counts: list[int]) -> int:
    """The pre-P1-VEC serial reduction (num_envs syncs)."""
    total = 0
    for env_id in range(len(step_counts)):
        n = step_counts[env_id]
        if n > 0:
            total += int(forced_actions[env_id, :n].sum().item())
    return total


def _forced_count_new(forced_actions: torch.Tensor, step_counts: list[int]) -> int:
    """The P1-VEC vectorized reduction (1 sync)."""
    max_steps = forced_actions.shape[1]
    step_counts_t = torch.as_tensor(step_counts, device=forced_actions.device, dtype=torch.long)
    step_valid = (
        torch.arange(max_steps, device=forced_actions.device)[None, :] < step_counts_t[:, None]
    )
    return int((forced_actions & step_valid).sum().item())


def test_p1vec_forced_count_integer_equality():
    g = torch.Generator().manual_seed(0)
    for num_envs, max_steps in [(5, 20), (12, 150), (1, 7), (8, 1)]:
        forced = torch.rand(num_envs, max_steps, generator=g) > 0.5
        # Ragged step counts including 0 and full-length, padded region must not count.
        step_counts = [int(torch.randint(0, max_steps + 1, (1,), generator=g)) for _ in range(num_envs)]
        old = _forced_count_old(forced, step_counts)
        new = _forced_count_new(forced, step_counts)
        assert old == new, f"forced_count mismatch envs={num_envs} steps={max_steps}: {old} != {new}"


def test_p1vec_padding_beyond_step_counts_excluded():
    # Forced flags set in the PADDED region (t >= step_counts) must be ignored by both.
    forced = torch.zeros(3, 10, dtype=torch.bool)
    forced[:, 8:] = True  # all padding region
    step_counts = [5, 3, 0]
    assert _forced_count_old(forced, step_counts) == 0
    assert _forced_count_new(forced, step_counts) == 0


def test_p1qloop_batched_compute_value_matches_loop():
    torch.manual_seed(1)
    slot_config = SlotConfig.default()
    policy = FactoredRecurrentActorCritic(state_dim=64, slot_config=slot_config)
    policy.eval()
    net = policy  # FactoredRecurrentActorCritic owns _compute_value directly

    hidden_dim = policy.lstm_hidden_dim
    lstm_out = torch.randn(1, 1, hidden_dim)

    with torch.no_grad():
        # Old: serial per-op.
        looped = torch.empty(NUM_OPS)
        for op_idx in range(NUM_OPS):
            op_tensor = torch.tensor([[op_idx]], dtype=torch.long)
            looped[op_idx] = net._compute_value(lstm_out, op_tensor).squeeze()

        # New: one batched call.
        op_indices = torch.arange(NUM_OPS, dtype=torch.long).reshape(NUM_OPS, 1)
        lstm_out_rep = lstm_out.expand(NUM_OPS, 1, -1).contiguous()
        batched = net._compute_value(lstm_out_rep, op_indices).reshape(NUM_OPS)

    assert batched.shape == (NUM_OPS,)
    torch.testing.assert_close(batched, looped, rtol=1e-5, atol=1e-6)
