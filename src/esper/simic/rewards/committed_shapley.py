"""Committed-Shapley synergy top-up: exact 2^k coalition Shapley over the
committed (FOSSILIZED) slot set, with the design's Goodhart guards.

This is the pure math for the default-OFF reward-credit term (PDR-0010/0012):

    phi(s)    = sum_{S subseteq C\\{s}} |S|!(k-|S|-1)!/k! * (v(S+{s}) - v(S))
    c_paid(s) = v({s}) - v(empty)            # terminal same-time standalone baseline
    gap(s)    = max(0, phi(s) - c_paid(s) - tau)
    raw(s)    = min(cap, scale * gap(s))
    G         = max(0, v(C) - v(empty))
    top_up(s) = raw(s) * min(1, G / sum_raw)  if sum_raw > 0 else 0

All accuracies are PERCENTAGE POINTS (0-100); the trainer's fused-val unpack
produces pp and the tau/cap flags are calibrated in pp. Delivery (normalization
into buffer units + the retro-write) lives at the PPO coordinator seam, not here.

Exact factorial only: k <= 3 is a hard contract (sampled/permutation Shapley is
refused by design — zero estimator variance, nothing for a policy to farm).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import factorial

__all__ = [
    "SlotTopUp",
    "CommittedShapleyResult",
    "CommittedShapleyEnvCredits",
    "compute_committed_shapley_topup",
]

# Exact-factorial contract bound: max_seeds is 3 in the shipped regime and the
# 2^k coalition family rides the terminal fused val pass (<= 8 extra configs).
MAX_EXACT_COALITION_SLOTS = 3


@dataclass(frozen=True)
class SlotTopUp:
    """Per-slot decomposition of the credit (all in percentage points)."""

    phi: float
    c_paid: float
    gap: float
    raw: float
    top_up: float


@dataclass(frozen=True)
class CommittedShapleyResult:
    """Full terminal credit computation for one environment."""

    per_slot: dict[str, SlotTopUp]
    g: float
    sum_raw: float
    clamp_binding: bool
    k: int


@dataclass(frozen=True)
class CommittedShapleyEnvCredits:
    """One env's terminal credits, ready for retro-write delivery.

    Built by the trainer at the terminal fused-val pass; consumed by the PPO
    coordinator at the pre-GAE seam. ``t_f_by_slot`` maps each committed slot
    to the buffer step index of its FOSSILIZE decision (recorded at execution
    time in ``env_state.fossilize_step_records`` — never reconstructed from
    epoch arithmetic). ``coalition_accs`` is the raw 2^k v(S) table the
    Shapley result was computed from — carried through to telemetry so the
    first ON run retires the transient-factorial calibration proxy.
    """

    env_idx: int
    result: CommittedShapleyResult
    t_f_by_slot: dict[str, int]
    coalition_accs: Mapping[frozenset[str], float]
    episode_idx: int | None = None


def compute_committed_shapley_topup(
    coalition_accs: Mapping[frozenset[str], float],
    slots: Sequence[str],
    *,
    scale: float,
    cap: float,
    tau: float,
) -> CommittedShapleyResult:
    """Compute per-slot Shapley top-ups from a complete 2^k coalition table.

    Args:
        coalition_accs: v(S) for EVERY subset S of ``slots`` (keys are
            frozensets of slot ids), in percentage points. A missing subset is
            a caller bug and raises — a silently-zeroed v(S) would corrupt phi.
        slots: the committed (FOSSILIZED) slot ids, k = len(slots) <= 3.
        scale: shapley_synergy_scale (blend weight).
        cap: shapley_synergy_cap (per-seed ceiling on raw, pp).
        tau: shapley_synergy_noise_floor (deadband, pp).
    """
    if scale < 0.0 or cap < 0.0 or tau < 0.0:
        raise ValueError(
            f"shapley_synergy parameters must be non-negative: "
            f"scale={scale}, cap={cap}, tau={tau}"
        )
    k = len(slots)
    if k > MAX_EXACT_COALITION_SLOTS:
        raise ValueError(
            f"committed-coalition Shapley is exact-factorial only (k <= "
            f"{MAX_EXACT_COALITION_SLOTS}); got k={k}. Sampled permutation "
            f"Shapley is refused by design."
        )
    slot_set = frozenset(slots)
    if len(slot_set) != k:
        raise ValueError(f"duplicate slot ids in committed set: {slots!r}")

    # Fail loud on an incomplete coalition table (direct indexing, no defaults).
    for r in range(k + 1):
        for combo in combinations(sorted(slot_set), r):
            if frozenset(combo) not in coalition_accs:
                raise ValueError(
                    f"missing coalition v({set(combo) or '{}'}) in coalition_accs; "
                    f"the 2^k table must be complete"
                )

    v_empty = coalition_accs[frozenset()]
    g = max(0.0, coalition_accs[slot_set] - v_empty)

    if k == 0:
        return CommittedShapleyResult(
            per_slot={}, g=g, sum_raw=0.0, clamp_binding=False, k=0
        )

    k_fact = factorial(k)
    per_slot: dict[str, SlotTopUp] = {}
    for s in slots:
        others = sorted(slot_set - {s})
        phi = 0.0
        for r in range(len(others) + 1):
            weight = factorial(r) * factorial(k - r - 1) / k_fact
            for combo in combinations(others, r):
                coalition = frozenset(combo)
                phi += weight * (
                    coalition_accs[coalition | {s}] - coalition_accs[coalition]
                )
        c_paid = coalition_accs[frozenset({s})] - v_empty
        gap = max(0.0, phi - c_paid - tau)
        raw = min(cap, scale * gap)
        per_slot[s] = SlotTopUp(phi=phi, c_paid=c_paid, gap=gap, raw=raw, top_up=0.0)

    sum_raw = sum(entry.raw for entry in per_slot.values())
    clamp_binding = sum_raw > 0.0 and g < sum_raw
    factor = min(1.0, g / sum_raw) if sum_raw > 0.0 else 0.0
    per_slot = {
        s: SlotTopUp(
            phi=entry.phi,
            c_paid=entry.c_paid,
            gap=entry.gap,
            raw=entry.raw,
            top_up=entry.raw * factor,
        )
        for s, entry in per_slot.items()
    }
    return CommittedShapleyResult(
        per_slot=per_slot, g=g, sum_raw=sum_raw, clamp_binding=clamp_binding, k=k
    )
