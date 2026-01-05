"""Executable spec tests for SHAPED rewards.

The SHAPED reward mode is the default dense shaping used in training. It is a
single function (`compute_contribution_reward`) with many interacting
components (attribution, anti-gaming penalties, PBRS, costs, warnings).

This package mirrors the escrow harness structure: small, explicit examples you
can read and say "yes, this is the shaped reward I think it is".
"""

