"""Executable spec tests for SIMPLIFIED rewards.

SIMPLIFIED is a deliberately small, 3-component reward used as a baseline and a
DRL-stable alternative to SHAPED:

1) PBRS stage progression (optional via `disable_pbrs`)
2) Uniform intervention cost on non-WAIT actions
3) Terminal bonus (optional via `disable_terminal_reward`):
   `(val_acc / 100) * 3 + 2 * num_contributing_fossilized`
"""

