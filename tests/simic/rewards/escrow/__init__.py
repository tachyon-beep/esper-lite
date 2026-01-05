"""ESCROW reward contract tests.

This package is intentionally structured as a "spec harness" rather than a
grab-bag of unit tests. The goal is that a human can read these tests and say:

  "Yes — this is the escrow reward, and it matches my mental model."

We split coverage into three layers:
1) Pure escrow-branch tests (no training loop state)
2) Stateful ledger invariants (credit continuity, delta clipping convergence)
3) Wiring/integration checks (stable accuracy timing, terminal clawback, slot targeting)
"""

