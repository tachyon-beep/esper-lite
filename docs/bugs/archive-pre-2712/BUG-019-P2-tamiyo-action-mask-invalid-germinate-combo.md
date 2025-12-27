# BUG-019: Invalid germination combos were representable (blend × alpha_algorithm)

- **Title:** `(blend, alpha_algorithm)` could form incompatible pairs during `GERMINATE` (e.g., `"gated"` + `ADD`)
- **Context:** Simic/Tamiyo factored action space; Kasmina enforces `blend_algorithm_id=="gated" ⇔ alpha_algorithm==GATE`
- **Impact (pre-fix):** P2 – any policy that samples heads independently (random agent, simple MLP baseline, etc.) can emit invalid germination combos that Kasmina rejects (crash or wasted samples).
- **Expected Behavior:** Invalid germination combos should be unrepresentable so “mask == physical legality” holds without policy-specific patches.
- **Observed Behavior (pre-fix):** Per-head masking cannot encode cross-head constraints; invalid pairs slipped through.
- **Status:** Fixed (architecture-level; invalid combos are unrepresentable)

## Root Cause Analysis

This is a **cross-head constraint** in a **factorized action space**:

- Kasmina’s invariant is pairwise:
  - `blend_algorithm_id == "gated"` **requires** `alpha_algorithm == GATE`
  - `alpha_algorithm == GATE` **requires** `blend_algorithm_id == "gated"`
- A per-head mask (one boolean vector per head) cannot express “only these *pairs* are legal” without conditioning on the other head’s sampled value.

## Resolution (Implemented)

We applied the **Composition Principle**: if two heads have rigid dependencies, they are not independent factors.

- **Contract:** Replaced `(blend, alpha_algorithm)` with a single `GerminationStyle` head enumerating only valid pairs.
- **Masking:** `compute_action_masks()` now masks `style` as a single head (no cross-head coupling needed).
- **Policy:** Removed the network-level “compatibility patch” (no conditional masking is required to prevent invalid pairs).
- **Execution:** Simic decodes `style_idx → (blend_algorithm_id, alpha_algorithm)` via lookup tables and calls Kasmina with a valid combination.
- **Defense-in-depth:** Kasmina still enforces the invariant in `SeedSlot.germinate()` (invalid inputs still fail fast if something bypasses the contract).

## Reproduction (Pre-Fix Only)

1. Build masks where `GERMINATE` is legal.
2. Sample `blend="gated"` and `alpha_algorithm=ADD` independently.
3. Call germination → Kasmina raises `ValueError`.

Post-fix, this is impossible because the action space contains only valid germination styles.

## Validation

- `GerminationStyle` mapping only contains valid Kasmina pairs: `tests/leyline/test_factored_actions.py:165`
- Policy forces a default `style` only when it is causally irrelevant (i.e., `op` is neither `GERMINATE` nor `SET_ALPHA_TARGET`): `tests/simic/test_tamiyo_network.py:210`
- End-to-end germination path uses `style_idx` and decodes correctly: `src/esper/simic/training/vectorized.py:2320`

## Links

- Contract (composite head): `src/esper/leyline/factored_actions.py:55`
- Masking (`style` head): `src/esper/tamiyo/policy/action_masks.py:172`
- Policy style head + non-(germinate|set-alpha) override: `src/esper/simic/agent/network.py:171`
- Execution decode (`STYLE_*` lookup tables): `src/esper/simic/training/vectorized.py:2111`
- Kasmina invariant enforcement: `src/esper/kasmina/slot.py:1075`
