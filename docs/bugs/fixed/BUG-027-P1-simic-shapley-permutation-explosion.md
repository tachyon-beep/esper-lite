# BUG-027: Shapley computation materializes all permutations (factorial blowup)

- **Title:** `CounterfactualEngine.compute_shapley_values()` builds `list(permutations(...))` (and even `len(list(permutations(...)))`) before sampling, causing factorial time/memory and ignoring `shapley_samples`
- **Context:** Simic counterfactual attribution; Shapley estimates at episode end
- **Impact:** P1 – performance timebomb / self-DoS. For `n>=10` active slots, `n!` permutation materialization can stall or OOM a training run even though we later cap to `n_perms<=100`.
- **Environment:** HEAD @ workspace; any run that calls `CounterfactualEngine.compute_shapley_values()` with multi-slot counterfactual matrices
- **Reproduction Steps:**
  1. In Python, construct a `CounterfactualMatrix` whose `configs[0].slot_ids` has length `n>=10`.
  2. Call `CounterfactualEngine(config).compute_shapley_values(matrix)`.
  3. Observe the process spends significant time/memory building the full permutation list before it slices to `n_perms`.
- **Expected Behavior:** Runtime should be bounded by `CounterfactualConfig.shapley_samples` (or another explicit cap) and should never materialize all `n!` permutations.
- **Observed Behavior:** `compute_shapley_values()` materializes all permutations twice (`len(list(permutations(...)))` and `perms = list(permutations(...))`), then shuffles and slices.
- **Logs/Telemetry:** Typically manifests as a hang or OOM near episode end when Shapley is computed.
- **Hypotheses:** Classic “sample permutations but accidentally materialized all permutations” implementation drift; conflicts with existing `_generate_shapley_configs()` sampling approach.
- **Fix Plan:** Replace factorial materialization with direct sampling:
  - Sample `n_perms = self.config.shapley_samples` random permutations via `perm = list(range(n)); random.shuffle(perm)` (optionally antithetic pairing).
  - Remove `len(list(permutations(...)))` and `list(permutations(...))` entirely.
  - (Optional) Align Shapley computation with the same sampling scheme used in `_generate_shapley_configs()`.
- **Validation Plan:**
  - Unit test that `compute_shapley_values()` completes quickly for `n=10` (or that it does not call `itertools.permutations` in a way that materializes all results).
  - Smoke: run PPO with `max_slots>=10` and confirm episode-end Shapley finishes in bounded time.
- **Status:** Open
- **Links:**
  - Offending code: `src/esper/simic/attribution/counterfactual.py:325`

