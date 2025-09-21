---
title: REPRODUCIBILITY CHECKLIST
split_mode: consolidated
appendix: "D"
coauthors:
  - John Morrissey
  - Codex CLI (OpenAI)
---

This checklist captures the key information required to reproduce the experiments conceptually without implementation details.

## D.1 Datasets & Splits

- Datasets: XOR (synthetic), make_moons (synthetic), CIFAR‑10 (standard)
- Split policies: train/validation/test with fixed random seeds
- Preprocessing: standardisation/normalisation as applicable

## D.2 Conditions

- Baseline (Frozen), Full fine‑tune, Adapter fine‑tune, Morphogenetic (policy + gates), Morphogenetic (heuristic + gates), Morphogenetic (no gates), Random triggers

## D.3 Measurement Windows

- Steady‑state window after convergence (e.g., last N validation epochs)
- Report mean ± 95% confidence interval over R independent runs

## D.4 Metrics

- Primary: ΔAcc/ΔF1, ΔParams, ΔLatency, ΔMemory
- Safety: Interface drift, isolation violations, gate pass rates, cull/embargo counts
- Policy: trigger precision/recall for useful growth, selection entropy, no‑op ratio

## D.5 Ablations

- Lifecycle gates, blending schedule, growth budgets, policy vs heuristic vs random triggers, seed site count, library size, robustness (trigger bait)

## D.6 Randomisation & Seeds

- Fixed random seed protocol across methods for fairness
- Document seeds used for reported aggregates

## D.7 Reporting

- Tables per dataset as specified (see Section 7.6)
- Plots per Section 9.12 (Pareto, Safety dashboard, Policy metrics)
