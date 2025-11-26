# Kasmina Germination Benchmarks

Artifacts:
- `seed_soak_summary.json` — 100-epoch soak run (alpha → 1.0, zero isolation violations).
- `perf_comparison.json` — seed-enabled vs disabled benchmark (20 epochs). Seed run shows alpha=1.0 and higher training latency due to grafted kernel; baseline remains faster with alpha=0.
