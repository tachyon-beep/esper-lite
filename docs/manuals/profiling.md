## Profiling (PPO Training)

This project supports end-to-end profiling of the PPO vectorized training loop via `torch.profiler`.

### Torch Profiler (CPU + CUDA timeline)

Enable trace capture from the training CLI:

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --devices cuda:0 cuda:1 \
  --config-json config.json \
  --gpu-preload \
  --rounds 2000 --envs 16 --num-workers 16 \
  --telemetry-dir ./telemetry --telemetry-level debug \
  --sanctum \
  --torch-profiler \
  --torch-profiler-dir ./profiler_traces/ppo_scale16 \
  --torch-profiler-wait 10 --torch-profiler-warmup 10 \
  --torch-profiler-active 2 --torch-profiler-repeat 1 \
  --torch-profiler-summary
```

Notes:
- Keep `--torch-profiler-active` small (1–3) to avoid multi-hundred-MB traces.
- Profiler “steps” correspond to PPO **inner epochs** (the `epoch 1..max_epochs` loop). With `--torch-profiler-wait 10 --torch-profiler-warmup 10 --torch-profiler-active 2`, the first trace only exports after `10+10+2=22` inner epochs; if you stop earlier you’ll get an empty/near-empty summary and no trace file.
- If you’re profiling with `--gpu-preload`, try `--experimental-gpu-preload-gather` to remove DataLoader collation overhead (experimental).
- By default, traces do **not** record shapes/memory/stacks. Enable only if needed:
  - `--torch-profiler-record-shapes`
  - `--torch-profiler-profile-memory`
  - `--torch-profiler-with-stack` (can explode trace size)

### Viewing Traces

The trace files are `*.pt.trace.json` under `--torch-profiler-dir`.

- **Perfetto (recommended):** open `https://ui.perfetto.dev/` and drag/drop the trace JSON.
- **TensorBoard:** install `tensorboard` and run:

```bash
uv pip install tensorboard
tensorboard --logdir ./profiler_traces/ppo_scale16
```

### CPU Hotspots (Python-level)

For Python overhead (data plumbing, telemetry formatting, etc.), use `cProfile`:

```bash
PYTHONPATH=src python -m cProfile -o prof_ppo.pstats -m esper.scripts.train ppo \
  --config-json config.json --rounds 1 --envs 16 --episode-length 5 --no-tui
python - <<'PY'
import pstats
pstats.Stats("prof_ppo.pstats").sort_stats("cumtime").print_stats(40)
PY
```
