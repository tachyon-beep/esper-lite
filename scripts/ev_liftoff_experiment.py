#!/usr/bin/env python
"""FAIR multi-epoch recurrent PPO EV-liftoff experiment driver.

Runs ONE arm of the K=4 vs K=1 comparison on REAL CIFAR (task=cifar_baseline).
The ONLY independent variable is K (recurrent_n_epochs); everything else is held
identical across arms via the constants below. The shared value-head init gain=0.1
treatment is applied at the SOURCE (factored_lstm.py:511) BEFORE launching either
arm -- this script does not touch it.

Launch K=4 on cuda:0 and K=1 on cuda:1 IN PARALLEL with identical seed/n_episodes.
EV (data.explained_variance, pre-update, closed-loop against regenerated rollouts)
lands in <telemetry_dir>/telemetry_*/events.jsonl under PPO_UPDATE_COMPLETED
(emitted in esper.simic.telemetry.emitters around lines 971-986).

Usage:
  PYTHONPATH=src uv run python scripts/ev_liftoff_experiment.py \
      --k 4 --device cuda:0 --telemetry-dir /tmp/ev_k4 --n-episodes 200
"""
from __future__ import annotations

import argparse
import math
import sys

from esper.simic.training.vectorized import train_ppo_vectorized

# --- HELD-CONSTANT across both arms (only K and device differ) ---
SEED = 42
N_ENVS = 4
MAX_EPOCHS = 150              # chunk_length auto-matches when left None
LR = 3e-4
CLIP_RATIO = 0.2
GAMMA = 0.995
GAE_LAMBDA = 0.98
VALUE_COEF = 0.5
VALUE_WARMUP_BATCHES = 0
ENTROPY_COEF_START = 0.1
ENTROPY_COEF_END = 0.01
ENTROPY_COEF_MIN = 0.01
LSTM_HIDDEN_DIM = 512
MAX_GRAD_NORM = 5.0
SLOTS = ["r0c0"]
REWARD_MODE = "shaped"
REWARD_FAMILY = "contribution"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EV-liftoff K=4 vs K=1 experiment arm")
    p.add_argument("--k", type=int, required=True,
                   help="recurrent_n_epochs: 4 for treatment arm, 1 for baseline arm")
    p.add_argument("--device", type=str, required=True,
                   help="CUDA device, e.g. cuda:0 (K=4) or cuda:1 (K=1)")
    p.add_argument("--telemetry-dir", type=str, required=True,
                   help="Per-arm telemetry root; events.jsonl is written under telemetry_*/")
    p.add_argument("--n-episodes", type=int, default=200,
                   help="PPO-update count for LSTM (==n_episodes). 200 primary, 100 fallback.")
    p.add_argument("--task", type=str, default="cifar_baseline",
                   choices=["cifar_baseline", "cifar_impaired"],
                   help="Host substrate. cifar_baseline (~6.4K params, augment) is the default "
                        "and preserves the running arms' behavior; cifar_impaired (77 params, "
                        "rescue) is the degraded-host arm with higher contribution SNR.")
    p.add_argument("--gpu-preload", action="store_true",
                   help="Preload CIFAR onto the GPU and augment on-GPU (eliminates the CPU "
                        "dataloader/H2D bottleneck). Applied identically to both arms so the "
                        "K=4 vs K=1 comparison stays fair.")
    args = p.parse_args(argv)
    if args.k < 1:
        p.error("--k must be >= 1 (K=1 single-epoch, K>1 internal multi-epoch)")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    n_episodes = args.n_episodes
    # entropy anneal spans the whole run; total_train_steps pins the penalty schedule
    # identically across arms (==n_episodes since ppo_updates_per_batch=1 for LSTM).
    entropy_anneal_episodes = n_episodes
    total_train_steps = n_episodes

    print(
        f"[ev-liftoff] arm K={args.k} device={args.device} "
        f"n_episodes={n_episodes} seed={SEED} task={args.task} "
        f"entropy_anneal_episodes={entropy_anneal_episodes} "
        f"entropy_anneal_steps={math.ceil(entropy_anneal_episodes / N_ENVS)} "
        f"total_train_steps={total_train_steps} gpu_preload={args.gpu_preload} "
        f"telemetry_dir={args.telemetry_dir}",
        flush=True,
    )

    agent, history = train_ppo_vectorized(
        n_episodes=n_episodes,
        n_envs=N_ENVS,
        max_epochs=MAX_EPOCHS,
        device=args.device,
        task=args.task,
        use_telemetry=True,
        lr=LR,
        clip_ratio=CLIP_RATIO,
        entropy_coef_start=ENTROPY_COEF_START,
        entropy_coef_end=ENTROPY_COEF_END,
        entropy_coef_min=ENTROPY_COEF_MIN,
        entropy_anneal_episodes=entropy_anneal_episodes,
        value_coef=VALUE_COEF,
        value_warmup_batches=VALUE_WARMUP_BATCHES,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ppo_updates_per_batch=1,            # mandatory for LSTM; K is internal
        recurrent_n_epochs=args.k,          # THE independent variable
        total_train_steps=total_train_steps,
        seed=SEED,
        amp=True,
        amp_dtype="bfloat16",
        max_grad_norm=MAX_GRAD_NORM,
        compile_mode="default",
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        chunk_length=None,                  # auto-matches max_epochs
        slots=SLOTS,
        reward_mode=REWARD_MODE,
        reward_family=REWARD_FAMILY,
        gpu_preload=args.gpu_preload,
        gpu_preload_augment=args.gpu_preload,   # fresh per-batch aug on GPU (matches CPU-path semantics)
        telemetry_dir=args.telemetry_dir,
        group_id=f"ev_liftoff_{args.task}_k{args.k}",
    )

    print(f"[ev-liftoff] arm K={args.k} DONE -- {len(history)} update records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
