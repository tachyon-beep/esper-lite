#!/usr/bin/env python3
"""Benchmark Tolaria compile warm-up vs steady state."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=3, help='Total epochs to run (>= compile warm-up st eps + steady-state).')
    parser.add_argument('--warmup-epochs', type=int, default=1, help='Number of eager epochs before enabling compile.')
    parser.add_argument('--output', type=Path, default=Path('docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/wp100_compile_multi_epoch/compile_bench.json'))
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def build_trainer(device: torch.device) -> tuple[nn.Module, SGD, DataLoader]:
    torch.manual_seed(1337)
    model = nn.Linear(16, 4).to(device)
    optimizer = SGD(model.parameters(), lr=0.05)
    inputs = torch.randn(128, 16)
    targets = torch.randint(0, 4, (128,))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    return model, optimizer, loader

def run_epoch(model: nn.Module, optimizer: SGD, loader: DataLoader, device: torch.device, compiled=False):
    criterion = nn.CrossEntropyLoss()
    if compiled:
        step_fn = torch.compile(lambda x, y: (model(x), y), mode='reduce-overhead', dynamic=False)
    else:
        step_fn = lambda x, y: (model(x), y)
    start = time.time()
    model.train()
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs, tgt = step_fn(inputs, targets)
        loss = criterion(outputs, tgt)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    return (time.time() - start) * 1000.0

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, optimizer, loader = build_trainer(device)
    results = {'warmup_epochs': args.warmup_epochs, 'epochs': args.epochs, 'device': str(device), 'ms_per_epoch': []}
    compiled = False
    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs:
            compiled = True
        latency_ms = run_epoch(model, optimizer, loader, device, compiled=compiled)
        results['ms_per_epoch'].append({'epoch': epoch, 'compiled': compiled, 'latency_ms': latency_ms})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
if __name__ == '__main__':
    main()
