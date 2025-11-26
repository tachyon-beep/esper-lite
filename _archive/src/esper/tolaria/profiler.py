"""Lightweight profiler hooks for Tolaria (Chrome traces)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch


@contextmanager
def maybe_profile(
    enabled: bool,
    *,
    trace_dir: str,
    active_steps: int = 50,  # unused in simple export mode
    name: str = "tolaria-epoch",
) -> Iterator[None]:
    if not enabled or not hasattr(torch, "profiler"):
        yield
        return
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():  # pragma: no cover - depends on GPU
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        out_file = trace_path / f"{name}-{timestamp}.json"
        canonical = trace_path / f"{name}.json"
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            yield
        # Export after the profiled block
        prof.export_chrome_trace(str(out_file))
        try:
            canonical.write_bytes(out_file.read_bytes())
        except Exception:
            pass
    except Exception:  # pragma: no cover - profiler optional
        yield


__all__ = ["maybe_profile"]
