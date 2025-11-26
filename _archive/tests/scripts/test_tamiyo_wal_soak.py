import asyncio
from pathlib import Path

from scripts.tamiyo_wal_soak import run_soak


def test_run_soak(tmp_path):
    result = asyncio.run(run_soak(tmp_path, iterations=10, inject_every=3, strict_reload=False))
    assert result.iterations == 10
    assert result.injections >= 3
    assert result.load_errors_retry_index >= 0
    assert result.load_errors_windows >= 0
