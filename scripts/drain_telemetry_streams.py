#!/usr/bin/env python3
"""Trim Esper telemetry streams to baseline."""

from __future__ import annotations

import argparse
import asyncio
from typing import Iterable

import redis.asyncio as aioredis

from esper.core import EsperSettings


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim Oona telemetry streams")
    parser.add_argument(
        "--streams",
        choices=("normal", "emergency", "both"),
        default="both",
        help="which telemetry streams to trim",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print current lengths without trimming",
    )
    return parser.parse_args(list(argv))


async def _trim(args: argparse.Namespace) -> int:
    settings = EsperSettings()
    redis = aioredis.from_url(settings.redis_url)

    targets: list[str]
    if args.streams == "normal":
        targets = [settings.oona_normal_stream]
    elif args.streams == "emergency":
        targets = [settings.oona_emergency_stream]
    else:
        targets = [settings.oona_normal_stream, settings.oona_emergency_stream]

    lengths: dict[str, int] = {}
    for stream in targets:
        lengths[stream] = await redis.xlen(stream)
        if not args.dry_run and lengths[stream] > 0:
            await redis.xtrim(stream, maxlen=0)
            lengths[stream] = await redis.xlen(stream)

    await redis.aclose()
    print({"streams": targets, "lengths": lengths, "dry_run": args.dry_run})
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv or [])
    return asyncio.run(_trim(args))


if __name__ == "__main__":
    raise SystemExit(main())
