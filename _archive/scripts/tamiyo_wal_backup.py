#!/usr/bin/env python3
"""Utility for backing up and restoring Tamiyo field-report persistence files."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_DATA_DIR = Path("var/tamiyo")
DEFAULT_WAL = DEFAULT_DATA_DIR / "field_reports.log"
DEFAULT_RETRY_INDEX = DEFAULT_DATA_DIR / "field_reports.index.json"
DEFAULT_WINDOWS = DEFAULT_DATA_DIR / "field_reports.windows.json"


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_backup(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = Path(args.dest).expanduser().resolve() / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    for source in (args.wal, args.retry_index, args.windows):
        path = Path(source).expanduser().resolve()
        if path.exists():
            _copy_if_exists(path, backup_dir / path.name)

    print(f"Backup created at {backup_dir}")


def run_restore(args: argparse.Namespace) -> None:
    backup_dir = Path(args.backup).expanduser().resolve()
    if not backup_dir.exists() or not backup_dir.is_dir():
        raise SystemExit(f"Backup directory not found: {backup_dir}")

    dest_dir = Path(args.dest).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename in (args.wal.name, args.retry_index.name, args.windows.name):
        source = backup_dir / filename
        if source.exists():
            _copy_if_exists(source, dest_dir / filename)

    print(f"Restored files to {dest_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tamiyo WAL backup/restore helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backup_parser = subparsers.add_parser("backup", help="Create a new backup")
    backup_parser.add_argument("--wal", default=DEFAULT_WAL, type=Path)
    backup_parser.add_argument("--retry-index", default=DEFAULT_RETRY_INDEX, type=Path)
    backup_parser.add_argument("--windows", default=DEFAULT_WINDOWS, type=Path)
    backup_parser.add_argument(
        "--dest",
        default=DEFAULT_DATA_DIR / "backups",
        type=Path,
        help="Directory where backups are stored",
    )
    backup_parser.set_defaults(func=run_backup)

    restore_parser = subparsers.add_parser("restore", help="Restore files from a backup")
    restore_parser.add_argument("backup", type=Path, help="Path to an existing backup directory")
    restore_parser.add_argument("--wal", default=DEFAULT_WAL, type=Path)
    restore_parser.add_argument("--retry-index", default=DEFAULT_RETRY_INDEX, type=Path)
    restore_parser.add_argument("--windows", default=DEFAULT_WINDOWS, type=Path)
    restore_parser.add_argument(
        "--dest",
        default=DEFAULT_DATA_DIR,
        type=Path,
        help="Destination directory for restored files",
    )
    restore_parser.set_defaults(func=run_restore)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
