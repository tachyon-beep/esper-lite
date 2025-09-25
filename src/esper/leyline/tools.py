"""Leyline tooling: bindings generator.

Console entry: `esper-leyline-generate`
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Leyline Python bindings from .proto")
    p.add_argument(
        "--proto-dir",
        type=Path,
        default=Path("contracts/leyline"),
        help="Directory containing leyline.proto",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("src/esper/leyline/_generated"),
        help="Output directory for generated Python files",
    )
    p.add_argument(
        "--proto-file",
        type=Path,
        default=None,
        help="Optional explicit proto file path (defaults to <proto-dir>/leyline.proto)",
    )
    return p.parse_args(argv)


def generate_bindings_main(argv: list[str] | None = None) -> int:
    ns = _parse_args(list(argv or sys.argv[1:]))
    proto_dir = ns.proto_dir.resolve()
    out_dir = ns.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    proto_file = (ns.proto_file or (proto_dir / "leyline.proto")).resolve()
    if not proto_file.exists():
        print(f"Error: {proto_file} not found", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{str(proto_dir)}",
        f"--python_out={str(out_dir)}",
        f"--pyi_out={str(out_dir)}",
        str(proto_file),
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"protoc failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    return 0


__all__ = ["generate_bindings_main"]
