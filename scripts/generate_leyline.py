#!/usr/bin/env python3
"""Regenerate Leyline Python bindings from contracts/leyline/leyline.proto.

Requires `grpcio-tools` in the environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    proto_dir = repo_root / "contracts" / "leyline"
    out_dir = repo_root / "src" / "esper" / "leyline" / "_generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    proto_file = proto_dir / "leyline.proto"
    if not proto_file.exists():
        print(f"Error: {proto_file} not found", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={out_dir}",
        f"--pyi_out={out_dir}",
        str(proto_file),
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"protoc failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    print("Leyline bindings generated in", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

