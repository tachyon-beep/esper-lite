#!/usr/bin/env python3
"""Regenerate Leyline protobuf bindings.

Executes `grpc_tools.protoc` against `contracts/leyline/leyline.proto` and
writes Python outputs to `src/esper/leyline/_generated/`.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "contracts" / "leyline"
PROTO_PATH = CONTRACT_DIR / "leyline.proto"
OUTPUT_DIR = ROOT / "src" / "esper" / "leyline" / "_generated"


def main() -> int:
    if not PROTO_PATH.exists():
        print(f"Proto file not found: {PROTO_PATH}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={CONTRACT_DIR}",
        f"--python_out={OUTPUT_DIR}",
        f"--pyi_out={OUTPUT_DIR}",
        PROTO_PATH.name,
    ]

    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=CONTRACT_DIR,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout.decode())
        return result.returncode

    target = OUTPUT_DIR / "leyline_pb2.py"
    if target.exists():
        content = target.read_text(encoding="utf-8")
        lines = content.splitlines()
        head = lines[:2]
        if "# pylint: skip-file" not in head:
            insert_at = 1 if lines and lines[0].startswith("# -*- coding") else 0
            lines.insert(insert_at, "# pylint: skip-file")
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
