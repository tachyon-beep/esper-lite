#!/usr/bin/env python3
"""Regenerate Leyline protobuf bindings.

Executes `grpc_tools.protoc` against `contracts/leyline/leyline.proto` and
writes Python outputs to `src/esper/leyline/_generated/`.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import sys

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "contracts" / "leyline"
PROTO_PATH = CONTRACT_DIR / "leyline.proto"
OUTPUT_DIR = ROOT / "src" / "esper" / "leyline" / "_generated"


def inject_pylint_skip(target: Path) -> None:
    if not target.exists():
        return
    content = target.read_text(encoding="utf-8")
    lines = content.splitlines()
    head = lines[:2]
    if "# pylint: skip-file" in head:
        return
    insert_at = 1 if lines and lines[0].startswith("# -*- coding") else 0
    lines.insert(insert_at, "# pylint: skip-file")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not PROTO_PATH.exists():
        print(f"Proto file not found: {PROTO_PATH}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    include_dir = resources.files("grpc_tools") / "_proto"

    args = [
        "protoc",
        f"-I{CONTRACT_DIR}",
        f"-I{include_dir}",
        f"--python_out={OUTPUT_DIR}",
        f"--pyi_out={OUTPUT_DIR}",
        str(PROTO_PATH),
    ]

    result = protoc.main(args)
    if result != 0:
        return result

    inject_pylint_skip(OUTPUT_DIR / "leyline_pb2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
