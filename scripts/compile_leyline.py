#!/usr/bin/env python3
"""Compile Leyline protobuf definitions into Python modules."""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

from grpc_tools import protoc


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    proto_dir = repo_root / "contracts" / "leyline"
    output_dir = repo_root / "src" / "esper" / "leyline" / "_generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    include_dir = resources.files("grpc_tools") / "_proto"

    args = [
        "protoc",
        f"-I{proto_dir}",
        f"-I{include_dir}",
        f"--python_out={output_dir}",
        f"--pyi_out={output_dir}",
        str(proto_dir / "leyline.proto"),
    ]

    return protoc.main(args)


if __name__ == "__main__":
    sys.exit(main())
