#!/usr/bin/env python3
"""Generate TypeScript interfaces from Sanctum schema dataclasses.

Usage:
    python scripts/generate_overwatch_types.py > src/esper/karn/overwatch/web/src/types/sanctum.ts

This generator converts Python dataclasses from esper.karn.sanctum.schema
to TypeScript interfaces for the Overwatch web dashboard.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from collections import deque
from datetime import datetime
from typing import Any, Union, get_args, get_origin, get_type_hints

# Add src to path for imports
sys.path.insert(0, "src")

from esper.karn.sanctum.schema import (
    BestRunRecord,
    CounterfactualConfig,
    CounterfactualSnapshot,
    DecisionSnapshot,
    EnvState,
    EventLogEntry,
    GPUStats,
    RewardComponents,
    RunConfig,
    SanctumSnapshot,
    SeedState,
    SystemVitals,
    TamiyoState,
)
from esper.leyline import SeedStage


def python_to_ts_type(py_type: type, depth: int = 0) -> str:
    """Convert Python type annotation to TypeScript type.

    Args:
        py_type: The Python type annotation to convert.
        depth: Recursion depth to prevent infinite loops.

    Returns:
        TypeScript type string.
    """
    if depth > 10:
        return "unknown"

    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle None/NoneType
    if py_type is type(None):
        return "null"

    # Handle UnionType (Python 3.10+ str | None syntax)
    if isinstance(py_type, types.UnionType):
        ts_types = [python_to_ts_type(arg, depth + 1) for arg in get_args(py_type)]
        # Deduplicate (e.g., int|float both become number)
        seen: set[str] = set()
        unique_types = [t for t in ts_types if t not in seen and not seen.add(t)]  # type: ignore[func-returns-value]
        return " | ".join(unique_types)

    # Handle typing.Union (includes Optional)
    if origin is Union:
        ts_types = [python_to_ts_type(arg, depth + 1) for arg in args]
        # Deduplicate (e.g., int|float both become number)
        seen: set[str] = set()
        unique_types = [t for t in ts_types if t not in seen and not seen.add(t)]  # type: ignore[func-returns-value]
        return " | ".join(unique_types)

    # Handle list
    if origin is list:
        if args:
            inner = python_to_ts_type(args[0], depth + 1)
            return f"{inner}[]"
        return "unknown[]"

    # Handle tuple - convert to array in TypeScript
    if origin is tuple:
        if args:
            # Check if it's a homogeneous tuple (e.g., tuple[str, ...])
            if len(args) == 2 and args[1] is ...:
                inner = python_to_ts_type(args[0], depth + 1)
                return f"{inner}[]"
            # Heterogeneous tuple - use array of union
            inner_types = [python_to_ts_type(arg, depth + 1) for arg in args]
            return f"[{', '.join(inner_types)}]"
        return "unknown[]"

    # Handle dict
    if origin is dict:
        if len(args) == 2:
            key_type = python_to_ts_type(args[0], depth + 1)
            val_type = python_to_ts_type(args[1], depth + 1)
            # TypeScript Record requires string or number keys
            if key_type == "number":
                key_type = "number"
            else:
                key_type = "string"
            return f"Record<{key_type}, {val_type}>"
        return "Record<string, unknown>"

    # Handle deque as array
    if origin is deque or (hasattr(py_type, "__origin__") and "deque" in str(py_type)):
        if args:
            inner = python_to_ts_type(args[0], depth + 1)
            return f"{inner}[]"
        return "number[]"

    # Primitive mappings
    type_map: dict[type, str] = {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean",
        datetime: "string",  # ISO string
        type(None): "null",
        Any: "unknown",
    }

    if py_type in type_map:
        return type_map[py_type]

    # Check if it's an enum (IntEnum)
    if hasattr(py_type, "__members__"):
        return py_type.__name__

    # Check if it's a dataclass we know
    if dataclasses.is_dataclass(py_type):
        return py_type.__name__

    # Handle forward references (strings)
    if isinstance(py_type, str):
        # Strip quotes from forward references
        return py_type.strip('"').strip("'")

    # Fallback
    return "unknown"


def generate_enum(enum_cls: type) -> str:
    """Generate TypeScript type for Python IntEnum.

    Args:
        enum_cls: The enum class to convert.

    Returns:
        TypeScript union type definition.
    """
    members = [f'"{m.name}"' for m in enum_cls]
    return f"export type {enum_cls.__name__} = {' | '.join(members)};"


def generate_interface(cls: type) -> str:
    """Generate TypeScript interface from Python dataclass.

    Args:
        cls: The dataclass to convert.

    Returns:
        TypeScript interface definition.

    Raises:
        ValueError: If cls is not a dataclass.
    """
    if not dataclasses.is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    lines = [f"export interface {cls.__name__} {{"]

    hints = get_type_hints(cls)
    for field_info in dataclasses.fields(cls):
        ts_type = python_to_ts_type(hints.get(field_info.name, field_info.type))
        lines.append(f"  {field_info.name}: {ts_type};")

    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    """Generate all TypeScript types."""
    print("// Auto-generated from Python schema - DO NOT EDIT")
    print("// Run: python scripts/generate_overwatch_types.py")
    print(f"// Generated from: esper.karn.sanctum.schema")
    print()

    # Generate enums
    print(generate_enum(SeedStage))
    print()

    # Generate interfaces in dependency order (dependencies first)
    dataclasses_to_generate = [
        # Base types (no dependencies)
        CounterfactualConfig,
        CounterfactualSnapshot,
        GPUStats,
        SeedState,
        RewardComponents,
        DecisionSnapshot,
        EventLogEntry,
        RunConfig,
        # Types with dependencies on base types
        BestRunRecord,
        TamiyoState,
        SystemVitals,
        EnvState,
        # Top-level snapshot (depends on everything)
        SanctumSnapshot,
    ]

    for cls in dataclasses_to_generate:
        print(generate_interface(cls))
        print()


if __name__ == "__main__":
    main()
