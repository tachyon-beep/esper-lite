"""Leyline contract utilities.

Hosts helper classes for managing protobuf schema versions referenced in
`docs/design/detailed_design/00-leyline.md`.
"""

from pathlib import Path

from ._generated import leyline_pb2
from .registry import ContractBundle, ContractRegistry

DEFAULT_BUNDLE_NAME = "leyline-core"
DEFAULT_BUNDLE_VERSION = "v1alpha1"


def register_default_bundle(registry: ContractRegistry | None = None) -> ContractRegistry:
    """Register the default Leyline protobuf bundle with the provided registry."""

    bundle = ContractBundle(
        name=DEFAULT_BUNDLE_NAME,
        version=DEFAULT_BUNDLE_VERSION,
        schema_dir=Path(__file__).resolve().parent / "_generated",
    )
    target = registry or ContractRegistry()
    target.register(bundle)
    return target


__all__ = [
    "ContractRegistry",
    "ContractBundle",
    "DEFAULT_BUNDLE_NAME",
    "DEFAULT_BUNDLE_VERSION",
    "leyline_pb2",
    "register_default_bundle",
]
