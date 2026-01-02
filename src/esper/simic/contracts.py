"""Simic Contracts - Re-exports from leyline for backwards compatibility.

The canonical seed protocol definitions now live in leyline.seed_protocols.
Import from esper.leyline instead of esper.simic.contracts.

This re-export exists only for backwards compatibility during migration.
"""

from esper.leyline.seed_protocols import (
    SeedStateProtocol,
    SeedSlotProtocol,
    SlottedHostProtocol,
)

__all__ = [
    "SeedStateProtocol",
    "SeedSlotProtocol",
    "SlottedHostProtocol",
]
