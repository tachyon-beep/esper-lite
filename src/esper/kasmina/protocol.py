"""Kasmina Protocol - Re-exports from leyline for backwards compatibility.

The canonical HostProtocol definition now lives in leyline.host_protocol.
Import from esper.leyline instead of esper.kasmina.protocol.

This re-export exists only for backwards compatibility during migration.
"""

from esper.leyline.host_protocol import HostProtocol

__all__ = ["HostProtocol"]
