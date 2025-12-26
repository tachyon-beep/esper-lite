"""OverwatchBackend - WebSocket server for Overwatch dashboard.

This module will be fully implemented in Task 5. For now it provides
a placeholder class so the package can be imported.
"""

from __future__ import annotations


class OverwatchBackend:
    """WebSocket backend broadcasting SanctumSnapshot to web clients.

    Full implementation will wrap SanctumAggregator and broadcast
    snapshots at 10 Hz over WebSocket.

    Implemented in Task 5.
    """

    def __init__(self, port: int = 8080) -> None:
        """Initialize the backend.

        Args:
            port: WebSocket server port (default 8080)
        """
        self.port = port
        raise NotImplementedError(
            "OverwatchBackend is a placeholder - full implementation in Task 5"
        )
