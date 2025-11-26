from __future__ import annotations

from pathlib import Path

import pytest

from esper.leyline import leyline_pb2
from esper.tamiyo import TamiyoGraphBuilder, TamiyoGraphBuilderConfig, TamiyoPersistenceError


def test_graph_builder_flush_raises_on_io_error(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    target_path = tmp_path / "norms.json"
    builder = TamiyoGraphBuilder(TamiyoGraphBuilderConfig(normalizer_path=target_path))

    original_write_text = Path.write_text

    def failing_write_text(self: Path, *args, **kwargs):  # type: ignore[override]
        if self == target_path:
            raise OSError("disk full")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write_text, raising=False)

    packet = leyline_pb2.SystemStatePacket(version=1, current_epoch=1, training_run_id="run-error")
    _ = packet.seed_states.add()
    with pytest.raises(TamiyoPersistenceError):
        builder.build(packet)

