from __future__ import annotations

from pathlib import Path

import pytest
import torch

from esper.core import EsperSettings
from esper.tolaria.profiler import maybe_profile
from esper.tolaria.trainer import TolariaTrainer, TrainingLoopConfig


class _TamiyoStub:
    def evaluate_epoch(self, state):  # type: ignore[no-untyped-def]
        command = type("Cmd", (), {"command_type": 0})()
        command.command_type = 0
        return command


class _KasminaStub:
    def apply_command(self, command):  # type: ignore[no-untyped-def]
        return None

    def export_seed_states(self):
        return []

    def advance_alpha(self, seed_id: str, *, steps: int = 1) -> float:
        return 0.0


def _trainer(tmp_path: Path, *, settings: EsperSettings) -> TolariaTrainer:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    model = nn.Sequential(nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 2, (4,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    trainer = TolariaTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        tamiyo=_TamiyoStub(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(max_epochs=1, gradient_accumulation_steps=1),
        settings=settings,
    )

    def _root(self):
        root = tmp_path / "tolaria"
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        return root

    trainer._checkpoint_root = _root.__get__(trainer, TolariaTrainer)  # type: ignore[attr-defined]
    return trainer


def test_maybe_profile_generates_timestamped_trace(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[str] = []

    class FakeProfiler:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def export_chrome_trace(self, path: str) -> None:
            paths.append(path)
            Path(path).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        torch.profiler,
        "ProfilerActivity",
        type("FakeActivity", (), {"CPU": "cpu", "CUDA": "cuda"}),
        raising=False,
    )
    monkeypatch.setattr(torch.profiler, "profile", lambda **_: FakeProfiler(), raising=False)  # type: ignore[arg-type]

    with maybe_profile(enabled=True, trace_dir=str(tmp_path / "traces"), name="epoch", active_steps=1):
        pass

    assert paths
    generated = Path(paths[0])
    assert generated.name.startswith("epoch-")
    assert generated.suffix == ".json"
    assert generated.exists()


def test_profiler_failure_emits_telemetry(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    fail_once = True

    from contextlib import contextmanager

    @contextmanager
    def failing_profile(**_kwargs):  # type: ignore[override]
        nonlocal fail_once
        yield
        if fail_once:
            fail_once = False
            raise RuntimeError("export failed")

    monkeypatch.setattr("esper.tolaria.trainer.maybe_profile", failing_profile, raising=False)
    recorded: list[tuple[str, dict[str, str]]] = []

    original_emit = TolariaTrainer._emit_event

    def _emit_event_spy(self, description, *, level, attributes=None):  # type: ignore[override]
        if description.startswith("tolaria.profiler"):
            recorded.append((description, dict(attributes or {})))
        return original_emit(self, description, level=level, attributes=attributes)

    monkeypatch.setattr(TolariaTrainer, "_emit_event", _emit_event_spy, raising=False)

    settings = EsperSettings().model_copy(
        update={
            "tolaria_profiler_enabled": True,
            "tolaria_profiler_dir": str(tmp_path / "traces"),
            "tolaria_profiler_active_steps": 1,
        }
    )

    trainer = _trainer(tmp_path, settings=settings)
    list(trainer.run())

    assert trainer._metrics["tolaria.profiler.traces_failed_total"] == pytest.approx(1.0)  # type: ignore[index]
    assert any(desc == "tolaria.profiler.export_failed" for desc, _ in recorded)
