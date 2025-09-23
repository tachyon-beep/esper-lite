from __future__ import annotations

import pytest
from fakeredis.aioredis import FakeRedis

from esper.core import EsperSettings
from esper.leyline import leyline_pb2
from esper.urza import UrzaLibrary
from esper.weatherlight.service_runner import WeatherlightService


@pytest.mark.asyncio
async def test_weatherlight_updates_urza_capabilities(tmp_path: pytest.PathLike, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESPER_LEYLINE_SECRET", "test-secret")
    settings = EsperSettings(
        redis_url="redis://localhost:6379/0",
        urza_artifact_dir=str(tmp_path / "urza" / "artifacts"),
        urza_database_url=f"sqlite:///{(tmp_path / 'urza' / 'catalog.db').as_posix()}",
    )
    service = WeatherlightService(settings=settings)
    await service.start()
    try:
        # Create a Urza record with parameters alpha,beta
        library: UrzaLibrary = service._urza_library  # type: ignore[attr-defined]
        bp_id = "BP_CAP"
        artifact_path = tmp_path / "artifact.pt"
        artifact_path.write_bytes(b"dummy")
        descriptor = leyline_pb2.BlueprintDescriptor(
            blueprint_id=bp_id,
            name="bp-cap",
            tier=leyline_pb2.BlueprintTier.BLUEPRINT_TIER_SAFE,
        )
        extras = {
            "graph_metadata": {
                "parameters": [
                    {"name": "alpha", "min": 0.0, "max": 1.0, "default": 0.5, "span": 1.0},
                    {"name": "beta", "min": -0.5, "max": 0.5, "default": 0.0, "span": 1.0},
                ]
            }
        }
        library.save(descriptor, artifact_path, extras=extras)

        # Inject a Kasmina seed in BLENDING for this blueprint
        mgr = service._kasmina_manager  # type: ignore[attr-defined]
        from esper.kasmina.seed_manager import SeedContext
        ctx = SeedContext(seed_id="seed-A")
        # Transition lifecycle to BLENDING via allowed path
        ctx.lifecycle.transition(leyline_pb2.SEED_STAGE_GERMINATED)
        ctx.lifecycle.transition(leyline_pb2.SEED_STAGE_TRAINING)
        ctx.lifecycle.transition(leyline_pb2.SEED_STAGE_BLENDING)
        ctx.metadata["blueprint_id"] = bp_id
        mgr._seeds["seed-A"] = ctx  # type: ignore[attr-defined]

        # Run the capability update directly
        service._update_urza_capabilities_from_kasmina()  # type: ignore[attr-defined]

        rec = library.get(bp_id)
        assert rec is not None
        gm = rec.extras.get("graph_metadata", {})  # type: ignore[index]
        assert isinstance(gm, dict)
        caps = gm.get("capabilities", {})
        assert isinstance(caps, dict)
        by_id = caps.get("allowed_parameters_by_seed_id", {})
        assert by_id.get("seed-A") == ["alpha", "beta"]
    finally:
        await service.shutdown()
