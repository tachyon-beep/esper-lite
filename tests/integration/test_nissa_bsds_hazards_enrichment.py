from __future__ import annotations

from pathlib import Path

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.leyline import leyline_pb2
from esper.nissa import NissaIngestor, NissaIngestorConfig
from esper.urza import UrzaLibrary


class _ElasticsearchStub:
    def __init__(self) -> None:
        self.indexed: list[tuple[str, dict]] = []

    def index(self, index: str, document: dict) -> None:
        self.indexed.append((index, document))


def _urza_with_hazards(tmp_path: Path, bp_id: str) -> UrzaLibrary:
    root = tmp_path / "urza"
    lib = UrzaLibrary(root=root)
    artifact = tmp_path / f"{bp_id}.pt"
    artifact.write_bytes(b"dummy")
    descriptor = BlueprintDescriptor(
        blueprint_id=bp_id,
        name=bp_id,
        tier=BlueprintTier.BLUEPRINT_TIER_SAFE,
        risk=0.35,
        stage=2,
        quarantine_only=False,
        approval_required=False,
        description="bsds-hazards-index-test",
    )
    lib.save(
        descriptor,
        artifact,
        extras={
            "bsds": {
                "risk_score": 0.6,
                "hazard_band": "HIGH",
                "handling_class": "restricted",
                "resource_profile": "cpu",
                "provenance": "URABRASK",
                "hazards": {
                    "grad_instability": "explode",
                    "precision": "sensitive",
                },
            }
        },
    )
    return lib


def test_nissa_enriches_es_doc_with_bsds_hazards(tmp_path: Path) -> None:
    es = _ElasticsearchStub()
    cfg = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
    )
    urza = _urza_with_hazards(tmp_path, "bp-haz")
    ingestor = NissaIngestor(cfg, es_client=es, urza=urza)

    pkt = leyline_pb2.TelemetryPacket(
        packet_id="bp-haz",
        source_subsystem="tamiyo",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
    )
    ingestor.ingest_telemetry(pkt)
    assert es.indexed, "document should be indexed"
    index, doc = es.indexed[-1]
    assert index == "telemetry"
    hazards = doc.get("bsds_hazards")
    assert isinstance(hazards, dict)
    assert hazards.get("grad_instability") == "explode"
    assert hazards.get("precision") == "sensitive"
