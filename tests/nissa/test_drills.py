from prometheus_client import CollectorRegistry

from esper.nissa import NissaIngestor, NissaIngestorConfig, run_all_drills


class _ElasticsearchStub:
    def __init__(self) -> None:
        self.indexed: list[tuple[str, dict]] = []

    def index(self, index: str, document: dict) -> None:
        self.indexed.append((index, document))


def _ingestor() -> NissaIngestor:
    registry = CollectorRegistry()
    es = _ElasticsearchStub()
    config = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
    )
    return NissaIngestor(config, es_client=es, registry=registry)


def test_fault_drills_trigger_and_clear_alerts() -> None:
    ingestor = _ingestor()
    results = run_all_drills(ingestor)
    assert all(info["alert"] is not None for info in results.values())
    assert all(info["cleared"] for info in results.values())
