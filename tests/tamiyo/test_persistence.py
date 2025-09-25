from __future__ import annotations

from datetime import UTC, datetime, timedelta

from esper.leyline import leyline_pb2
from esper.tamiyo import FieldReportStore, FieldReportStoreConfig


def _build_report(report_id: str, *, issued_at: datetime) -> leyline_pb2.FieldReport:
    report = leyline_pb2.FieldReport(
        version=1,
        report_id=report_id,
        command_id=f"cmd-{report_id}",
        training_run_id="run-1",
        seed_id="seed-1",
        blueprint_id="bp-1",
        outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
    )
    report.issued_at.FromDatetime(issued_at)
    return report


def test_store_reload_recovers_reports(tmp_path) -> None:
    path = tmp_path / "field_reports.log"
    config = FieldReportStoreConfig(path=path)
    store = FieldReportStore(config)
    report = _build_report("rpt-1", issued_at=datetime.now(tz=UTC))
    store.append(report)

    reloaded = FieldReportStore(config)
    reports = reloaded.reports()
    assert len(reports) == 1
    assert reports[0].report_id == "rpt-1"


def test_store_retention_prunes_expired_entries(tmp_path) -> None:
    path = tmp_path / "field_reports.log"
    config = FieldReportStoreConfig(path=path, retention=timedelta(hours=1))
    store = FieldReportStore(config)
    stale = _build_report("stale", issued_at=datetime.now(tz=UTC) - timedelta(days=2))
    fresh = _build_report("fresh", issued_at=datetime.now(tz=UTC))
    store.append(stale)
    store.append(fresh)

    store.enforce_retention(now=datetime.now(tz=UTC))
    reports = store.reports()
    assert [report.report_id for report in reports] == ["fresh"]

    reloaded = FieldReportStore(config)
    assert [report.report_id for report in reloaded.reports()] == ["fresh"]
