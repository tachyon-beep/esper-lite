from esper.nissa import AlertEngine, AlertRouter, AlertRule


def test_alert_engine_triggers_after_threshold() -> None:
    router = AlertRouter()
    rule = AlertRule(
        name="test_alert",
        metric="metric.value",
        threshold=10.0,
        for_count=2,
        routes=("slack",),
    )
    engine = AlertEngine([rule], router=router)

    engine.evaluate({"metric.value": 9.0}, source="source")
    assert not engine.active_alerts

    engine.evaluate({"metric.value": 11.0}, source="source")
    assert not engine.active_alerts

    engine.evaluate({"metric.value": 12.0}, source="source")
    assert "test_alert" in engine.active_alerts
    events = router.events()
    assert events[-1].routes == ("slack",)
    assert router.slack_notifications()
