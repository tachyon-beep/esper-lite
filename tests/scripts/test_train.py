def test_telemetry_lifecycle_only_flag_wired():
    import esper.scripts.train as train

    parser = train.build_parser()
    args = parser.parse_args(["heuristic", "--telemetry-lifecycle-only"])
    assert args.telemetry_lifecycle_only is True

