from i2sar.workflow import insar_workflow_specs, rtc_workflow_specs, scene_workflow_specs


def test_scene_workflow_specs():
    specs = scene_workflow_specs("scene1")
    assert [spec.task_id for spec in specs] == [
        "scene.import_scene",
        "scene.validate_scene",
        "scene.multilook",
        "scene.quicklook",
    ]


def test_insar_and_rtc_workflow_specs_are_distinct():
    insar = insar_workflow_specs("m__s")
    rtc = rtc_workflow_specs("scene1")

    assert any(spec.task_id == "insar.form_interferogram" for spec in insar)
    assert any(spec.task_id == "rtc.compute_rtc_factor" for spec in rtc)
    assert all(not spec.task_id.startswith("insar.") for spec in rtc)
