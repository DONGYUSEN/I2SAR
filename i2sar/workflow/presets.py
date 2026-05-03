from __future__ import annotations

from i2sar.workflow.task import TaskSpec


def scene_workflow_specs(scene_id: str) -> list[TaskSpec]:
    return [
        TaskSpec("scene.import_scene", "scene", scene_id, params={"stage": "import"}),
        TaskSpec("scene.validate_scene", "scene", scene_id, params={"stage": "validate"}),
        TaskSpec("scene.multilook", "scene", scene_id, params={"stage": "multilook"}),
        TaskSpec("scene.quicklook", "scene", scene_id, params={"stage": "quicklook"}),
    ]


def insar_workflow_specs(pair_id: str) -> list[TaskSpec]:
    return [
        TaskSpec("insar.create_pair", "pair", pair_id, params={"stage": "create_pair"}),
        TaskSpec("insar.prepare_geometry", "pair", pair_id, params={"stage": "geometry"}),
        TaskSpec("insar.register", "pair", pair_id, params={"stage": "registration"}),
        TaskSpec(
            "insar.form_interferogram",
            "pair",
            pair_id,
            params={"stage": "interferometry"},
        ),
        TaskSpec("insar.unwrap", "pair", pair_id, params={"stage": "unwrapping"}),
        TaskSpec("insar.phase_to_los", "pair", pair_id, params={"stage": "displacement"}),
        TaskSpec("insar.export", "pair", pair_id, params={"stage": "export"}),
    ]


def rtc_workflow_specs(scene_id: str) -> list[TaskSpec]:
    return [
        TaskSpec("rtc.prepare_geometry", "scene", scene_id, params={"stage": "geometry"}),
        TaskSpec("rtc.compute_rtc_factor", "scene", scene_id, params={"stage": "rtc_factor"}),
        TaskSpec(
            "rtc.terrain_correct",
            "scene",
            scene_id,
            params={"stage": "terrain_correction"},
        ),
        TaskSpec("rtc.radiometric_normalize", "scene", scene_id, params={"stage": "radiometry"}),
        TaskSpec("rtc.geocode", "scene", scene_id, params={"stage": "geocode"}),
        TaskSpec("rtc.make_rtc_product", "scene", scene_id, params={"stage": "product"}),
        TaskSpec("rtc.export", "scene", scene_id, params={"stage": "export"}),
    ]
