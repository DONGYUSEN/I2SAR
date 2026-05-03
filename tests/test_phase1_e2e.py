import h5py

from i2sar import Project
from i2sar.core.enums import AcquisitionMode, EntityType, ProductType
from i2sar.model import ProductInfo, SceneInfo
from i2sar.workflow import insar_workflow_specs, rtc_workflow_specs, run_empty_task


def _task_ids(h5):
    return {group.attrs["task_id"] for group in h5["workflow/tasks"].values()}


def test_phase1_e2e_project_scene_pair_product_workflows(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    master = SceneInfo("master", "sentinel1", AcquisitionMode.TOPS, "2026-04-28T00:00:00Z")
    slave = SceneInfo("slave", "sentinel1", AcquisitionMode.TOPS, "2026-04-28T00:12:00Z")

    project.create_scene(master)
    project.create_scene(slave)
    project.create_pair("master", "slave")
    project.create_product(ProductInfo("rtc_master", ProductType.RTC, EntityType.SCENE, "master"))

    for spec in insar_workflow_specs("master__slave"):
        run_empty_task(project, spec)
    for spec in rtc_workflow_specs("master"):
        run_empty_task(project, spec)

    with h5py.File(project.path, "r") as h5:
        assert "master" in h5["scenes"]
        assert "master__slave" in h5["pairs"]
        assert "rtc_master" in h5["products"]
        task_ids = _task_ids(h5)
        assert "insar.form_interferogram" in task_ids
        assert "rtc.compute_rtc_factor" in task_ids
