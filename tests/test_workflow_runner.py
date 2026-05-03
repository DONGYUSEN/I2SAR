import h5py

from i2sar import Project
from i2sar.core.enums import TaskStatus
from i2sar.workflow import TaskSpec, run_empty_task


def test_task_hash_is_stable_for_equivalent_json_params():
    first = TaskSpec("scene.validate", "scene", "s1", params={"b": 2, "a": {"y": 2, "x": 1}})
    second = TaskSpec("scene.validate", "scene", "s1", params={"a": {"x": 1, "y": 2}, "b": 2})
    changed = TaskSpec("scene.validate", "scene", "s1", params={"a": {"x": 1, "y": 3}, "b": 2})

    assert first.task_hash() == second.task_hash()
    assert first.task_hash() != changed.task_hash()


def test_empty_task_completes_then_skips(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    spec = TaskSpec(
        task_id="project.empty",
        owner_type="project",
        owner_id="demo",
        inputs=[],
        outputs=["project.h5:/workflow/tasks/project.empty"],
        params={"step": "empty"},
    )

    first = run_empty_task(project, spec, resume=True)
    second = run_empty_task(project, spec, resume=True)

    assert first.status is TaskStatus.COMPLETED
    assert second.status is TaskStatus.SKIPPED

    with h5py.File(project.path, "r") as h5:
        task = next(iter(h5["workflow/tasks"].values()))
        assert task.attrs["status"] == "skipped"
        assert task.attrs["task_hash"] == spec.task_hash()


def test_task_group_names_do_not_collide_for_similar_task_ids(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    slash = TaskSpec("a/b", "project", "demo", params={"value": "slash"})
    underscore = TaskSpec("a_b", "project", "demo", params={"value": "underscore"})

    run_empty_task(project, slash)
    run_empty_task(project, underscore)

    with h5py.File(project.path, "r") as h5:
        tasks = h5["workflow/tasks"]
        task_ids = {group.attrs["task_id"] for group in tasks.values()}
        assert task_ids == {"a/b", "a_b"}
        assert len(tasks) == 2


def test_skipped_task_updates_project_timestamp(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    spec = TaskSpec("project.empty", "project", "demo", params={"step": "empty"})
    run_empty_task(project, spec, resume=True)

    with h5py.File(project.path, "a") as h5:
        h5.attrs["updated_at"] = "2000-01-01T00:00:00Z"

    run_empty_task(project, spec, resume=True)

    with h5py.File(project.path, "r") as h5:
        assert h5.attrs["updated_at"] != "2000-01-01T00:00:00Z"


def test_changed_task_hash_records_invalidation_history(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    first = TaskSpec("project.empty", "project", "demo", params={"step": "one"})
    second = TaskSpec("project.empty", "project", "demo", params={"step": "two"})

    first_record = run_empty_task(project, first, resume=True)
    second_record = run_empty_task(project, second, resume=True)

    assert first_record.status is TaskStatus.COMPLETED
    assert second_record.status is TaskStatus.COMPLETED

    with h5py.File(project.path, "r") as h5:
        task = h5["workflow/tasks"][next(iter(h5["workflow/tasks"].keys()))]
        assert task.attrs["task_hash"] == second.task_hash()
        invalidations = task["invalidations"]
        assert len(invalidations) == 1
        entry = invalidations["000001"]
        assert entry.attrs["task_hash"] == first.task_hash()
        assert entry.attrs["status"] == "invalidated"
