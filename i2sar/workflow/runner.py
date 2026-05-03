from __future__ import annotations

import hashlib
import json

import h5py

from i2sar.core.enums import TaskStatus
from i2sar.core.ids import slugify_id
from i2sar.core.time import utc_now_iso
from i2sar.project import Project
from i2sar.workflow.task import TaskRecord, TaskSpec


def _task_group_name(task_id: str) -> str:
    safe = slugify_id(task_id)
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe}--{digest}"


def _record_invalidation(group: h5py.Group) -> None:
    invalidations = group.require_group("invalidations")
    next_index = len(invalidations) + 1
    entry = invalidations.create_group(f"{next_index:06d}")
    if "task_hash" in group.attrs:
        entry.attrs["task_hash"] = group.attrs["task_hash"]
    entry.attrs["status"] = TaskStatus.INVALIDATED.value
    if "updated_at" in group.attrs:
        entry.attrs["updated_at"] = group.attrs["updated_at"]
    elif "finished_at" in group.attrs:
        entry.attrs["finished_at"] = group.attrs["finished_at"]


def run_empty_task(project: Project, spec: TaskSpec, resume: bool = True) -> TaskRecord:
    task_hash = spec.task_hash()
    group_name = _task_group_name(spec.task_id)
    now = utc_now_iso()
    with h5py.File(project.path, "a") as h5:
        tasks = h5.require_group("workflow/tasks")
        if group_name in tasks:
            group = tasks[group_name]
            if resume and group.attrs.get("task_hash") == task_hash and group.attrs.get("status") in {"completed", "skipped"}:
                group.attrs["status"] = TaskStatus.SKIPPED.value
                group.attrs["updated_at"] = now
                h5.attrs["updated_at"] = utc_now_iso()
                return TaskRecord(spec.task_id, task_hash, TaskStatus.SKIPPED)
            if group.attrs.get("task_hash") != task_hash:
                _record_invalidation(group)
                group.attrs["status"] = TaskStatus.INVALIDATED.value

        group = tasks.require_group(group_name)
        group.attrs["task_id"] = spec.task_id
        group.attrs["task_hash"] = task_hash
        group.attrs["owner_type"] = spec.owner_type
        group.attrs["owner_id"] = spec.owner_id
        group.attrs["started_at"] = now
        group.attrs["status"] = TaskStatus.RUNNING.value
        group.attrs["inputs_json"] = json.dumps(spec.inputs, sort_keys=True)
        group.attrs["outputs_json"] = json.dumps(spec.outputs, sort_keys=True)
        group.attrs["params_json"] = json.dumps(spec.params, sort_keys=True)
        group.attrs["finished_at"] = utc_now_iso()
        group.attrs["status"] = TaskStatus.COMPLETED.value
        h5.attrs["updated_at"] = utc_now_iso()
    return TaskRecord(spec.task_id, task_hash, TaskStatus.COMPLETED)
