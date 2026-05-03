from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from i2sar.core.enums import TaskStatus


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    owner_type: str
    owner_id: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def task_hash(self) -> str:
        payload = {
            "task_id": self.task_id,
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "params": self.params,
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_hash: str
    status: TaskStatus
