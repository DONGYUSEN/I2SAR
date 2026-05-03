from __future__ import annotations

import re


def slugify_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise ValueError("identifier cannot be empty")
    return slug


def make_pair_id(master_scene_id: str, slave_scene_id: str) -> str:
    return f"{slugify_id(master_scene_id)}__{slugify_id(slave_scene_id)}"
