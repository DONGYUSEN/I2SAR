from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from i2sar.io.base import SourceRef


def source_storage(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    stem_suffix = Path(path.stem).suffix.lower()
    if suffix == ".zip":
        return "zip"
    if suffix in {".tar", ".tgz"} or (suffix == ".gz" and stem_suffix == ".tar"):
        return "tar"
    return "file"


def list_product_members(path: str | Path) -> list[str]:
    path = Path(path)
    storage = source_storage(path)
    if storage == "zip":
        with zipfile.ZipFile(path) as zf:
            return sorted(name for name in zf.namelist() if not name.endswith("/"))
    if storage == "tar":
        with tarfile.open(path) as tf:
            return sorted(name for name in tf.getnames() if not name.endswith("/"))
    if path.is_dir():
        return sorted(str(member.relative_to(path)) for member in path.rglob("*") if member.is_file())
    if path.is_file():
        return [path.name]
    raise FileNotFoundError(path)


def build_member_ref(path: str | Path, member: str) -> SourceRef:
    path = Path(path)
    storage = source_storage(path)
    if storage == "file":
        return SourceRef(path=str((path / member).resolve()), storage="file")
    return SourceRef(path=str(path.resolve()), storage=storage, member=member)
