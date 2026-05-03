from __future__ import annotations

import tarfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from i2sar.io.source import source_storage


def read_xml_root(source: str | Path, member: str) -> ET.Element:
    source = Path(source)
    storage = source_storage(source)
    if storage == "zip":
        with zipfile.ZipFile(source) as zf:
            with zf.open(member) as fh:
                return ET.fromstring(fh.read())
    if storage == "tar":
        with tarfile.open(source) as tf:
            fh = tf.extractfile(member)
            if fh is None:
                raise FileNotFoundError(member)
            with fh:
                return ET.fromstring(fh.read())
    path = source / member if source.is_dir() else source
    return ET.parse(path).getroot()


def find_required(root: ET.Element, path: str) -> ET.Element:
    elem = root.find(path)
    if elem is None:
        raise ValueError(f"missing XML element: {path}")
    return elem


def text(elem: ET.Element | None, path: str, default: str = "") -> str:
    if elem is None:
        return default
    value = elem.findtext(path)
    return default if value is None else value.strip()


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value.split()[0])
    except (IndexError, ValueError):
        return default


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(value.split()[0])
    except (IndexError, ValueError):
        return default


def gps_seconds(timestamp: str) -> float:
    if not timestamp:
        return 0.0
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    ts = timestamp
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    elif len(ts) < 6 or not (ts[-6] in ("+", "-") and ts[-3] == ":"):
        ts = ts + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - gps_epoch).total_seconds()
