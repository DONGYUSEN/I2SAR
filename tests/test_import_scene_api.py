import pytest

from i2sar import Project
from i2sar.core.enums import AcquisitionMode
from i2sar.io import import_scene
from i2sar.io.base import ImportResult, SourceRef


def test_import_scene_rejects_unknown_auto_source(tmp_path):
    project = Project.create(tmp_path / "project", name="project")
    unknown = tmp_path / "unknown"
    unknown.mkdir()

    with pytest.raises(ValueError, match="could not detect importer"):
        import_scene(project, unknown)


def test_import_scene_defaults_tianyi_to_stripmap_and_sentinel_to_tops(monkeypatch, tmp_path):
    seen = []

    def fake_import(self, project):
        seen.append((self.sensor, self.acquisition_mode))
        return ImportResult(
            scene_id="scene",
            scene_path=tmp_path / "scene.h5",
            sensor=self.sensor,
            acquisition_mode=self.acquisition_mode,
            slc=SourceRef("/tmp/scene.tiff"),
        )

    monkeypatch.setattr("i2sar.io.SafeLikeImporter.import_to_project", fake_import)
    project = Project.create(tmp_path / "project", name="project")

    import_scene(project, tmp_path / "tianyi", sensor="tianyi")
    import_scene(project, tmp_path / "s1", sensor="sentinel1")

    assert seen == [("tianyi", AcquisitionMode.STRIPMAP), ("sentinel1", AcquisitionMode.TOPS)]
