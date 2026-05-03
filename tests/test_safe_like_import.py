import h5py
import numpy as np
import tifffile
import zipfile

from i2sar import Project
from i2sar.core.enums import AcquisitionMode
from i2sar.io.safe_like import SafeLikeImporter


def _make_safe_like(root):
    (root / "annotation" / "calibration").mkdir(parents=True)
    (root / "measurement").mkdir()
    (root / "annotation" / "scene.xml").write_text("<root/>", encoding="utf-8")
    (root / "annotation" / "calibration" / "calibration.xml").write_text("<root/>", encoding="utf-8")
    tifffile.imwrite(root / "measurement" / "scene.tiff", np.array([[1 + 2j]], dtype=np.complex64))
    (root / "manifest.safe").write_text("manifest", encoding="utf-8")


def test_safe_like_discovery_and_single_band_complex_attrs(monkeypatch, tmp_path):
    product = tmp_path / "SAFE"
    _make_safe_like(product)
    importer = SafeLikeImporter(product, sensor="tianyi", acquisition_mode=AcquisitionMode.STRIPMAP)
    files = importer.discover_files()

    assert files["annotation"].endswith("annotation/scene.xml")
    assert files["calibration"].endswith("annotation/calibration/calibration.xml")
    assert files["manifest"].endswith("manifest.safe")
    assert files["tiff"].endswith("measurement/scene.tiff")

    monkeypatch.setattr(
        importer,
        "parse",
        lambda: importer._parsed_from_parts(
            scene_id="safe_scene",
            acquisition_time="2026-01-01T00:00:00Z",
            acquisition={"polarisation": "VV"},
            scene={"sceneCorners": [{"lat": 0.0, "lon": 0.0}]},
            radar_grid={"numberOfRows": 10, "numberOfColumns": 20},
            orbit={"stateVectors": []},
            doppler={},
            slc_member=files["tiff"],
        ),
    )
    project = Project.create(tmp_path / "project", name="project")
    result = importer.import_to_project(project)

    with h5py.File(result.scene_path, "r") as h5:
        assert h5.attrs["sensor"] == "tianyi"
        assert h5["slc"].attrs["sample_format"] == "cint16"
        assert h5["slc"].attrs["storage_layout"] == "single_band_complex"
        assert h5["slc"].attrs["complex_band_count"] == 1
        assert h5["slc"].attrs["processing_format"] == "single_band_cfloat32"
        assert h5["slc/data"].dtype.fields["real"][0] == np.dtype("float32")


def test_safe_like_tops_creates_tops_schema_groups(monkeypatch, tmp_path):
    product = tmp_path / "S1_SAFE"
    _make_safe_like(product)
    importer = SafeLikeImporter(product, sensor="sentinel1", acquisition_mode=AcquisitionMode.TOPS)
    files = importer.discover_files()
    monkeypatch.setattr(
        importer,
        "parse",
        lambda: importer._parsed_from_parts(
            scene_id="s1_tops",
            acquisition_time="2026-01-01T00:00:00Z",
            acquisition={"polarisation": "VV"},
            scene={"sceneCorners": []},
            radar_grid={"numberOfRows": 10, "numberOfColumns": 20},
            orbit={"stateVectors": []},
            doppler={},
            slc_member=files["tiff"],
        ),
    )
    project = Project.create(tmp_path / "project", name="project")
    result = importer.import_to_project(project)

    with h5py.File(result.scene_path, "r") as h5:
        assert "/tops/swaths" in h5
        assert "/tops/bursts" in h5


def test_safe_like_parse_reads_minimal_annotation_xml_and_writes_scene(tmp_path):
    product = tmp_path / "SAFE"
    (product / "annotation" / "calibration").mkdir(parents=True)
    (product / "measurement").mkdir()
    tifffile.imwrite(product / "measurement" / "scene.tiff", np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64))
    (product / "manifest.safe").write_text("manifest", encoding="utf-8")
    (product / "annotation" / "calibration" / "calibration.xml").write_text("<root/>", encoding="utf-8")
    orbits = "\n".join(
        f"""
        <orbit><time>2026-01-01T00:00:{idx:02d}Z</time>
          <position><x>{7000000 + idx}</x><y>{idx}</y><z>{idx + 1}</z></position>
          <velocity><x>1</x><y>2</y><z>3</z></velocity>
        </orbit>
        """
        for idx in range(2)
    )
    (product / "annotation" / "scene.xml").write_text(
        f"""
        <product>
          <adsHeader>
            <missionId>S1</missionId><mode>IW</mode><polarisation>VV</polarisation>
            <startTime>2026-01-01T00:00:00Z</startTime><stopTime>2026-01-01T00:01:00Z</stopTime>
          </adsHeader>
          <generalAnnotation>
            <productInformation><radarFrequency>5405000000</radarFrequency><prf>1200</prf><platformHeading>12</platformHeading></productInformation>
            <orbitList>{orbits}</orbitList>
          </generalAnnotation>
          <imageAnnotation><imageInformation>
            <numberOfLines>10</numberOfLines><numberOfSamples>20</numberOfSamples>
            <azimuthPixelSpacing>2</azimuthPixelSpacing><rangePixelSpacing>3</rangePixelSpacing>
            <azimuthFrequency>1200</azimuthFrequency><slantRangeTime>0.001</slantRangeTime><look_side>right</look_side>
          </imageInformation></imageAnnotation>
          <geolocationGrid><geolocationGridPointList>
            <geolocationGridPoint>
              <line>0</line><pixel>0</pixel><latitude>30</latitude><longitude>100</longitude>
              <incidenceAngle>35</incidenceAngle><slantRangeTime>0.001</slantRangeTime><azimuthTime>2026-01-01T00:00:00Z</azimuthTime>
            </geolocationGridPoint>
          </geolocationGridPointList></geolocationGrid>
          <dopplerCentroid><dcEstimateList><dcEstimate>
            <azimuthTime>2026-01-01T00:00:00Z</azimuthTime><t0>0.001</t0><dataDcPolynomial>1.0 2.0</dataDcPolynomial>
          </dcEstimate></dcEstimateList></dopplerCentroid>
        </product>
        """,
        encoding="utf-8",
    )

    project = Project.create(tmp_path / "project", name="project")
    importer = SafeLikeImporter(product, sensor="sentinel1", acquisition_mode=AcquisitionMode.TOPS)
    result = importer.import_to_project(project)

    with h5py.File(result.scene_path, "r") as h5:
        assert h5.attrs["sensor"] == "sentinel1"
        assert h5.attrs["acquisition_mode"] == "tops"
        assert h5["orbit/position"].shape == (2, 3)
        assert h5["slc"].attrs["storage_layout"] == "single_band_complex"
        assert h5["slc/data"].shape == (1, 2)


def test_safe_like_zip_import_preserves_vsi_member_reference(tmp_path):
    zip_path = tmp_path / "S1_SAFE.zip"
    annotation = """
    <product>
      <adsHeader>
        <missionId>S1</missionId><mode>IW</mode><polarisation>VV</polarisation>
        <startTime>2026-01-01T00:00:00Z</startTime><stopTime>2026-01-01T00:01:00Z</stopTime>
      </adsHeader>
      <generalAnnotation>
        <productInformation><radarFrequency>5405000000</radarFrequency><prf>1200</prf><platformHeading>12</platformHeading></productInformation>
        <orbitList></orbitList>
      </generalAnnotation>
      <imageAnnotation><imageInformation>
        <numberOfLines>10</numberOfLines><numberOfSamples>20</numberOfSamples>
        <azimuthPixelSpacing>2</azimuthPixelSpacing><rangePixelSpacing>3</rangePixelSpacing>
        <azimuthFrequency>1200</azimuthFrequency><slantRangeTime>0.001</slantRangeTime><look_side>right</look_side>
      </imageInformation></imageAnnotation>
      <geolocationGrid><geolocationGridPointList></geolocationGridPointList></geolocationGrid>
      <dopplerCentroid><dcEstimateList></dcEstimateList></dopplerCentroid>
    </product>
    """
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("S1.SAFE/annotation/scene.xml", annotation)
        zf.writestr("S1.SAFE/annotation/calibration/calibration.xml", "<root/>")
        tiff_path = tmp_path / "scene.tiff"
        tifffile.imwrite(tiff_path, np.array([[1 + 2j]], dtype=np.complex64))
        zf.write(tiff_path, "S1.SAFE/measurement/scene.tiff")
        zf.writestr("S1.SAFE/manifest.safe", "manifest")

    project = Project.create(tmp_path / "project", name="project")
    importer = SafeLikeImporter(zip_path, sensor="sentinel1", acquisition_mode=AcquisitionMode.TOPS)
    result = importer.import_to_project(project)

    with h5py.File(result.scene_path, "r") as h5:
        assert h5["slc"].attrs["storage"] == "zip"
        assert h5["slc"].attrs["member"] == "S1.SAFE/measurement/scene.tiff"
        assert h5["slc"].attrs["path"] == str(zip_path.resolve())
        assert h5["slc/data"][0, 0]["real"] == 1.0
