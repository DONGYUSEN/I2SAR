import h5py
import numpy as np
import zipfile

from i2sar import Project
from i2sar.io.lutan import LutanImporter


def _write_multiband_tiff(filepath, data):
    try:
        from osgeo import gdal
    except ImportError:
        import tifffile

        tifffile.imwrite(filepath, data)
        return

    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    rows, cols, bands = data.shape
    out_ds = driver.Create(str(filepath), cols, rows, bands, gdal.GDT_Int16)
    for idx in range(bands):
        out_ds.GetRasterBand(idx + 1).WriteArray(data[..., idx])
    out_ds = None


def test_lutan_import_writes_smoothed_and_raw_orbit(monkeypatch, tmp_path):
    product = tmp_path / "product"
    product.mkdir()
    
    data = np.random.randint(-32768, 32767, (10, 20, 2), dtype=np.int16)
    _write_multiband_tiff(product / "scene_SLC.tiff", data)
    
    (product / "scene.meta.xml").write_text("<root/>", encoding="utf-8")

    orbit = {
        "stateVectors": [
            {
                "timeUTC": f"2026-01-01T00:00:{idx:02d}Z",
                "gpsTime": float(idx),
                "posX": 1.0 + idx,
                "posY": 2.0,
                "posZ": 3.0,
                "velX": 4.0,
                "velY": 5.0,
                "velZ": 6.0,
            }
            for idx in range(12)
        ]
    }

    importer = LutanImporter(product)
    monkeypatch.setattr(
        importer,
        "parse",
        lambda: importer._parsed_from_parts(
            scene_id="lutan_scene",
            acquisition_time="2026-01-01T00:00:00Z",
            acquisition={"polarisation": "HH"},
            scene={"sceneCorners": [{"lat": 0.0, "lon": 0.0}]},
            radar_grid={"numberOfRows": 10, "numberOfColumns": 20},
            orbit=orbit,
            doppler={},
            slc_member="scene_SLC.tiff",
        ),
    )

    project = Project.create(tmp_path / "project", name="project")
    result = importer.import_to_project(project)

    with h5py.File(result.scene_path, "r") as h5:
        assert h5["orbit"].attrs["smoothed"] == True
        assert h5["orbit_raw/position"][0, 0] == 1.0
        assert h5["slc"].attrs["sample_format"] == "iq_int16"
        assert h5["slc"].attrs["storage_layout"] == "two_band_iq"


def test_lutan_parse_reads_minimal_meta_xml_and_writes_scene(tmp_path):
    product = tmp_path / "product"
    product.mkdir()
    
    data = np.random.randint(-32768, 32767, (10, 20, 2), dtype=np.int16)
    _write_multiband_tiff(product / "scene_SLC.tiff", data)
    
    state_vectors = "\n".join(
        f"""
        <stateVec>
          <timeUTC>2026-01-01T00:00:{idx:02d}Z</timeUTC>
          <posX>{7000000 + idx}</posX><posY>{idx}</posY><posZ>{idx + 1}</posZ>
          <velX>1</velX><velY>2</velY><velZ>3</velZ>
        </stateVec>
        """
        for idx in range(8)
    )
    (product / "scene.meta.xml").write_text(
        f"""
        <product>
          <generalHeader><mission>LT1</mission><sensorMode>SM</sensorMode></generalHeader>
          <productInfo>
            <acquisitionInfo>
              <sensor>LuTan-1</sensor><imagingMode>stripmap</imagingMode>
              <lookDirection>RIGHT</lookDirection><polarisationMode>HH</polarisationMode>
            </acquisitionInfo>
            <imageDataInfo>
              <imageRaster>
                <rowSpacing>2.0</rowSpacing><columnSpacing>3.0</columnSpacing>
                <groundRangeResolution>4.0</groundRangeResolution><azimuthResolution>5.0</azimuthResolution>
                <numberOfRows>10</numberOfRows><numberOfColumns>20</numberOfColumns>
              </imageRaster>
            </imageDataInfo>
            <sceneInfo>
              <sceneID>Lutan Scene 01</sceneID>
              <start><timeUTC>2026-01-01T00:00:00Z</timeUTC></start>
              <stop><timeUTC>2026-01-01T00:01:00Z</timeUTC></stop>
              <rangeTime><firstPixel>0.001</firstPixel><lastPixel>0.002</lastPixel></rangeTime>
              <sceneCenterCoord><lat>30</lat><lon>100</lon><incidenceAngle>35</incidenceAngle></sceneCenterCoord>
              <sceneCornerCoord name="ul"><refRow>0</refRow><refColumn>0</refColumn><lat>30</lat><lon>100</lon></sceneCornerCoord>
              <sceneAverageHeight>100</sceneAverageHeight><headingAngle>12</headingAngle>
            </sceneInfo>
          </productInfo>
          <platform><orbit>
            <orbitHeader>
              <generationSystem>test</generationSystem><sensor>LT1</sensor><accuracy>unknown</accuracy>
              <stateVectorRefFrame>ECEF</stateVectorRefFrame><stateVectorRefTime>2026-01-01T00:00:00Z</stateVectorRefTime>
              <stateVecFormat>ECEF</stateVecFormat><numStateVectors>8</numStateVectors>
              <firstStateTime><firstStateTimeUTC>2026-01-01T00:00:00Z</firstStateTimeUTC></firstStateTime>
              <lastStateTime><lastStateTimeUTC>2026-01-01T00:00:07Z</lastStateTimeUTC></lastStateTime>
              <stateVectorTimeSpacing>1</stateVectorTimeSpacing>
            </orbitHeader>
            {state_vectors}
          </orbit></platform>
          <instrument><radarParameters><centerFrequency>5405000000</centerFrequency></radarParameters>
            <settings><settingRecord><PRF>1200</PRF></settingRecord></settings>
          </instrument>
          <processing>
            <doppler><dopplerCentroid><dopplerEstimate>
              <timeUTC>2026-01-01T00:00:00Z</timeUTC><dopplerAtMidRange>10</dopplerAtMidRange>
              <combinedDoppler><validityRangeMin>0</validityRangeMin><validityRangeMax>1</validityRangeMax>
                <referencePoint>0</referencePoint><polynomialDegree>1</polynomialDegree>
                <coefficient exponent="0">1.5</coefficient><coefficient exponent="1">2.5</coefficient>
              </combinedDoppler>
            </dopplerEstimate></dopplerCentroid></doppler>
            <processingParameter><rangeLooks>1</rangeLooks><azimuthLooks>1</azimuthLooks></processingParameter>
            <processingFlags><geocodedFlag>false</geocodedFlag></processingFlags>
          </processing>
        </product>
        """,
        encoding="utf-8",
    )

    project = Project.create(tmp_path / "project", name="project")
    result = LutanImporter(product).import_to_project(project)

    with h5py.File(result.scene_path, "r") as h5:
        assert h5.attrs["scene_id"] == "Lutan_Scene_01"
        assert h5["radar_grid"].attrs["json"]
        assert h5["orbit"].attrs["smoothed"] == True
        assert h5["orbit/position"].shape == (8, 3)
        assert h5["slc"].attrs["sample_format"] == "iq_int16"
        assert h5["slc/data"].shape == (10, 20)


def test_lutan_zip_import_preserves_vsi_member_reference(tmp_path):
    zip_path = tmp_path / "lutan.zip"
    state_vectors = "\n".join(
        f"""
        <stateVec>
          <timeUTC>2026-01-01T00:00:{idx:02d}Z</timeUTC>
          <posX>{7000000 + idx}</posX><posY>{idx}</posY><posZ>{idx + 1}</posZ>
          <velX>1</velX><velY>2</velY><velZ>3</velZ>
        </stateVec>
        """
        for idx in range(2)
    )
    meta = f"""
    <product>
      <generalHeader><mission>LT1</mission><sensorMode>SM</sensorMode></generalHeader>
      <productInfo>
        <acquisitionInfo><sensor>LuTan-1</sensor><imagingMode>stripmap</imagingMode><lookDirection>RIGHT</lookDirection><polarisationMode>HH</polarisationMode></acquisitionInfo>
        <imageDataInfo><imageRaster><rowSpacing>2</rowSpacing><columnSpacing>3</columnSpacing><groundRangeResolution>4</groundRangeResolution><azimuthResolution>5</azimuthResolution><numberOfRows>10</numberOfRows><numberOfColumns>20</numberOfColumns></imageRaster></imageDataInfo>
        <sceneInfo>
          <sceneID>Lutan Zip</sceneID><start><timeUTC>2026-01-01T00:00:00Z</timeUTC></start><stop><timeUTC>2026-01-01T00:01:00Z</timeUTC></stop>
          <rangeTime><firstPixel>0.001</firstPixel><lastPixel>0.002</lastPixel></rangeTime>
          <sceneCenterCoord><lat>30</lat><lon>100</lon><incidenceAngle>35</incidenceAngle></sceneCenterCoord>
        </sceneInfo>
      </productInfo>
      <platform><orbit><orbitHeader><numStateVectors>2</numStateVectors></orbitHeader>{state_vectors}</orbit></platform>
      <instrument><radarParameters><centerFrequency>5405000000</centerFrequency></radarParameters><settings><settingRecord><PRF>1200</PRF></settingRecord></settings></instrument>
      <processing><processingParameter><rangeLooks>1</rangeLooks><azimuthLooks>1</azimuthLooks></processingParameter></processing>
    </product>
    """
    with zipfile.ZipFile(zip_path, "w") as zf:
        tiff_path = tmp_path / "product_SLC.tiff"
        data = np.random.randint(-32768, 32767, (10, 20, 2), dtype=np.int16)
        _write_multiband_tiff(tiff_path, data)
        zf.write(tiff_path, "LT/product_SLC.tiff")
        zf.writestr("LT/product.meta.xml", meta)

    project = Project.create(tmp_path / "project", name="project")
    result = LutanImporter(zip_path).import_to_project(project)

    with h5py.File(result.scene_path, "r") as h5:
        assert h5["slc"].attrs["storage"] == "zip"
        assert h5["slc"].attrs["member"] == "LT/product_SLC.tiff"
        assert h5["orbit"].attrs["smoothed"] == False
