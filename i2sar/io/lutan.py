from __future__ import annotations

from pathlib import Path
from typing import Any

from i2sar.core.enums import AcquisitionMode
from i2sar.core.ids import slugify_id
from i2sar.io.base import ImportResult, ParsedScene
from i2sar.io.scene_writer import write_parsed_scene
from i2sar.io.source import build_member_ref, list_product_members
from i2sar.io.xml import as_float, as_int, find_required, gps_seconds, read_xml_root, text
from i2sar.orbit.smooth import smooth_lutan_orbit
from i2sar.project import Project


class LutanImporter:
    def __init__(self, source: str | Path):
        self.source = Path(source)

    def discover_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        for member in list_product_members(self.source):
            upper = member.upper()
            lower = member.lower()
            if lower.endswith((".tiff", ".tif")):
                if "SLC" in upper:
                    files["tiff"] = member
                elif "tiff" not in files:
                    files["tiff"] = member
            elif lower.endswith(".meta.xml"):
                files["meta_xml"] = member
            elif lower.endswith(".incidence.xml"):
                files["incidence_xml"] = member
            elif lower.endswith(".rpc"):
                files["rpc"] = member
        return files

    def _parsed_from_parts(
        self,
        *,
        scene_id: str,
        acquisition_time: str,
        acquisition: dict[str, Any],
        scene: dict[str, Any],
        radar_grid: dict[str, Any],
        orbit: dict[str, Any],
        doppler: dict[str, Any],
        slc_member: str,
    ) -> ParsedScene:
        raw_orbit = orbit
        smoothed_orbit = smooth_lutan_orbit(orbit)
        slc_ref = build_member_ref(self.source, slc_member)
        return ParsedScene(
            scene_id=slugify_id(scene_id),
            sensor="lutan",
            acquisition_mode=AcquisitionMode.STRIPMAP,
            acquisition_time=acquisition_time,
            acquisition={"source": "lutan", **acquisition},
            scene=scene,
            radar_grid=radar_grid,
            orbit=smoothed_orbit,
            orbit_raw=raw_orbit,
            doppler=doppler,
            slc=slc_ref,
            slc_attrs={
                "format": "TIFF",
                "sample_format": "iq_int16",
                "storage_layout": "two_band_iq",
                "complex_band_count": 1,
            },
            source_refs={"slc": slc_ref},
        )

    def parse(self) -> ParsedScene:
        files = self.discover_files()
        if "meta_xml" not in files:
            raise FileNotFoundError(f"No .meta.xml found in {self.source}")
        if "tiff" not in files:
            raise FileNotFoundError(f"No SLC TIFF found in {self.source}")
        root = read_xml_root(self.source, files["meta_xml"])
        general = self._extract_general(root)
        acq_info = self._extract_acquisition_info(root)
        image_info = self._extract_image_data_info(root)
        scene = self._extract_scene_info(root)
        radar = self._extract_radar_params(root)
        processing = self._extract_processing_info(root)
        acquisition = {
            "mission": general["mission"],
            "sensorMode": general["sensorMode"],
            "polarisation": acq_info["polarisationMode"],
            "imagingMode": acq_info["imagingMode"],
            "lookDirection": acq_info["lookDirection"],
            "centerFrequency": radar["centerFrequency"],
            "prf": radar["prf"],
            "startTimeUTC": scene["startTimeUTC"],
            "stopTimeUTC": scene["stopTimeUTC"],
            "startGPSTime": gps_seconds(scene["startTimeUTC"]),
            "stopGPSTime": gps_seconds(scene["stopTimeUTC"]),
            "rangeTimeFirstPixel": scene["rangeTimeFirstPixel"],
            "rangeTimeLastPixel": scene["rangeTimeLastPixel"],
            "centerLat": scene["sceneCenterCoord"].get("lat", 0.0),
            "centerLon": scene["sceneCenterCoord"].get("lon", 0.0),
            "headingAngle": scene["headingAngle"],
            "sceneAverageHeight": scene["sceneAverageHeight"],
            "incidenceAngleCenter": scene["sceneCenterCoord"].get("incidenceAngle", 0.0),
        }
        radar_grid = {
            "numberOfRows": image_info["numberOfRows"],
            "numberOfColumns": image_info["numberOfColumns"],
            "rowSpacing": image_info["rowSpacing"],
            "columnSpacing": image_info["columnSpacing"],
            "groundRangeResolution": image_info["groundRangeResolution"],
            "azimuthResolution": image_info["azimuthResolution"],
            "prf": radar["prf"],
            "rangeLooks": processing["rangeLooks"],
            "azimuthLooks": processing["azimuthLooks"],
            "geocodedFlag": processing["geocodedFlag"],
            "rangeTimeFirstPixel": scene["rangeTimeFirstPixel"],
            "rangeTimeLastPixel": scene["rangeTimeLastPixel"],
        }
        scene_payload = {
            "sceneCenterCoord": scene["sceneCenterCoord"],
            "sceneCorners": list(scene["sceneCorners"].values()),
            "sceneAverageHeight": scene["sceneAverageHeight"],
            "headingAngle": scene["headingAngle"],
        }
        return self._parsed_from_parts(
            scene_id=scene["sceneID"],
            acquisition_time=scene["startTimeUTC"],
            acquisition=acquisition,
            scene=scene_payload,
            radar_grid=radar_grid,
            orbit=self._extract_orbit(root),
            doppler=self._extract_doppler(root),
            slc_member=files["tiff"],
        )

    def import_to_project(self, project: Project) -> ImportResult:
        return write_parsed_scene(project, self.parse())

    def _extract_general(self, root) -> dict[str, Any]:
        gh = find_required(root, "generalHeader")
        return {
            "mission": text(gh, "mission"),
            "sensorMode": text(gh, "sensorMode"),
            "generationTime": text(gh, "generationTime"),
        }

    def _extract_acquisition_info(self, root) -> dict[str, Any]:
        ai = find_required(root, "productInfo/acquisitionInfo")
        return {
            "sensor": text(ai, "sensor"),
            "imagingMode": text(ai, "imagingMode"),
            "lookDirection": text(ai, "lookDirection"),
            "polarisationMode": text(ai, "polarisationMode"),
            "elevationBeamConfiguration": text(ai, "elevationBeamConfiguration"),
        }

    def _extract_image_data_info(self, root) -> dict[str, Any]:
        raster = find_required(root, "productInfo/imageDataInfo/imageRaster")
        return {
            "rowSpacing": as_float(text(raster, "rowSpacing")),
            "columnSpacing": as_float(text(raster, "columnSpacing")),
            "groundRangeResolution": as_float(text(raster, "groundRangeResolution")),
            "azimuthResolution": as_float(text(raster, "azimuthResolution")),
            "numberOfRows": as_int(text(raster, "numberOfRows")),
            "numberOfColumns": as_int(text(raster, "numberOfColumns")),
        }

    def _extract_scene_info(self, root) -> dict[str, Any]:
        si = find_required(root, "productInfo/sceneInfo")
        center = find_required(si, "sceneCenterCoord")
        corners = {}
        for corner in si.findall("sceneCornerCoord"):
            name = corner.get("name", f"corner_{len(corners)}")
            corners[name] = {
                "refRow": as_int(text(corner, "refRow")),
                "refColumn": as_int(text(corner, "refColumn")),
                "lat": as_float(text(corner, "lat")),
                "lon": as_float(text(corner, "lon")),
                "azimuthTimeUTC": text(corner, "azimuthTimeUTC"),
                "rangeTime": as_float(text(corner, "rangeTime")),
                "incidenceAngle": as_float(text(corner, "incidenceAngle")),
            }
        return {
            "sceneID": text(si, "sceneID", self.source.stem),
            "startTimeUTC": text(si.find("start"), "timeUTC"),
            "stopTimeUTC": text(si.find("stop"), "timeUTC"),
            "rangeTimeFirstPixel": as_float(text(si.find("rangeTime"), "firstPixel")),
            "rangeTimeLastPixel": as_float(text(si.find("rangeTime"), "lastPixel")),
            "sceneCenterCoord": {
                "refRow": as_int(text(center, "refRow")),
                "refColumn": as_int(text(center, "refColumn")),
                "lat": as_float(text(center, "lat")),
                "lon": as_float(text(center, "lon")),
                "azimuthTimeUTC": text(center, "azimuthTimeUTC"),
                "rangeTime": as_float(text(center, "rangeTime")),
                "incidenceAngle": as_float(text(center, "incidenceAngle")),
            },
            "sceneAverageHeight": as_float(text(si, "sceneAverageHeight")),
            "sceneCorners": corners,
            "headingAngle": as_float(text(si, "headingAngle")),
        }

    def _extract_orbit(self, root) -> dict[str, Any]:
        orbit_elem = find_required(root, "platform/orbit")
        header = find_required(orbit_elem, "orbitHeader")
        state_vectors = []
        for sv in orbit_elem.findall("stateVec"):
            time_utc = text(sv, "timeUTC")
            state_vectors.append(
                {
                    "timeUTC": time_utc,
                    "gpsTime": gps_seconds(time_utc),
                    "posX": as_float(text(sv, "posX")),
                    "posY": as_float(text(sv, "posY")),
                    "posZ": as_float(text(sv, "posZ")),
                    "velX": as_float(text(sv, "velX")),
                    "velY": as_float(text(sv, "velY")),
                    "velZ": as_float(text(sv, "velZ")),
                }
            )
        return {
            "header": {
                "generationSystem": text(header, "generationSystem"),
                "sensor": text(header, "sensor"),
                "accuracy": text(header, "accuracy"),
                "stateVectorRefFrame": text(header, "stateVectorRefFrame"),
                "stateVectorRefTime": text(header, "stateVectorRefTime"),
                "stateVecFormat": text(header, "stateVecFormat"),
                "numStateVectors": as_int(text(header, "numStateVectors"), len(state_vectors)),
                "firstStateTimeUTC": text(header.find("firstStateTime"), "firstStateTimeUTC"),
                "lastStateTimeUTC": text(header.find("lastStateTime"), "lastStateTimeUTC"),
                "stateVectorTimeSpacing": as_float(text(header, "stateVectorTimeSpacing")),
            },
            "stateVectors": state_vectors,
        }

    def _extract_doppler(self, root) -> dict[str, Any]:
        doppler = root.find("processing/doppler")
        if doppler is None:
            return {}
        estimate = doppler.find("dopplerCentroid/dopplerEstimate")
        combined = estimate.find("combinedDoppler") if estimate is not None else None
        degree = as_int(text(combined, "polynomialDegree")) if combined is not None else 0
        return {
            "azDeskew": text(doppler, "azDeskew"),
            "dopplerBasebandEstimationMethod": text(doppler, "dopplerBasebandEstimationMethod"),
            "dopplerGeometricEstimationMethod": text(doppler, "dopplerGeometricEstimationMethod"),
            "dopplerCentroidCoordinateType": text(doppler, "dopplerCentroidCoordinateType"),
            "dopplerEstimate": {
                "timeUTC": text(estimate, "timeUTC"),
                "dopplerAtMidRange": as_float(text(estimate, "dopplerAtMidRange")),
                "dopplerAmbiguity": as_int(text(estimate, "dopplerAmbiguity")),
                "geometricDopplerFlag": text(estimate, "geometricDopplerFlag"),
            },
            "combinedDoppler": {
                "validityRangeMin": as_float(text(combined, "validityRangeMin")),
                "validityRangeMax": as_float(text(combined, "validityRangeMax")),
                "referencePoint": as_float(text(combined, "referencePoint")),
                "polynomialDegree": degree,
                "coefficients": [as_float(text(combined, f"coefficient[@exponent='{idx}']")) for idx in range(degree + 1)],
            },
        }

    def _extract_radar_params(self, root) -> dict[str, Any]:
        radar = root.find("instrument/radarParameters")
        settings = root.find("instrument/settings")
        return {
            "centerFrequency": as_float(text(radar, "centerFrequency")),
            "prf": as_float(text(settings, "settingRecord/PRF")),
        }

    def _extract_processing_info(self, root) -> dict[str, Any]:
        proc = root.find("processing")
        return {
            "rangeLooks": as_int(text(proc, "processingParameter/rangeLooks"), 1),
            "azimuthLooks": as_int(text(proc, "processingParameter/azimuthLooks"), 1),
            "geocodedFlag": text(proc, "processingFlags/geocodedFlag"),
        }
