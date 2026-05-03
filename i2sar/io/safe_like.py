from __future__ import annotations

from pathlib import Path
from typing import Any

from i2sar.core.enums import AcquisitionMode
from i2sar.core.ids import slugify_id
from i2sar.io.base import ImportResult, ParsedScene
from i2sar.io.scene_writer import write_parsed_scene
from i2sar.io.source import build_member_ref, list_product_members
from i2sar.io.xml import as_float, as_int, find_required, gps_seconds, read_xml_root, text
from i2sar.project import Project


class SafeLikeImporter:
    def __init__(self, source: str | Path, *, sensor: str, acquisition_mode: AcquisitionMode):
        self.source = Path(source)
        self.sensor = sensor
        self.acquisition_mode = acquisition_mode

    def discover_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        for member in list_product_members(self.source):
            low = member.lower()
            normalized = f"/{low}"
            if low.endswith(".xml") and "/annotation/" in normalized and "/calibration/" not in normalized:
                files["annotation"] = member
            elif low.endswith(".xml") and "/annotation/calibration/" in normalized:
                files["calibration"] = member
            elif low.endswith("manifest.safe"):
                files["manifest"] = member
            elif low.endswith((".tiff", ".tif")) and "/measurement/" in normalized:
                files["tiff"] = member
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
        slc_ref = build_member_ref(self.source, slc_member)
        return ParsedScene(
            scene_id=slugify_id(scene_id),
            sensor=self.sensor,
            acquisition_mode=self.acquisition_mode,
            acquisition_time=acquisition_time,
            acquisition={"source": self.sensor, **acquisition},
            scene=scene,
            radar_grid=radar_grid,
            orbit=orbit,
            orbit_raw=orbit,
            doppler=doppler,
            slc=slc_ref,
            slc_attrs={
                "format": "TIFF",
                "sample_format": "cint16",
                "storage_layout": "single_band_complex",
                "complex_band_count": 1,
                "processing_format": "single_band_cfloat32",
            },
            source_refs={"slc": slc_ref},
        )

    def parse(self) -> ParsedScene:
        files = self.discover_files()
        if "annotation" not in files:
            raise FileNotFoundError(f"No annotation XML found in {self.source}")
        if "tiff" not in files:
            raise FileNotFoundError(f"No measurement TIFF found in {self.source}")
        root = read_xml_root(self.source, files["annotation"])
        acquisition = self._extract_acquisition(root)
        scene = self._extract_scene_info(root)
        return self._parsed_from_parts(
            scene_id=self._scene_id(acquisition),
            acquisition_time=acquisition["startTimeUTC"],
            acquisition=acquisition,
            scene=scene,
            radar_grid=self._extract_radar_grid(root),
            orbit=self._extract_orbit(root),
            doppler=self._extract_doppler(root),
            slc_member=files["tiff"],
        )

    def import_to_project(self, project: Project) -> ImportResult:
        return write_parsed_scene(project, self.parse())

    def _scene_id(self, acquisition: dict[str, Any]) -> str:
        parts = [
            acquisition.get("mission") or self.sensor,
            acquisition.get("sensorMode") or self.acquisition_mode.value,
            acquisition.get("polarisation") or "pol",
            acquisition.get("startTimeUTC") or self.source.stem,
        ]
        return "_".join(part for part in parts if part)

    def _extract_acquisition(self, root) -> dict[str, Any]:
        header = find_required(root, "adsHeader")
        info = root.find("generalAnnotation/productInformation")
        image_info = find_required(root, "imageAnnotation/imageInformation")
        geo_pts = root.findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint")
        lats = [as_float(text(pt, "latitude")) for pt in geo_pts]
        lons = [as_float(text(pt, "longitude")) for pt in geo_pts]
        incs = [as_float(text(pt, "incidenceAngle")) for pt in geo_pts]
        look_raw = text(image_info, "look_side", "left").lower()
        return {
            "mission": text(header, "missionId"),
            "sensorMode": text(header, "mode"),
            "polarisation": text(header, "polarisation"),
            "imagingMode": text(header, "mode"),
            "lookDirection": "LEFT" if look_raw == "left" else "RIGHT",
            "centerFrequency": as_float(text(info, "radarFrequency")),
            "prf": as_float(text(info, "prf", text(image_info, "azimuthFrequency"))),
            "startTimeUTC": text(header, "startTime"),
            "stopTimeUTC": text(header, "stopTime"),
            "startGPSTime": gps_seconds(text(header, "startTime")),
            "stopGPSTime": gps_seconds(text(header, "stopTime")),
            "rangeTimeFirstPixel": as_float(text(image_info, "slantRangeTime")),
            "rangeTimeLastPixel": as_float(text(image_info, "slantRangeTime"))
            + (as_int(text(image_info, "numberOfSamples"), 1) - 1)
            * as_float(text(image_info, "rangePixelSpacing"))
            * 2.0
            / 299792458.0,
            "centerLat": sum(lats) / len(lats) if lats else 0.0,
            "centerLon": sum(lons) / len(lons) if lons else 0.0,
            "headingAngle": as_float(text(info, "platformHeading")),
            "sceneAverageHeight": 0.0,
            "incidenceAngleCenter": sum(incs) / len(incs) if incs else 0.0,
        }

    def _extract_scene_info(self, root) -> dict[str, Any]:
        corners = []
        for pt in root.findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint"):
            azimuth_time = text(pt, "azimuthTime")
            corners.append(
                {
                    "line": as_int(text(pt, "line")),
                    "pixel": as_int(text(pt, "pixel")),
                    "lat": as_float(text(pt, "latitude")),
                    "lon": as_float(text(pt, "longitude")),
                    "incidenceAngle": as_float(text(pt, "incidenceAngle")),
                    "slantRangeTime": as_float(text(pt, "slantRangeTime")),
                    "azimuthTime": azimuth_time,
                    "azimuthTimeUTC": azimuth_time,
                    "timeUTC": azimuth_time,
                }
            )
        return {"sceneCorners": corners}

    def _extract_orbit(self, root) -> dict[str, Any]:
        state_vectors = []
        for orbit in root.findall("generalAnnotation/orbitList/orbit"):
            timestamp = text(orbit, "time")
            state_vectors.append(
                {
                    "timeUTC": timestamp,
                    "gpsTime": gps_seconds(timestamp),
                    "posX": as_float(text(orbit, "position/x")),
                    "posY": as_float(text(orbit, "position/y")),
                    "posZ": as_float(text(orbit, "position/z")),
                    "velX": as_float(text(orbit, "velocity/x")),
                    "velY": as_float(text(orbit, "velocity/y")),
                    "velZ": as_float(text(orbit, "velocity/z")),
                }
            )
        spacing = state_vectors[1]["gpsTime"] - state_vectors[0]["gpsTime"] if len(state_vectors) >= 2 else 0.0
        first = state_vectors[0]["timeUTC"] if state_vectors else ""
        last = state_vectors[-1]["timeUTC"] if state_vectors else ""
        return {
            "header": {
                "generationSystem": "SAFE-like",
                "sensor": self.sensor,
                "accuracy": "unknown",
                "stateVectorRefFrame": "Earth Fixed",
                "stateVectorRefTime": first,
                "stateVecFormat": "ECEF",
                "numStateVectors": len(state_vectors),
                "firstStateTimeUTC": first,
                "lastStateTimeUTC": last,
                "stateVectorTimeSpacing": spacing,
            },
            "stateVectors": state_vectors,
        }

    def _extract_radar_grid(self, root) -> dict[str, Any]:
        header = find_required(root, "adsHeader")
        image_info = find_required(root, "imageAnnotation/imageInformation")
        first_range_time = as_float(text(image_info, "slantRangeTime"))
        range_spacing = as_float(text(image_info, "rangePixelSpacing"))
        columns = as_int(text(image_info, "numberOfSamples"))
        return {
            "numberOfRows": as_int(text(image_info, "numberOfLines")),
            "numberOfColumns": columns,
            "rowSpacing": as_float(text(image_info, "azimuthPixelSpacing")),
            "columnSpacing": range_spacing,
            "groundRangeResolution": range_spacing,
            "azimuthResolution": as_float(text(image_info, "azimuthPixelSpacing")),
            "prf": as_float(text(image_info, "azimuthFrequency")),
            "rangeLooks": 1,
            "azimuthLooks": 1,
            "geocodedFlag": "false",
            "rangeTimeFirstPixel": first_range_time,
            "rangeTimeLastPixel": first_range_time + (columns - 1) * range_spacing * 2.0 / 299792458.0,
            "sensing_start_s": gps_seconds(text(header, "startTime")),
        }

    def _extract_doppler(self, root) -> dict[str, Any]:
        first = root.find("dopplerCentroid/dcEstimateList/dcEstimate")
        if first is None:
            return {}
        coeffs = [as_float(item) for item in text(first, "dataDcPolynomial", "0").split()]
        image_info = find_required(root, "imageAnnotation/imageInformation")
        last_range_time = as_float(text(image_info, "slantRangeTime")) + (
            as_int(text(image_info, "numberOfSamples"), 1) - 1
        ) * as_float(text(image_info, "rangePixelSpacing")) * 2.0 / 299792458.0
        t0 = as_float(text(first, "t0"))
        return {
            "azDeskew": "FALSE",
            "dopplerBasebandEstimationMethod": "dcEstimateList",
            "dopplerGeometricEstimationMethod": "dcEstimateList",
            "dopplerCentroidCoordinateType": "RAW",
            "dopplerEstimate": {
                "timeUTC": text(first, "azimuthTime"),
                "dopplerAtMidRange": coeffs[0] if coeffs else 0.0,
                "dopplerAmbiguity": 0,
                "geometricDopplerFlag": "false",
            },
            "combinedDoppler": {
                "validityRangeMin": t0,
                "validityRangeMax": last_range_time,
                "referencePoint": t0,
                "polynomialDegree": len(coeffs) - 1,
                "coefficients": coeffs,
            },
        }
