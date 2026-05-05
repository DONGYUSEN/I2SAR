from pathlib import Path

from i2sar.interferometry.strip_insar import _detect_file_type


def test_detect_file_type_handles_temp_stripmap_inputs():
    p = Path("/home/ysdong/Temp/S1A_IW_SLC__1SDV_20230625T114146_20230625T114213_049142_05E8CA_CCD3.zip")
    assert _detect_file_type(str(p)) == "zip"
