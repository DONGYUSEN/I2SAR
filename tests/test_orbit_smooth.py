from i2sar.orbit.smooth import smooth_lutan_orbit


def _orbit(count: int) -> dict:
    return {
        "header": {"numStateVectors": count},
        "stateVectors": [
            {
                "timeUTC": f"2026-01-01T00:00:{idx:02d}Z",
                "gpsTime": float(idx),
                "posX": 7000000.0 + idx,
                "posY": 100.0 + idx,
                "posZ": 200.0 + idx,
                "velX": 1.0,
                "velY": 2.0,
                "velZ": 3.0,
            }
            for idx in range(count)
        ],
    }


def test_smooth_lutan_orbit_skips_when_state_vectors_insufficient():
    smoothed = smooth_lutan_orbit(_orbit(7))

    assert smoothed["smoothed"] is False
    assert smoothed["smoothing"]["status"] == "skipped-insufficient-state-vectors"
    assert smoothed["smoothing"]["state_vector_count"] == 7


def test_smooth_lutan_orbit_records_provenance_for_enough_vectors():
    smoothed = smooth_lutan_orbit(_orbit(12))

    assert smoothed["smoothed"] is True
    assert smoothed["smoothing"]["algorithm"] == "isce2-lutan1-orbit-filter"
    assert smoothed["smoothing"]["degree"] == 5
    assert smoothed["smoothing"]["sigma"] == 4.0
    assert smoothed["smoothing"]["max_iter"] == 3
    assert len(smoothed["stateVectors"]) == 12
