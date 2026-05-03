"""Tests for LookSide enum."""

from i2sar.core.enums import LookSide


class TestLookSide:
    def test_look_side_left(self):
        assert LookSide.LEFT == "left"
        assert str(LookSide.LEFT) == "left"

    def test_look_side_right(self):
        assert LookSide.RIGHT == "right"
        assert str(LookSide.RIGHT) == "right"

    def test_look_side_iteration(self):
        values = list(LookSide)
        assert len(values) == 2
        assert LookSide.LEFT in values
        assert LookSide.RIGHT in values

    def test_look_side_from_string(self):
        assert LookSide("left") is LookSide.LEFT
        assert LookSide("right") is LookSide.RIGHT
