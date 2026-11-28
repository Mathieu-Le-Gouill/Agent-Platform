import pytest

from agent_platform.core.schemas.dimensions import bit_depth_for_mode


class TestBitDepthForMode:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [("RGB", 8), ("L", 8), ("I;16", 16), ("1", 1), ("UNKNOWN", None)],
    )
    def test_maps_pil_mode_to_bit_depth(self, mode, expected):
        assert bit_depth_for_mode(mode) == expected
