from agent_platform.core.schemas.dimensions import bit_depth_for_mode as _get_bit_depth


class TestGetBitDepth:
    async def test_rgb_8bit(self):
        assert _get_bit_depth("RGB") == 8

    async def test_grayscale_8bit(self):
        assert _get_bit_depth("L") == 8

    async def test_grayscale_16bit(self):
        assert _get_bit_depth("I;16") == 16

    async def test_binary_1bit(self):
        assert _get_bit_depth("1") == 1

    async def test_unknown_returns_none(self):
        assert _get_bit_depth("UNKNOWN") is None
