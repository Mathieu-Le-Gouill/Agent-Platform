from unittest.mock import MagicMock
from uuid import uuid4

import pytest


@pytest.fixture
def mock_pil_image():
    from PIL import Image

    img = MagicMock(spec=Image.Image)
    img.mode = "RGB"
    img.size = (640, 480)
    img.info = {}
    img.getexif.return_value = {}
    return img


@pytest.fixture
def fake_image_document():
    from agent_platform.core.schemas.dimensions import Dimensions
    from agent_platform.core.schemas.document import ImageDocument

    return ImageDocument(
        id=uuid4(),
        source="/test/image.png",
        content=b"fake_image_bytes",
        dimensions=Dimensions(width=640, height=480, depth=8),
        color_space="RGB",
        channels=3,
    )


@pytest.fixture
def fake_audio_document():
    from agent_platform.core.schemas.document import AudioDocument

    return AudioDocument(
        id=uuid4(),
        source="/test/audio.wav",
        content=b"fake_audio_bytes",
        sample_rate=44100,
        channels=2,
        duration=5.0,
        subtype="FLOAT",
    )


@pytest.fixture
def fake_video_document():
    from agent_platform.core.schemas.dimensions import Dimensions
    from agent_platform.core.schemas.document import VideoDocument

    return VideoDocument(
        id=uuid4(),
        source="/test/video.mp4",
        content=b"fake_video_bytes",
        dimensions=Dimensions(width=1920, height=1080),
        duration=30.0,
        frame_rate=24.0,
        codec="h264",
    )
