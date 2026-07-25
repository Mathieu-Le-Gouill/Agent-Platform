import io

import av
import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from agent_platform.core.schemas.document import (
    AudioDocument,
    ImageDocument,
    VideoDocument,
    _image_bit_depth,
    _infer_audio_format,
    _infer_image_format,
    _infer_video_format,
)
from agent_platform.core.schemas.enums import AudioFormat, ImageFormat, VideoFormat


@pytest.fixture
def png_path(tmp_path):
    path = tmp_path / "pixel.png"
    Image.new("RGB", (2, 3), color=(10, 20, 30)).save(path)
    return str(path)


@pytest.fixture
def rgba_png_path(tmp_path):
    path = tmp_path / "pixel_alpha.png"
    Image.new("RGBA", (2, 2), color=(1, 2, 3, 4)).save(path)
    return str(path)


@pytest.fixture
def wav_path(tmp_path):
    path = tmp_path / "tone.wav"
    data = np.zeros((100, 1), dtype=np.float32)
    sf.write(str(path), data, 16000, format="WAV")
    return str(path)


@pytest.fixture
def mp4_path(tmp_path):
    path = tmp_path / "clip.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=24)
    stream.width = 4
    stream.height = 4
    stream.pix_fmt = "yuv420p"
    for _ in range(2):
        frame = av.VideoFrame.from_ndarray(
            np.zeros((4, 4, 3), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return str(path)


class TestImageDocumentLoadContent:
    def test_loads_metadata_from_file(self, png_path):
        doc = ImageDocument.load_content(png_path)
        assert doc.source == png_path
        assert doc.format == ImageFormat.PNG
        assert doc.dimensions.width == 2
        assert doc.dimensions.height == 3
        assert doc.channels == 3
        assert doc.has_alpha is False
        assert doc.metadata.title == "pixel"

    def test_detects_alpha_channel(self, rgba_png_path):
        doc = ImageDocument.load_content(rgba_png_path)
        assert doc.has_alpha is True
        assert doc.channels == 4
        assert doc.color_space == "RGBA"

    def test_content_bytes_are_readable(self, png_path):
        doc = ImageDocument.load_content(png_path)
        assert len(doc.content) > 0
        with open(png_path, "rb") as f:
            assert doc.content == f.read()


class TestImageDocumentSaveContent:
    def test_save_with_known_format(self, png_path, tmp_path):
        doc = ImageDocument.load_content(png_path)
        out = tmp_path / "out"
        doc.save_content(str(out))
        saved = tmp_path / "out.png"
        assert saved.exists()
        with Image.open(saved) as img:
            assert img.size == (2, 3)

    def test_save_with_unknown_format_uses_pil_detection(self, tmp_path):
        buf = io.BytesIO()
        Image.new("RGB", (3, 3), color=(5, 5, 5)).save(buf, format="PNG")
        doc = ImageDocument(content=buf.getvalue(), format=ImageFormat.UNKNOWN)
        out = tmp_path / "detected.png"
        doc.save_content(str(out))
        assert out.exists()

    def test_save_appends_extension_when_missing(self, png_path, tmp_path):
        doc = ImageDocument.load_content(png_path)
        out = tmp_path / "no_ext"
        doc.save_content(str(out))
        assert (tmp_path / "no_ext.png").exists()


class TestAudioDocumentLoadContent:
    def test_loads_metadata_from_file(self, wav_path):
        doc = AudioDocument.load_content(wav_path)
        assert doc.source == wav_path
        assert doc.format == AudioFormat.WAV
        assert doc.sample_rate == 16000
        assert doc.channels == 1
        assert doc.duration == pytest.approx(100 / 16000)
        assert doc.metadata.title == "tone"

    def test_content_is_float32_bytes(self, wav_path):
        doc = AudioDocument.load_content(wav_path)
        arr = np.frombuffer(doc.content, dtype=np.float32)
        assert len(arr) == 100


class TestAudioDocumentSaveContent:
    def test_roundtrip_mono(self, wav_path, tmp_path):
        doc = AudioDocument.load_content(wav_path)
        out = tmp_path / "roundtrip.wav"
        doc.save_content(str(out))
        assert out.exists()
        data, sr = sf.read(str(out))
        assert sr == 16000
        assert len(data) == 100

    def test_roundtrip_multichannel(self, tmp_path):
        data = np.zeros((50, 2), dtype=np.float32)
        src = tmp_path / "stereo.wav"
        sf.write(str(src), data, 8000, format="WAV")
        doc = AudioDocument.load_content(str(src))
        assert doc.channels == 2

        out = tmp_path / "stereo_out.wav"
        doc.save_content(str(out))
        out_data, sr = sf.read(str(out), always_2d=True)
        assert sr == 8000
        assert out_data.shape[1] == 2

    def test_save_defaults_sample_rate_when_missing(self, tmp_path):
        doc = AudioDocument(
            content=np.zeros(10, dtype=np.float32).tobytes(),
            format=AudioFormat.WAV,
            sample_rate=None,
        )
        out = tmp_path / "default_rate.wav"
        doc.save_content(str(out))
        _, sr = sf.read(str(out))
        assert sr == 16000


class TestVideoDocumentLoadContent:
    def test_loads_metadata_from_file(self, mp4_path):
        doc = VideoDocument.load_content(mp4_path)
        assert doc.source == mp4_path
        assert doc.format == VideoFormat.MP4
        assert doc.dimensions.width == 4
        assert doc.dimensions.height == 4
        assert doc.codec == "mpeg4"
        assert doc.has_audio is False
        assert doc.audio_codec is None
        assert doc.metadata.title == "clip"

    def test_content_bytes_are_readable(self, mp4_path):
        doc = VideoDocument.load_content(mp4_path)
        with open(mp4_path, "rb") as f:
            assert doc.content == f.read()

    def test_no_video_stream_raises(self, tmp_path):
        wav = tmp_path / "not_video.wav"
        sf.write(str(wav), np.zeros((10, 1), dtype=np.float32), 16000, format="WAV")
        with pytest.raises(ValueError, match="No video stream"):
            VideoDocument.load_content(str(wav))


class TestVideoDocumentSaveContent:
    def test_save_writes_raw_content(self, mp4_path, tmp_path):
        doc = VideoDocument.load_content(mp4_path)
        out = tmp_path / "saved.mp4"
        doc.save_content(str(out))
        with open(out, "rb") as f:
            assert f.read() == doc.content


class TestFormatInferenceHelpers:
    def test_infer_image_format_known_extension(self):
        assert _infer_image_format("photo.png") == ImageFormat.PNG

    def test_infer_image_format_unknown_extension(self):
        assert _infer_image_format("photo.xyz") == ImageFormat.UNKNOWN

    def test_infer_audio_format_known_extension(self):
        assert _infer_audio_format("clip.wav") == AudioFormat.WAV

    def test_infer_audio_format_unknown_extension(self):
        assert _infer_audio_format("clip.xyz") == AudioFormat.UNKNOWN

    def test_infer_video_format_known_extension(self):
        assert _infer_video_format("movie.mp4") == VideoFormat.MP4

    def test_infer_video_format_unknown_extension(self):
        assert _infer_video_format("movie.xyz") == VideoFormat.UNKNOWN

    def test_image_bit_depth_rgb(self):
        assert _image_bit_depth("RGB") == 8
