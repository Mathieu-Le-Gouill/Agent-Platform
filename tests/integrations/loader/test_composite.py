import sys
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytest.importorskip(
    "agent_platform.integrations.loader.composite.auto",
    reason="composite loader not yet implemented",
)


@pytest.fixture(autouse=True)
def _mock_langchain_community(monkeypatch):
    mock_lc = MagicMock()
    mock_lc_loaders = MagicMock()
    mock_lc_unstructured = MagicMock()
    monkeypatch.setitem(sys.modules, "langchain_community", mock_lc)
    monkeypatch.setitem(
        sys.modules, "langchain_community.document_loaders", mock_lc_loaders
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_community.document_loaders.unstructured",
        mock_lc_unstructured,
    )


from agent_platform.core.schemas.document import (
    AudioDocument,
    ImageDocument,
    TextDocument,
    VideoDocument,
)
from agent_platform.core.schemas.enums import (
    AudioFormat,
    DocumentFormat,
    ImageFormat,
    VideoFormat,
)


class TestAutoLoader:
    def _make_mock_loader(self):
        inst = MagicMock()
        inst.load = AsyncMock()
        return inst

    @pytest.mark.asyncio
    async def test_auto_loader_dispatches_to_image_loader(self):
        from agent_platform.integrations.loader.composite.auto import AutoLoader

        loader = AutoLoader()
        loader._image_loader = self._make_mock_loader()
        mock_doc = ImageDocument(
            id=uuid4(), source="/test/image.png", format=ImageFormat.PNG
        )
        loader._image_loader.load.return_value = [mock_doc]

        results = await loader.load("/test/image.png")

        assert len(results) == 1
        assert results[0].format == ImageFormat.PNG
        loader._image_loader.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_loader_dispatches_to_audio_loader(self):
        from agent_platform.integrations.loader.composite.auto import AutoLoader

        loader = AutoLoader()
        loader._audio_loader = self._make_mock_loader()
        mock_doc = AudioDocument(
            id=uuid4(), source="/test/audio.wav", format=AudioFormat.WAV
        )
        loader._audio_loader.load.return_value = [mock_doc]

        results = await loader.load("/test/audio.wav")

        assert len(results) == 1
        assert results[0].format == AudioFormat.WAV
        loader._audio_loader.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_loader_dispatches_to_video_loader(self):
        from agent_platform.integrations.loader.composite.auto import AutoLoader

        loader = AutoLoader()
        loader._video_loader = self._make_mock_loader()
        mock_doc = VideoDocument(
            id=uuid4(), source="/test/video.mp4", format=VideoFormat.MP4
        )
        loader._video_loader.load.return_value = [mock_doc]

        results = await loader.load("/test/video.mp4")

        assert len(results) == 1
        assert results[0].format == VideoFormat.MP4
        loader._video_loader.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_loader_dispatches_to_text_loader_by_default(self):
        from agent_platform.integrations.loader.composite.auto import AutoLoader

        loader = AutoLoader()
        loader._text_loader = self._make_mock_loader()
        mock_doc = TextDocument(
            id=uuid4(), source="/test/doc.pdf", format=DocumentFormat.PDF
        )
        loader._text_loader.load.return_value = [mock_doc]

        results = await loader.load("/test/doc.pdf")

        assert len(results) == 1
        assert results[0].format == DocumentFormat.PDF
        loader._text_loader.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_many_yields_results(self):
        from agent_platform.integrations.loader.composite.auto import AutoLoader

        loader = AutoLoader()
        loader._image_loader = self._make_mock_loader()
        loader._text_loader = self._make_mock_loader()
        img_doc = ImageDocument(
            id=uuid4(), source="/test/img.png", format=ImageFormat.PNG
        )
        text_doc = TextDocument(
            id=uuid4(), source="/test/notes.txt", format=DocumentFormat.TXT
        )
        loader._image_loader.load.return_value = [img_doc]
        loader._text_loader.load.return_value = [text_doc]

        results = [
            docs
            async for docs in loader.load_many(["/test/img.png", "/test/notes.txt"])
        ]

        assert len(results) == 2
        assert results[0][0].format == ImageFormat.PNG
        assert results[1][0].format == DocumentFormat.TXT

    def test_parses_image_extension(self):
        from agent_platform.integrations.loader.composite.auto import AutoLoader

        loader = AutoLoader()
        sub = loader._loader_for("/test/image.png")
        assert sub is loader._image_loader

    def test_parses_audio_extension(self):
        from agent_platform.integrations.loader.composite.auto import AutoLoader

        loader = AutoLoader()
        sub = loader._loader_for("/test/sound.mp3")
        assert sub is loader._audio_loader

    def test_parses_video_extension(self):
        from agent_platform.integrations.loader.composite.auto import AutoLoader

        loader = AutoLoader()
        sub = loader._loader_for("/test/movie.mp4")
        assert sub is loader._video_loader

    def test_parses_text_extension(self):
        from agent_platform.integrations.loader.composite.auto import AutoLoader

        loader = AutoLoader()
        sub = loader._loader_for("/test/document.pdf")
        assert sub is loader._text_loader

    def test_unknown_extension_falls_back_to_text(self):
        from agent_platform.integrations.loader.composite.auto import AutoLoader

        loader = AutoLoader()
        sub = loader._loader_for("/test/file.xyz")
        assert sub is loader._text_loader


class TestMultiLoader:
    @pytest.mark.asyncio
    async def test_load_delegates_to_auto_loader(self):
        from agent_platform.integrations.loader.composite.multi import MultiLoader

        mock_auto = MagicMock()
        mock_auto.load = AsyncMock()
        mock_doc = TextDocument(id=uuid4(), source="/test/doc.txt")
        mock_auto.load.return_value = [mock_doc]

        loader = MultiLoader()
        loader._auto = mock_auto
        results = await loader.load("/test/doc.txt")

        assert len(results) == 1
        mock_auto.load.assert_called_once_with("/test/doc.txt", None)

    @pytest.mark.asyncio
    async def test_load_with_config_passthrough(self):
        from agent_platform.core.interfaces.loader.config import LoaderConfig
        from agent_platform.integrations.loader.composite.multi import MultiLoader

        mock_auto = MagicMock()
        mock_auto.load = AsyncMock()
        loader = MultiLoader()
        loader._auto = mock_auto
        config = LoaderConfig()
        await loader.load("/test/doc.txt", config=config)

        mock_auto.load.assert_called_once_with("/test/doc.txt", config)

    @pytest.mark.asyncio
    async def test_load_many_delegates(self):
        from agent_platform.integrations.loader.composite.multi import MultiLoader

        mock_auto = MagicMock()
        mock_auto.load_many = MagicMock()
        async_gen = AsyncMock()
        async_gen.__aiter__.return_value = iter(
            [[TextDocument(id=uuid4(), source="/a.txt")]]
        )
        mock_auto.load_many.return_value = async_gen

        loader = MultiLoader()
        loader._auto = mock_auto
        results = [docs async for docs in loader.load_many(["/a.txt"])]

        assert len(results) == 1
