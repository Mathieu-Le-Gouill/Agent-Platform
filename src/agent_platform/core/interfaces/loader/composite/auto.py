import os
from typing import AsyncIterator, Sequence

from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.interfaces.loader.text.strategies.unstructured import (
    UnstructuredFileLoader,
)
from agent_platform.core.interfaces.loader.image.strategies.pil import PILImageLoader
from agent_platform.core.interfaces.loader.audio.strategies.soundfile import (
    SoundFileLoader,
)
from agent_platform.core.interfaces.loader.video.strategies.pyav import PyAVLoader
from agent_platform.core.schemas.document import Document
from agent_platform.core.schemas.enums import FileFormat, MediaType


class AutoLoader:
    """Dispatches to the correct loader based on file extension / media type."""

    def __init__(self) -> None:
        self._text_loader = UnstructuredFileLoader()
        self._image_loader = PILImageLoader()
        self._audio_loader = SoundFileLoader()
        self._video_loader = PyAVLoader()

    def _loader_for(self, source: str):
        ext = os.path.splitext(source)[1].lstrip(".")
        fmt = FileFormat.from_extension(ext)
        match fmt.media_type:
            case MediaType.IMAGE:
                return self._image_loader
            case MediaType.AUDIO:
                return self._audio_loader
            case MediaType.VIDEO:
                return self._video_loader
            case _:
                return self._text_loader

    async def load(
        self, source: str, config: LoaderConfig | None = None
    ) -> Sequence[Document]:
        loader = self._loader_for(source)
        return await loader.load(source, config)

    async def load_many(
        self,
        sources: list[str],
        config: LoaderConfig | None = None,
    ) -> AsyncIterator[Sequence[Document]]:
        for source in sources:
            loader = self._loader_for(source)
            results = await loader.load(source, config)
            yield results
