from collections.abc import Sequence
from uuid import uuid4

import soundfile as sf

from agent_platform.core.interfaces.loader.audio.base import BaseAudioLoader
from agent_platform.core.interfaces.loader.audio.config import AudioLoaderConfig
from agent_platform.core.schemas.document import AudioDocument, DocumentMetadata
from agent_platform.core.schemas.enums import AudioFormat, FileFormat


class SoundFileLoader(BaseAudioLoader):
    async def load(
        self, source: str, config: AudioLoaderConfig | None = None
    ) -> Sequence[AudioDocument]:
        info = sf.info(source)
        data, sample_rate = sf.read(source, dtype="float32", always_2d=True)

        ff = FileFormat.from_path(source)
        if isinstance(ff.extension_format, AudioFormat):
            fmt = ff.extension_format
        else:
            fmt = AudioFormat.UNKNOWN

        return [
            AudioDocument(
                id=uuid4(),
                source=source,
                metadata=DocumentMetadata(
                    title=None,
                    extra={"subtype": info.subtype},
                ),
                content=data.tobytes(),
                format=fmt,
                sample_rate=sample_rate,
                channels=info.channels,
                duration=info.frames / sample_rate if sample_rate else None,
                subtype=info.subtype,
            )
        ]
