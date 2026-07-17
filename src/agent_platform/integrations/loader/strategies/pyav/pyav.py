from typing import Sequence
from uuid import uuid4

import av

from agent_platform.core.interfaces.loader.video.base import BaseVideoLoader
from agent_platform.core.interfaces.loader.video.config import VideoLoaderConfig
from agent_platform.core.schemas.dimensions import Dimensions
from agent_platform.core.schemas.document import VideoDocument, DocumentMetadata
from agent_platform.core.schemas.enums import VideoFormat, FileFormat


class PyAVLoader(BaseVideoLoader):
    async def load(
        self, source: str, config: VideoLoaderConfig | None = None
    ) -> Sequence[VideoDocument]:
        ff = FileFormat.from_path(source)
        if isinstance(ff.extension_format, VideoFormat):
            video_fmt = ff.extension_format
        else:
            video_fmt = VideoFormat.UNKNOWN

        with av.open(source) as container:
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            audio_stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )

            if video_stream is None:
                return []

            duration_sec = (
                float(container.duration / av.utils.time_base)
                if container.duration
                else None
            )
            frame_rate = (
                float(video_stream.average_rate) if video_stream.average_rate else None
            )
            width = video_stream.width
            height = video_stream.height
            codec = video_stream.codec.name if video_stream.codec else None
            bitrate = video_stream.bit_rate

            has_audio = audio_stream is not None
            audio_codec = (
                audio_stream.codec.name if audio_stream and audio_stream.codec else None
            )
            audio_channels = audio_stream.channels if audio_stream else None
            audio_sample_rate = audio_stream.rate if audio_stream else None
            audio_format_name = (
                audio_stream.format.name
                if audio_stream and audio_stream.format
                else None
            )

        with open(source, "rb") as f:
            raw_bytes = f.read()

        return [
            VideoDocument(
                id=uuid4(),
                source=source,
                metadata=DocumentMetadata(
                    extra={"audio_format": audio_format_name}
                    if audio_format_name
                    else {},
                ),
                content=raw_bytes,
                format=video_fmt,
                duration=duration_sec,
                dimensions=Dimensions(width=width, height=height),
                frame_rate=frame_rate,
                codec=codec,
                bitrate=bitrate,
                has_audio=has_audio,
                audio_codec=audio_codec,
                audio_channels=audio_channels,
                audio_sample_rate=audio_sample_rate,
            )
        ]
