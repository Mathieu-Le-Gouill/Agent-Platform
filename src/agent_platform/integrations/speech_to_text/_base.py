from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript


async def buffered_stream(
    frames: AsyncIterator[AudioChunk],
    min_duration_ms: float,
    transcribe_buffer: Callable[[list[AudioChunk]], Awaitable[Transcript]],
) -> AsyncIterator[Transcript]:
    """Batch incoming frames into windows of at least `min_duration_ms` and
    transcribe each window as it fills, flushing whatever remains at the end.

    Shared by providers whose "streaming" mode is really client-side batching
    over their batch `transcribe`-style call (whisperx, faster_whisper,
    openai); providers with a genuine low-latency streaming protocol
    (deepgram) implement `stream()` directly instead.
    """
    buffer: list[AudioChunk] = []
    buffer_ms = 0.0

    async for chunk in frames:
        buffer.append(chunk)
        buffer_ms += chunk.end - chunk.start

        if buffer_ms < min_duration_ms:
            continue

        yield await transcribe_buffer(buffer)
        buffer = []
        buffer_ms = 0.0

    if buffer:
        yield await transcribe_buffer(buffer)
