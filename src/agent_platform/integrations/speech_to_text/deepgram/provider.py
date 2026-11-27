from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator
from typing import Any

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_timeout,
)
from agent_platform.core.errors import (
    ProviderError,
    ValidationError,
    error_logged,
    require_secret,
)
from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.core.schemas.enums import AudioFormat, DataType
from agent_platform.integrations.credentials import DeepgramCredentials
from agent_platform.integrations.sdk_errors import translate_sdk_error
from agent_platform.integrations.speech_to_text.deepgram.config import DeepgramConfig
from agent_platform.integrations.speech_to_text.deepgram.mappers import (
    parse_deepgram_result,
)
from agent_platform.integrations.speech_to_text.language import parse_language

_ERROR_MESSAGE = "Speech-to-text failed"
_translate = functools.partial(translate_sdk_error, operation=_ERROR_MESSAGE)

# Applied when neither `DeepgramConfig.timeout` nor `ClientOptions.timeout` is
# set. Long audio needs minutes, and the SDK's own default is 60 seconds.
DEFAULT_STT_TIMEOUT_SECONDS = 300.0

# Deepgram's names for headerless PCM samples
# (https://developers.deepgram.com/docs/encoding): linear32 is 32-bit float.
_RAW_ENCODINGS = {DataType.FLOAT32: "linear32", DataType.INT16: "linear16"}
_CONTAINER_MAGIC = (b"RIFF", b"fLaC", b"OggS", b"ID3")


def _is_adts_header(header: bytes) -> bool:
    # ISO/IEC 13818-7 ADTS: 12-bit sync 0xFFF, layer 00, sampling index < 13.
    return header[1] & 0xF6 == 0xF0 and (header[2] >> 2) & 0x0F < 13


def _is_mpeg_audio_header(header: bytes) -> bool:
    # ISO/IEC 11172-3 / 13818-3 frame header: 11-bit sync, then version,
    # layer, bitrate and sampling-rate fields, none of them reserved. Free
    # format (bitrate index 0) is refused too: int16 -1 followed by 0, common
    # in dithered silence, would otherwise pass.
    version = (header[1] >> 3) & 0x03
    layer = (header[1] >> 1) & 0x03
    bitrate = header[2] >> 4
    sample_rate = (header[2] >> 2) & 0x03
    return (
        header[1] & 0xE0 == 0xE0
        and version != 0b01
        and layer != 0b00
        and bitrate not in (0b0000, 0b1111)
        and sample_rate != 0b11
    )


def _is_containerized(chunk: AudioChunk) -> bool:
    # `chunk.format` alone cannot tell: `AudioDSP.build_chunks` tags raw samples
    # with the source file's format, so the bytes are checked first.
    data = chunk.data
    if data.startswith(_CONTAINER_MAGIC) or data[4:8] == b"ftyp":
        return True
    if chunk.format not in (AudioFormat.MP3, AudioFormat.AAC):
        return False
    if len(data) < 4 or data[0] != 0xFF:
        return False
    return _is_adts_header(data) or _is_mpeg_audio_header(data)


def _stream_audio_params(chunk: AudioChunk) -> dict[str, Any]:
    """Return the socket parameters that describe `chunk`'s audio.

    Deepgram reads the header of containerized audio and must not be given an
    encoding for it; headerless samples need `encoding` and `sample_rate`.
    """
    if _is_containerized(chunk):
        return {}
    encoding = _RAW_ENCODINGS.get(chunk.dtype)
    if encoding is None:
        raise ValidationError(
            f"Deepgram streaming has no encoding for raw {chunk.dtype} samples; "
            "send float32 or int16 PCM, or a containerized format"
        )
    if chunk.channels != 1:
        # Platform raw audio is planar (`AudioIO.from_tensor` writes [C, N]),
        # and Deepgram's docs do not state the layout it expects for raw
        # multi-channel audio, so only mono is sent.
        raise ValidationError(
            f"Deepgram streaming accepts mono raw audio only, got "
            f"{chunk.channels} channels; downmix first or send a containerized format"
        )
    return {
        "encoding": encoding,
        "sample_rate": chunk.sample_rate,
        "channels": chunk.channels,
    }


class DeepgramSTT(BaseSpeechToText[DeepgramConfig]):
    def __init__(
        self,
        credentials: DeepgramCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, DeepgramCredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> DeepgramConfig:
        return DeepgramConfig()

    def _api_key(self) -> str:
        api_key = require_secret(
            self._credentials.api_key, "Deepgram API key is required"
        )
        return api_key.get_secret_value()

    def _build_client(self, config: DeepgramConfig):
        from deepgram import AsyncDeepgramClient

        # `AsyncDeepgramClient`'s generated `BaseClient.__init__` (deepgram/base_client.py)
        # takes plain `timeout: float | None` / `max_retries: int | None` kwargs, so those
        # two resolve cleanly (the platform's `with_retry` is the only retry layer, so
        # the SDK makes none). `base_url` does not: the SDK has no single base-URL kwarg, only an `environment:
        # DeepgramClientEnvironment` object with 4 separate URLs (`base`/`production`/
        # `agent`/`agent_rest` for REST vs. websocket vs. the voice-agent endpoints), so a
        # single `ClientOptions.base_url` string has no faithful 1:1 translation. Left
        # unwired rather than guessing which of the 4 URLs it should populate.
        timeout = resolve_timeout(config.timeout, self._client_options)
        return AsyncDeepgramClient(
            api_key=self._api_key(),
            timeout=timeout if timeout is not None else DEFAULT_STT_TIMEOUT_SECONDS,
            max_retries=0,
        )

    @error_logged(re_raise=ProviderError, message=_ERROR_MESSAGE)
    @with_retry(translate=_translate)
    async def transcribe(
        self, audio: AudioChunk, config: DeepgramConfig | None = None
    ) -> Transcript:
        config = config or self._default_config()
        client = self._build_client(config)

        kwargs = dict(
            model=config.model,
            smart_format=config.smart_format,
            punctuate=config.punctuate,
            diarize=config.diarize,
            utterances=True,
        )
        if config.language:
            kwargs["language"] = config.language
        elif config.detect_language:
            kwargs["detect_language"] = True

        response = await client.listen.v1.media.transcribe_file(
            request=audio.data, **kwargs
        )

        channel = response.results.channels[0]
        language = parse_language(channel.detected_language or config.language or "en")

        utterances = []
        for alt in channel.alternatives:
            if not alt.words:
                text = alt.paragraphs.transcript if alt.paragraphs else ""
                utterances.append(Utterance(text=text, confidence=alt.confidence))
            else:
                for word in alt.words:
                    speaker = getattr(word, "speaker", None)
                    utterances.append(
                        Utterance(
                            text=word.word,
                            start_ms=int(word.start * 1000),
                            end_ms=int(word.end * 1000),
                            confidence=word.confidence,
                            speaker=str(speaker) if speaker is not None else None,
                        )
                    )

        return Transcript(
            utterances=utterances,
            language=language,
            metadata={"stt_provider": "deepgram", "model": config.model},
        )

    def stream(
        self, frames: AsyncIterator[AudioChunk], config: DeepgramConfig | None = None
    ) -> AsyncIterator[Transcript]:
        config = config or self._default_config()

        async def _stream() -> AsyncIterator[Transcript]:
            # The socket's audio parameters are fixed when it opens, so they
            # are read from the first frame.
            first = await anext(frames, None)
            if first is None:
                return
            audio_params = _stream_audio_params(first)
            client = self._build_client(config)

            connect_kwargs: dict[str, Any] = dict(
                model=config.model,
                smart_format=config.smart_format,
                punctuate=config.punctuate,
                utterance_end_ms="1000",
                **audio_params,
            )
            if config.language:
                connect_kwargs["language"] = config.language
            # `detect_language` is not supported for streaming, only pre-recorded audio.

            async with client.listen.v1.connect(**connect_kwargs) as socket:

                async def _send_frames() -> None:
                    await socket.send_media(first.data)
                    async for chunk in frames:
                        await socket.send_media(chunk.data)
                    await socket.send_close_stream()

                sender = asyncio.create_task(_send_frames())
                async for message in socket:
                    if isinstance(message, bytes):
                        continue
                    utterances = parse_deepgram_result(message.model_dump_json())
                    if utterances:
                        yield Transcript(
                            utterances=utterances,
                            metadata={"stt_provider": "deepgram", "streaming": True},
                        )
                await sender

        return _stream()
