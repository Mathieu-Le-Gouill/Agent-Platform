from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import numpy as np
import whisperx

from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.integrations.speech_to_text.whisperx.config import WhisperXConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.integrations.speech_to_text.utils import parse_language


class WhisperXSTT(BaseSpeechToText[WhisperXConfig]):
    def __init__(self) -> None:
        self._model: Any = None
        self._align_models: dict[str, tuple[Any, Any]] = {}
        self._diarize_pipeline: Any = None

    def _default_config(self) -> WhisperXConfig:
        return WhisperXConfig()

    def _load_model_sync(self, config: WhisperXConfig) -> Any:
        return whisperx.load_model(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )

    async def _load_model(self, config: WhisperXConfig) -> Any:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model_sync, config)
        return self._model

    def _load_align_model_sync(self, language: str, device: str) -> tuple[Any, Any]:
        if language not in self._align_models:
            self._align_models[language] = whisperx.load_align_model(
                language_code=language, device=device
            )
        return self._align_models[language]

    async def _ensure_align_model(self, language: str, device: str) -> tuple[Any, Any]:
        return await asyncio.to_thread(self._load_align_model_sync, language, device)

    async def _align(
        self, result: dict, audio_np: np.ndarray, language: str, config: WhisperXConfig
    ) -> dict:
        model_a, metadata = await self._ensure_align_model(language, config.device)
        return await asyncio.to_thread(
            whisperx.align,
            result["segments"],
            model_a,
            metadata,
            audio_np,
            config.device,
            return_char_alignments=False,
        )

    def _load_diarize_pipeline_sync(self, config: WhisperXConfig) -> Any:
        from whisperx.diarize import DiarizationPipeline

        if self._diarize_pipeline is None:
            self._diarize_pipeline = DiarizationPipeline(
                use_auth_token=config.hf_token, device=config.device
            )
        return self._diarize_pipeline

    async def _diarize(
        self, audio_np: np.ndarray, result: dict, config: WhisperXConfig
    ) -> dict:
        from whisperx.diarize import assign_word_speakers

        pipeline = await asyncio.to_thread(self._load_diarize_pipeline_sync, config)
        diarize_df = await asyncio.to_thread(
            pipeline,
            audio_np,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
        )
        return await asyncio.to_thread(assign_word_speakers, diarize_df, result)

    async def transcribe(
        self, audio: AudioChunk, config: WhisperXConfig | None = None
    ) -> Transcript:
        config = config or self._default_config()
        model = await self._load_model(config)

        audio_np = np.frombuffer(audio.data, dtype=np.float32).reshape(1, -1)

        result = await asyncio.to_thread(
            model.transcribe,
            audio_np,
            batch_size=config.batch_size,
            language=config.language,
        )

        result = await self._postprocess(audio_np, result, config)

        return _map_transcript(result, config)

    def stream(
        self, frames: AsyncIterator[AudioChunk], config: WhisperXConfig | None = None
    ) -> AsyncIterator[Transcript]:
        config = config or self._default_config()

        async def _stream() -> AsyncIterator[Transcript]:
            model = await self._load_model(config)

            buffer: list[AudioChunk] = []
            buffer_ms = 0

            async for chunk in frames:
                buffer.append(chunk)
                buffer_ms += chunk.end - chunk.start

                if buffer_ms < config.min_duration_ms:
                    continue

                yield await self._transcribe_buffer(model, buffer, config)
                buffer.clear()
                buffer_ms = 0

            if buffer:
                yield await self._transcribe_buffer(model, buffer, config)

        return _stream()

    async def _postprocess(
        self, audio_np: np.ndarray, result: dict, config: WhisperXConfig
    ) -> dict:
        language = result.get("language") or config.language
        if config.align and language:
            result = await self._align(result, audio_np, language, config)
            result.setdefault("language", language)

        if config.diarize:
            result = await self._diarize(audio_np, result, config)
            result.setdefault("language", language)

        return result

    async def _transcribe_buffer(
        self, model: Any, buffer: list[AudioChunk], config: WhisperXConfig
    ) -> Transcript:
        audio_np = np.concatenate(
            [np.frombuffer(c.data, dtype=np.float32) for c in buffer]
        ).reshape(1, -1)

        result = await asyncio.to_thread(
            model.transcribe,
            audio_np,
            batch_size=config.batch_size,
            language=config.language,
        )

        result = await self._postprocess(audio_np, result, config)

        return _map_transcript(result, config)


def _map_transcript(result: dict, config: WhisperXConfig) -> Transcript:
    language = parse_language(result.get("language", "") or "")

    utterances = []
    for seg in result.get("segments", []):
        words = seg.get("words") or []
        scores = [w["score"] for w in words if w.get("score") is not None]
        confidence = sum(scores) / len(scores) if scores else None
        speaker = seg.get("speaker")
        utterances.append(
            Utterance(
                text=seg["text"].strip(),
                start_ms=int(seg.get("start", 0) * 1000),
                end_ms=int(seg.get("end", 0) * 1000),
                confidence=confidence,
                speaker=speaker,
            )
        )

    return Transcript(
        utterances=utterances,
        language=language,
        metadata={"stt_provider": "whisperx", "model": config.model_size},
    )
