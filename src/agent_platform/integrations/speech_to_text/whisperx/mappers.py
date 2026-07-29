from __future__ import annotations

from typing import Any

from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.integrations.speech_to_text.language import parse_language
from agent_platform.integrations.speech_to_text.whisperx.config import WhisperXConfig

__all__ = ["map_transcript"]


def map_transcript(result: dict[str, Any], config: WhisperXConfig) -> Transcript:
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
