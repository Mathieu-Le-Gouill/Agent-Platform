from __future__ import annotations

from faster_whisper.transcribe import Segment

from agent_platform.core.schemas.conversation import Utterance
from agent_platform.integrations.speech_to_text.faster_whisper.config import (
    FasterWhisperConfig,
)

__all__ = ["map_utterances"]


def map_utterances(
    segments: list[Segment], config: FasterWhisperConfig
) -> list[Utterance]:
    if config.word_timestamps:
        utterances = []
        for seg in segments:
            if not seg.words:
                utterances.append(
                    Utterance(
                        text=seg.text.strip(),
                        start_ms=int(seg.start * 1000),
                        end_ms=int(seg.end * 1000),
                    )
                )
                continue
            for word in seg.words:
                utterances.append(
                    Utterance(
                        text=word.word.strip(),
                        start_ms=int(word.start * 1000),
                        end_ms=int(word.end * 1000),
                        confidence=word.probability,
                    )
                )
        return utterances

    return [
        Utterance(
            text=seg.text.strip(),
            start_ms=int(seg.start * 1000),
            end_ms=int(seg.end * 1000),
        )
        for seg in segments
    ]
