from __future__ import annotations

import json

from agent_platform.core.schemas.conversation import Utterance

__all__ = ["parse_deepgram_result"]


def parse_deepgram_result(raw: str) -> list[Utterance]:
    data = json.loads(raw)
    channel = data.get("channel", {})
    alternatives = channel.get("alternatives", [])
    if not alternatives:
        return []

    alt = alternatives[0]
    transcript_text = alt.get("transcript", "").strip()
    if not transcript_text:
        return []

    words = alt.get("words", [])
    if words:
        return [
            Utterance(
                text=w["word"],
                start_ms=int(w["start"] * 1000),
                end_ms=int(w["end"] * 1000),
                confidence=w.get("confidence"),
            )
            for w in words
        ]

    return [Utterance(text=transcript_text)]
