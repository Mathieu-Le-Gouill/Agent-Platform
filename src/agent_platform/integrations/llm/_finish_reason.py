from __future__ import annotations

from agent_platform.core.schemas.enums import FinishReason

__all__ = ["to_finish_reason"]

# Vendors spell the same handful of outcomes differently ("end_turn" vs
# "stop", "tool_use" vs "tool_calls", "max_tokens" vs "length"), so one table
# covers every provider instead of six near-identical ones. A provider whose
# raw value is missing here reports UNKNOWN rather than a plausible-looking
# STOP, since a wrong finish reason is worse than an absent one.
_FINISH_REASONS: dict[str, FinishReason] = {
    "complete": FinishReason.STOP,
    "end_turn": FinishReason.STOP,
    "eos_token": FinishReason.STOP,
    "stop": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP_SEQUENCE,
    "length": FinishReason.LENGTH,
    "max_output_tokens": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
    "model_length": FinishReason.LENGTH,
    "function_call": FinishReason.TOOL_CALL,
    "tool_call": FinishReason.TOOL_CALL,
    "tool_calls": FinishReason.TOOL_CALL,
    "tool_use": FinishReason.TOOL_CALL,
    "blocklist": FinishReason.CONTENT_FILTER,
    "content_filter": FinishReason.CONTENT_FILTER,
    "prohibited_content": FinishReason.CONTENT_FILTER,
    "recitation": FinishReason.CONTENT_FILTER,
    "refusal": FinishReason.CONTENT_FILTER,
    "safety": FinishReason.CONTENT_FILTER,
    "error": FinishReason.ERROR,
    "malformed_function_call": FinishReason.ERROR,
}


def to_finish_reason(raw: object) -> FinishReason:
    """Translate a vendor's raw stop reason into `FinishReason`.

    Accepts whatever the SDK hands over (a string, an enum, `None`): SDKs
    disagree on the type as much as on the spelling, and a call site should
    not have to normalize before asking.
    """
    if raw is None:
        return FinishReason.UNKNOWN
    text = raw.value if hasattr(raw, "value") else str(raw)
    if not isinstance(text, str):
        text = str(text)
    return _FINISH_REASONS.get(text.strip().lower(), FinishReason.UNKNOWN)
