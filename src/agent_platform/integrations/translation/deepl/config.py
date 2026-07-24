from __future__ import annotations

from agent_platform.core.interfaces.translation.config import TranslationConfig


class DeepLConfig(TranslationConfig):
    # Controls tone of the translation: "less"/"more" (informal/formal), "prefer_less"/"prefer_more",
    # or "default". Only supported for a subset of target languages.
    formality: str | None = None
    # If True, preserves original formatting such as punctuation and capitalization even when it
    # deviates from target-language conventions.
    preserve_formatting: bool | None = None
    # Additional text passed for context to improve translation quality; not itself translated
    # or included in the output.
    context: str | None = None
    # Selects which translation model to use, e.g. "quality_optimized", "prefer_quality_optimized",
    # or "latency_optimized".
    model_type: str | None = None
    # ID of a glossary to apply for the translation; source/target languages must match the glossary.
    glossary_id: str | None = None
    # Controls how input text is split into sentences: "0" (no splitting), "1" (split on
    # punctuation and newlines, default), or "nonewlines" (split on punctuation only).
    split_sentences: str | None = None
    # Specifies how tags in the input text are handled, e.g. "xml" or "html".
    tag_handling: str | None = None


# sources: https://developers.deepl.com/docs/api-reference/translate
