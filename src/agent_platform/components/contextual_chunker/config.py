from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ContextualChunkerConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    context_prompt_template: str = Field(
        default=(
            "You are generating retrieval context for a document chunk.\n\n"
            "Document:\n"
            "<document>\n"
            "{document}\n"
            "</document>\n\n"
            "Chunk:\n"
            "<chunk>\n"
            "{chunk}\n"
            "</chunk>\n\n"
            "Give a short, succinct context (1-2 sentences) to situate this "
            "chunk within the overall document, to improve search retrieval "
            "of the chunk. Answer only with the context, nothing else."
        )
    )
    max_document_chars: int = 8000
