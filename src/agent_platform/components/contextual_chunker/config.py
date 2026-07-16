from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field


class ContextualChunkerConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    context_prompt_template: str = Field(
        default=(
            "Here is the document:\n<document>\n{document}\n</document>\n\n"
            "Here is a chunk from the document:\n<chunk>\n{chunk}\n</chunk>\n\n"
            "Give a short, succinct context (1-2 sentences) to situate this "
            "chunk within the overall document, to improve search retrieval "
            "of the chunk. Answer only with the context, nothing else."
        )
    )
    max_document_chars: int = 8000
