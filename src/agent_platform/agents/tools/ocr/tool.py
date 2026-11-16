from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.errors import ToolSourceNotAllowedError
from agent_platform.agents.tools.safe_execution import safe_call
from agent_platform.components.ingestion.ocr.component import OCR
from agent_platform.core.interfaces.ocr.config import OCRConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import ImageDocument
from agent_platform.core.schemas.message import ContentBlock, ImageBlock, TextBlock
from agent_platform.utils.score import filter_by_score

_URL_SCHEMES = ("http", "https")
_DENIED_MESSAGE = "OCR source is not allowed"


class OCRInput(BaseModel):
    source: str = Field(..., description="Image file path or URL")
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold [0-1]",
    )


@dataclass(frozen=True)
class OCRSourcePolicy:
    """Which sources the model may ask `OCRTool` to read.

    `allowed_dir` is a folder whose files (subfolders included) may be read;
    `allowed_hosts` are the host names a URL may point to, compared in lower
    case. The defaults allow nothing.
    """

    allowed_dir: Path | None = None
    allowed_hosts: frozenset[str] = frozenset()

    def resolve(self, source: str) -> str:
        """Return the source to read, or raise `ToolSourceNotAllowedError`."""
        parsed = urlparse(source)
        if parsed.scheme.lower() in _URL_SCHEMES:
            allowed = {host.lower() for host in self.allowed_hosts}
            if (parsed.hostname or "").lower() in allowed:
                # The backends match "http://" case-sensitively; any other
                # casing would reach them as a local path.
                return parsed._replace(scheme=parsed.scheme.lower()).geturl()
            raise ToolSourceNotAllowedError(_DENIED_MESSAGE)

        if self.allowed_dir is not None:
            path = Path(source).resolve()
            if path.is_relative_to(self.allowed_dir.resolve()):
                return str(path)
        raise ToolSourceNotAllowedError(_DENIED_MESSAGE)


class OCRTool(Tool):
    name = "ocr"
    description = "Extract text from an image file or URL using OCR."
    input_schema = OCRInput
    output_schema = None

    def __init__(self, ocr: OCR, policy: OCRSourcePolicy | None = None) -> None:
        """`policy` limits the sources `run()` reads; `None` allows none."""
        self._ocr = ocr
        self._policy = policy or OCRSourcePolicy()

    async def run(self, **kwargs: Any) -> list[TextChunk]:
        validated = OCRInput(**kwargs)
        source = self._policy.resolve(validated.source)
        config = OCRConfig(min_confidence=validated.min_confidence)
        results = await safe_call(
            self._ocr.arun((source, config)),
            "OCR extraction failed",
        )
        return filter_by_score(
            results,
            validated.min_confidence,
            key=lambda c: c.confidence.normalized if c.confidence else 1.0,
        )

    def to_blocks(self, source: str, chunks: Sequence[TextChunk]) -> list[ContentBlock]:
        blocks: list[ContentBlock] = [self._image_block(source)]
        text = "\n".join(chunk.text for chunk in chunks if chunk.text)
        if text:
            blocks.append(TextBlock(text=text))
        return blocks

    def _image_block(self, source: str) -> ImageBlock:
        if source.startswith(("http://", "https://")):
            return ImageBlock(image=source)
        return ImageBlock(image=ImageDocument.load_content(source))
