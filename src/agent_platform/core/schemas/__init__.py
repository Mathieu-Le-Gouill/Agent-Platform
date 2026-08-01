from __future__ import annotations

import functools
from typing import Any

from pydantic import BaseModel

from agent_platform.core.interfaces.classification.response import (
    ClassificationPrediction,
    ClassificationResponse,
    ClassificationResult,
)
from agent_platform.core.schemas.bounding_box import BoundingBox
from agent_platform.core.schemas.chunk import (
    AudioChunk,
    Chunk,
    TextChunk,
    VideoChunk,
)
from agent_platform.core.schemas.cluster import Cluster
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.core.schemas.dimensions import Dimensions
from agent_platform.core.schemas.document import (
    AudioDocument,
    Document,
    DocumentMetadata,
    ImageDocument,
    TextDocument,
    VideoDocument,
)
from agent_platform.core.schemas.embedding import Embedding
from agent_platform.core.schemas.enums import (
    AudioFormat,
    DataType,
    DocumentFormat,
    FileFormat,
    FinishReason,
    ImageFormat,
    Language,
    MediaFormat,
    MediaType,
    SimilarityMetric,
    VideoFormat,
)
from agent_platform.core.schemas.mcp import MCPToolSpec
from agent_platform.core.schemas.message import (
    AssistantMessage,
    AudioBlock,
    BaseMessage,
    ContentBlock,
    ContentMessage,
    ImageBlock,
    Message,
    MessageRole,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from agent_platform.core.schemas.score import Score, ScoreKind
from agent_platform.core.schemas.span import SampleSpan, TimeSpan
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.schemas.vector import SparseVector


@functools.lru_cache(maxsize=128)
def model_schema(model: type[BaseModel]) -> dict[str, Any]:
    # bare BaseModel is the "no parameters" sentinel for tools; special-cased to
    # avoid a stray "BaseModel"-titled schema being sent to the provider
    if model is BaseModel:
        return {"type": "object", "properties": {}}
    return model.model_json_schema()


__all__ = [
    "Document",
    "TextDocument",
    "ImageDocument",
    "AudioDocument",
    "VideoDocument",
    "DocumentMetadata",
    "Chunk",
    "TextChunk",
    "AudioChunk",
    "VideoChunk",
    "Embedding",
    "Score",
    "ScoreKind",
    "TokenUsage",
    "TimeSpan",
    "SampleSpan",
    "Cluster",
    "Dimensions",
    "BoundingBox",
    "MediaType",
    "DocumentFormat",
    "ImageFormat",
    "AudioFormat",
    "VideoFormat",
    "DataType",
    "Language",
    "FileFormat",
    "FinishReason",
    "MediaFormat",
    "SimilarityMetric",
    "MessageRole",
    "ToolCall",
    "ToolResult",
    "BaseMessage",
    "ContentMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "TextBlock",
    "ImageBlock",
    "AudioBlock",
    "ContentBlock",
    "Message",
    "Utterance",
    "Transcript",
    "ClassificationPrediction",
    "ClassificationResult",
    "ClassificationResponse",
    "SparseVector",
    "MCPToolSpec",
    "model_schema",
]
