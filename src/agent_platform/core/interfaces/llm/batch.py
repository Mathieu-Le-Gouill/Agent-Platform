from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.schemas.message import Prompt


class BatchStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class BatchRequest(BaseModel, frozen=True):
    custom_id: str
    prompt: Prompt
    config: GenerationConfig | None = None


class BatchJob(BaseModel, frozen=True):
    id: str
    status: BatchStatus
    request_count: int | None = None
    completed_count: int | None = None


class BatchResult(BaseModel, frozen=True):
    custom_id: str
    response: LLMResponse | None = None
    error: str | None = None


class BaseBatchLLMProvider(ABC):
    """Async batch inference: submit many prompts as one job, poll, then fetch results.

    A separate mixin ABC rather than part of `BaseLLMProvider`: only vendors
    with a first-class async batch API (OpenAI, Anthropic) can implement this
    without a `NotImplementedError` stub, so it isn't forced onto every LLM
    provider.
    """

    @abstractmethod
    async def submit_batch(self, requests: list[BatchRequest]) -> BatchJob: ...

    @abstractmethod
    async def get_batch_status(self, batch_id: str) -> BatchJob: ...

    @abstractmethod
    async def fetch_batch_results(self, batch_id: str) -> list[BatchResult]: ...
