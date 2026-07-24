from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.huggingface.config import (
    HuggingFaceGenerationConfig,
)
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig


class ClassificationMode(StrEnum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_CLASSIFICATION = "text_classification"


class ClassificationExample(BaseModel):
    text: str
    label: str


class LLMClassifierConfig(BaseModel):
    classification_mode: ClassificationMode = ClassificationMode.TEXT_CLASSIFICATION
    system_prompt: str | None = None
    multi_label: bool = False
    threshold: float | None = None
    unknown_label: str = "unknown"
    return_confidence: bool = False
    examples: list[ClassificationExample] = []


class AnthropicLLMClassifierConfig(LLMClassifierConfig, AnthropicGenerationConfig):
    pass


class OpenAILLMClassifierConfig(LLMClassifierConfig, OpenAIGenerationConfig):
    pass


class MistralLLMClassifierConfig(LLMClassifierConfig, MistralGenerationConfig):
    pass


class OllamaLLMClassifierConfig(LLMClassifierConfig, OllamaGenerationConfig):
    model: str = "mikgr/doctype-classifier-vl"


class HuggingFaceLLMClassifierConfig(LLMClassifierConfig, HuggingFaceGenerationConfig):
    model: str = "distilbert-base-uncased-finetuned-sst-2-english"
