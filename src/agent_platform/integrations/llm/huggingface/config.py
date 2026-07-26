from __future__ import annotations

from agent_platform.core.interfaces.llm.config import GenerationConfig


class HuggingFaceGenerationConfig(GenerationConfig):
    task: str = "text-generation"
    """Inference task type; unused since the provider moved to the OpenAI-compatible
    `InferenceClient.chat_completion` API, which has no task concept. Kept for
    config backward-compatibility."""

    repo_id: str = "deepseek-ai/DeepSeek-R1-0528"
    """HuggingFace Hub model repository id to run inference against."""

    provider: str = "auto"
    """Inference provider for the repo_id model (e.g. "cerebras"); "auto" picks the
    first available provider ordered by the huggingface.co/settings/inference-providers preference."""

    repetition_penalty: float | None = None
    """Penalty for repeated tokens; 1.0 means no penalty."""

    do_sample: bool | None = None
    """Activate logits sampling instead of greedy decoding."""

    typical_p: float | None = None
    """Typical decoding mass; see "Typical Decoding for Natural Language Generation"."""

    return_full_text: bool | None = None
    """Whether to prepend the prompt to the generated text."""


# sources: https://huggingface.co/docs/huggingface_hub/package_reference/inference_client
