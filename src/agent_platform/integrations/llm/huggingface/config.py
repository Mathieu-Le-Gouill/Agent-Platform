from __future__ import annotations

from agent_platform.core.interfaces.llm.config import GenerationConfig


class HuggingFaceGenerationConfig(GenerationConfig):
    # `model` is a HuggingFace Hub repository id such as "deepseek-ai/DeepSeek-R1-0528".

    provider: str = "auto"
    """Inference provider for the `model` repository (e.g. "cerebras"); "auto" picks the
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
