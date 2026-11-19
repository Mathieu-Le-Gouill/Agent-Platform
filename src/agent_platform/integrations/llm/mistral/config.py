from __future__ import annotations

from agent_platform.core.interfaces.llm.config import GenerationConfig


class MistralGenerationConfig(GenerationConfig):
    """Mistral config. `model` is a Mistral model id such as "mistral-medium-latest".

    The generic fields cover everything the adapter sends, so this class adds no
    field of its own.
    """


"""
sources: https://docs.mistral.ai/getting-started/models/models_overview/
         https://docs.mistral.ai/capabilities/structured_output/custom
"""
