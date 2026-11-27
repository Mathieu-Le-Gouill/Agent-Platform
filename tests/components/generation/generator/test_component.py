import pytest

from agent_platform.components.generation.generator.component import (
    Generator,
    GeneratorInput,
)
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.errors import LLMGenerationError
from tests.helpers import make_fake_llm_response


class TestGenerator:
    async def test_forwards_prompt_and_config_to_backend(self, mock_llm, fake_prompt):
        response = make_fake_llm_response(content="Hi there")
        mock_llm.agenerate.return_value = response
        config = GenerationConfig(temperature=0.2)

        result = await Generator(mock_llm).arun(GeneratorInput(fake_prompt, config))

        assert result is response
        mock_llm.agenerate.assert_awaited_once_with(fake_prompt, config)

    async def test_forwards_none_config(self, mock_llm, fake_prompt):
        mock_llm.agenerate.return_value = make_fake_llm_response()

        await Generator(mock_llm).arun(GeneratorInput(fake_prompt, None))

        mock_llm.agenerate.assert_awaited_once_with(fake_prompt, None)

    async def test_backend_error_propagates(self, mock_llm, fake_prompt):
        mock_llm.agenerate.side_effect = LLMGenerationError("boom")

        with pytest.raises(LLMGenerationError, match="boom"):
            await Generator(mock_llm).arun(GeneratorInput(fake_prompt, None))
