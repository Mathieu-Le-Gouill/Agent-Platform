import pytest

from agent_platform.config.model_string import parse_model_string
from agent_platform.core.errors import ConfigError


def test_parse_model_string_splits_on_first_colon():
    assert parse_model_string("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")


def test_parse_model_string_keeps_huggingface_repo_id_intact():
    provider, model = parse_model_string("huggingface:meta-llama/Llama-3-8B")
    assert provider == "huggingface"
    assert model == "meta-llama/Llama-3-8B"


@pytest.mark.parametrize("value", ["gpt-4o-mini", ":gpt-4o-mini", "openai:", "openai"])
def test_parse_model_string_rejects_malformed_input(value):
    with pytest.raises(ConfigError):
        parse_model_string(value)
