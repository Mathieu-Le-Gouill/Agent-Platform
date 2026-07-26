import pytest
from pydantic import ValidationError

from agent_platform.core.config import ModelConfig, ProviderConfig, parse_model_string
from agent_platform.core.errors import ConfigError


def test_provider_config_defaults():
    cfg = ProviderConfig()
    assert cfg.extra_params == {}


def test_provider_config_is_frozen():
    cfg = ProviderConfig()
    try:
        cfg.extra_params = {"a": 1}
        assert False, "expected ValidationError"
    except ValidationError:
        pass


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.model == ""
    assert cfg.extra_params == {}


def test_model_config_construction():
    cfg = ModelConfig(model="gpt-4")
    assert cfg.model == "gpt-4"


def test_model_config_is_provider_config():
    assert issubclass(ModelConfig, ProviderConfig)


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
