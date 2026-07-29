from pydantic import ValidationError

from agent_platform.core.config import ModelConfig, ProviderConfig


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
