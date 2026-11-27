import pytest
from pydantic import ValidationError

from agent_platform.core.config import ModelConfig, ProviderConfig, RequestOptions


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


def test_request_options_max_retries_defaults_to_none():
    assert RequestOptions().max_retries is None


@pytest.mark.parametrize("value", [0, 5])
def test_request_options_accepts_non_negative_max_retries(value):
    assert RequestOptions(max_retries=value).max_retries == value


def test_request_options_rejects_negative_max_retries():
    with pytest.raises(ValidationError):
        RequestOptions(max_retries=-1)
