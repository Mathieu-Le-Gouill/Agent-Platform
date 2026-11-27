import pytest
from pydantic import ValidationError

from agent_platform.core.credentials import ClientOptions


def test_client_options_max_retries_default():
    assert ClientOptions().max_retries == 3


def test_client_options_accepts_zero_max_retries():
    assert ClientOptions(max_retries=0).max_retries == 0


def test_client_options_rejects_negative_max_retries():
    with pytest.raises(ValidationError):
        ClientOptions(max_retries=-1)
