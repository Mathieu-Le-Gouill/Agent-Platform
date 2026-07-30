from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pydantic import SecretStr

from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.moderation.openai.config import OpenAIModerationConfig
from agent_platform.integrations.moderation.openai.provider import OpenAIModeration


def _creds(key: str = "sk-test") -> OpenAICredentials:
    return OpenAICredentials(api_key=SecretStr(key))


class _DictLike:
    def __init__(self, data: dict) -> None:
        self._data = data

    def model_dump(self) -> dict:
        return self._data


def _response(flagged: bool, categories: dict, scores: dict) -> SimpleNamespace:
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                flagged=flagged,
                categories=_DictLike(categories),
                category_scores=_DictLike(scores),
            )
        ]
    )


class TestOpenAIModerationClient:
    def test_client_uses_credentials_api_key(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.moderation.openai.provider.OpenAI"
        )
        provider = OpenAIModeration(_creds("topsecret"))
        provider._sync_client(OpenAIModerationConfig())
        _, kwargs = mock_openai.call_args
        assert kwargs["api_key"] == "topsecret"

    def test_client_kwargs_default_max_retries(self):
        provider = OpenAIModeration(_creds())
        kwargs = provider._client_kwargs(OpenAIModerationConfig(max_retries=None))
        assert kwargs["max_retries"] == 3
        assert "timeout" not in kwargs

    def test_client_kwargs_explicit_timeout_and_retries(self):
        provider = OpenAIModeration(_creds())
        cfg = OpenAIModerationConfig(timeout=15.0, max_retries=5)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["timeout"] == 15.0
        assert kwargs["max_retries"] == 5

    def test_client_kwargs_client_options_fallback(self):
        from agent_platform.core.credentials import ClientOptions

        provider = OpenAIModeration(
            _creds(), client_options=ClientOptions(timeout=60.0, max_retries=7)
        )
        kwargs = provider._client_kwargs(OpenAIModerationConfig())
        assert kwargs["timeout"] == 60.0
        assert kwargs["max_retries"] == 7


class TestOpenAIModerationSync:
    def test_moderate_maps_categories(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.moderation.openai.provider.OpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.moderations.create.return_value = _response(
            flagged=True,
            categories={"violence": True, "hate": False},
            scores={"violence": 0.9, "hate": 0.1},
        )

        provider = OpenAIModeration(_creds())
        result = provider.moderate("some text")

        assert result.flagged is True
        violence = next(c for c in result.categories if c.name == "violence")
        assert violence.flagged is True
        assert violence.score.value == 0.9

        _, kwargs = mock_client.moderations.create.call_args
        assert kwargs["model"] == "omni-moderation-latest"
        assert kwargs["input"] == "some text"


class TestOpenAIModerationAsync:
    async def test_amoderate_maps_categories(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.moderation.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.moderations.create = AsyncMock(
            return_value=_response(
                flagged=False,
                categories={"violence": False},
                scores={"violence": 0.01},
            )
        )

        provider = OpenAIModeration(_creds())
        result = await provider.amoderate("clean text")

        assert result.flagged is False
        assert result.categories[0].flagged is False

    async def test_amoderate_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_openai = mocker.patch(
            "agent_platform.integrations.moderation.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return _response(flagged=False, categories={}, scores={})

        mock_client.moderations.create = flaky

        provider = OpenAIModeration(_creds())
        result = await provider.amoderate("hi")

        assert calls["n"] == 2
        assert result.flagged is False

    async def test_amoderate_translates_permanent_failure(self, mocker, no_retry_sleep):
        import pytest

        from agent_platform.core.errors import ProviderError

        mock_openai = mocker.patch(
            "agent_platform.integrations.moderation.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.moderations.create = always_fails

        provider = OpenAIModeration(_creds())
        with pytest.raises(ProviderError, match="Moderation request failed"):
            await provider.amoderate("hi")
