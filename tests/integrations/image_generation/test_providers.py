import base64
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

pytest.importorskip("openai")

from agent_platform.core.credentials import ClientOptions
from agent_platform.core.errors import ProviderError
from agent_platform.core.schemas.document import ImageDocument
from agent_platform.core.schemas.enums import ImageFormat
from agent_platform.integrations.credentials import (
    MidjourneyCredentials,
    OpenAICredentials,
)
from agent_platform.integrations.image_generation.dalle.config import DalleConfig
from agent_platform.integrations.image_generation.dalle.mappers import (
    validate_n as _validate_n,
)
from agent_platform.integrations.image_generation.dalle.mappers import (
    validate_size as _validate_size,
)
from agent_platform.integrations.image_generation.dalle.provider import (
    DallEImageGenerator,
)
from agent_platform.integrations.image_generation.midjourney.config import (
    MidjourneyConfig,
)
from agent_platform.integrations.image_generation.midjourney.mappers import (
    size_to_aspect as _size_to_aspect,
)
from agent_platform.integrations.image_generation.midjourney.provider import (
    MidjourneyGenerator,
)
from agent_platform.integrations.image_generation.stable_diffusion.config import (
    StableDiffusionConfig,
)

try:
    from agent_platform.integrations.image_generation.stable_diffusion.mappers import (
        parse_size as _parse_size,
    )
    from agent_platform.integrations.image_generation.stable_diffusion.provider import (
        StableDiffusionGenerator,
    )

    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False


class TestDallEImageGenerator:
    def test_constructor_valid_model(self):
        gen = DallEImageGenerator()
        assert gen._default_config().model == "dall-e-3"
        assert gen._default_config().quality is None

    def test_constructor_valid_model_dalle2(self):
        cfg = DalleConfig(model="dall-e-2", quality="hd")
        assert cfg.model == "dall-e-2"
        assert cfg.quality == "hd"

    def test_constructor_invalid_model(self):
        with pytest.raises(ProviderError, match="Unsupported model"):
            gen = DallEImageGenerator()
            import asyncio

            asyncio.run(gen.generate("prompt", config=DalleConfig(model="dall-e-4")))

    def test_constructor_invalid_model_empty(self):
        with pytest.raises((ValueError, Exception)):
            gen = DallEImageGenerator()
            import asyncio

            asyncio.run(gen.generate("prompt", config=DalleConfig(model="")))

    def test_constructor_default_model(self):
        gen = DallEImageGenerator()
        assert gen._default_config().model == "dall-e-3"


class TestValidateSize:
    @pytest.mark.parametrize(
        "model,size",
        [
            ("dall-e-2", "256x256"),
            ("dall-e-2", "512x512"),
            ("dall-e-2", "1024x1024"),
            ("dall-e-3", "1024x1024"),
            ("dall-e-3", "1792x1024"),
            ("dall-e-3", "1024x1792"),
        ],
    )
    def test_valid_sizes(self, model, size):
        _validate_size(model, size)

    @pytest.mark.parametrize(
        "model,size,match",
        [
            ("dall-e-3", "256x256", "Invalid size.*256x256.*dall-e-3"),
            ("dall-e-2", "1792x1024", "Invalid size.*1792x1024.*dall-e-2"),
        ],
    )
    def test_invalid_size_for_model(self, model, size, match):
        with pytest.raises(ValueError, match=match):
            _validate_size(model, size)

    @pytest.mark.parametrize(
        "model,size",
        [
            ("dall-e-2", "invalid"),
            ("dall-e-3", ""),
        ],
    )
    def test_malformed_size_raises(self, model, size):
        with pytest.raises(ValueError):
            _validate_size(model, size)


@pytest.mark.skipif(not _HAS_DIFFUSERS, reason="requires diffusers")
class TestStableDiffusionGenerator:
    def test_constructor_defaults(self):
        gen = StableDiffusionGenerator(StableDiffusionConfig())
        assert gen._config.model == "stable-diffusion-v1-5/stable-diffusion-v1-5"
        assert gen._config.device == "cpu"
        assert gen._config.safety_checker is True

    def test_constructor_custom(self):
        gen = StableDiffusionGenerator(
            config=StableDiffusionConfig(
                model="custom/model",
                device="cuda",
                safety_checker=False,
            )
        )
        assert gen._config.model == "custom/model"
        assert gen._config.device == "cuda"
        assert gen._config.safety_checker is False


@pytest.mark.skipif(not _HAS_DIFFUSERS, reason="requires diffusers")
class TestParseSize:
    def test_none_returns_default(self):
        assert _parse_size(None) == (512, 512)

    def test_valid_square(self):
        assert _parse_size("512x512") == (512, 512)

    def test_valid_landscape(self):
        assert _parse_size("1024x768") == (1024, 768)

    def test_valid_portrait(self):
        assert _parse_size("768x1024") == (768, 1024)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid size"):
            _parse_size("invalid")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid size"):
            _parse_size("")

    def test_extra_parts_raises(self):
        with pytest.raises(ValueError, match="Invalid size"):
            _parse_size("100x200x300")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="Invalid size"):
            _parse_size("abcxdef")

    def test_non_positive_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _parse_size("0x512")

    def test_case_insensitive(self):
        assert _parse_size("1024X768") == (1024, 768)


class TestMidjourneyGenerator:
    def test_constructor_defaults(self):
        gen = MidjourneyGenerator()
        assert gen._default_config().timeout is None
        assert gen._default_config().process_mode == "fast"
        assert gen._base_url() == "http://localhost:8080"

    def test_constructor_custom_timeout(self):
        cfg = MidjourneyConfig(timeout=60)
        assert cfg.timeout == 60

    def test_base_url_from_client_options(self):
        gen = MidjourneyGenerator(
            client_options=ClientOptions(base_url="https://mj.example.com")
        )
        assert gen._base_url() == "https://mj.example.com"

    def test_base_url_from_env_fallback(self, monkeypatch):
        monkeypatch.setenv("MIDJOURNEY_API_URL", "https://env.example.com")
        gen = MidjourneyGenerator()
        assert gen._base_url() == "https://env.example.com"

    def test_base_url_client_options_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MIDJOURNEY_API_URL", "https://env.example.com")
        gen = MidjourneyGenerator(
            client_options=ClientOptions(base_url="https://mj.example.com")
        )
        assert gen._base_url() == "https://mj.example.com"


class TestSizeToAspect:
    def test_none_returns_1_1(self):
        assert _size_to_aspect(None) == "1:1"

    def test_square(self):
        assert _size_to_aspect("1024x1024") == "1:1"

    def test_landscape_16_9(self):
        assert _size_to_aspect("1920x1080") == "16:9"

    def test_landscape_4_3(self):
        assert _size_to_aspect("1024x768") == "4:3"

    def test_portrait(self):
        assert _size_to_aspect("768x1024") == "3:4"

    def test_invalid_format_returns_1_1(self):
        assert _size_to_aspect("invalid") == "1:1"

    def test_empty_string_returns_1_1(self):
        assert _size_to_aspect("") == "1:1"

    def test_extra_parts_returns_1_1(self):
        assert _size_to_aspect("100x200x300") == "1:1"

    def test_non_numeric_returns_1_1(self):
        assert _size_to_aspect("abcxdef") == "1:1"

    def test_case_insensitive(self):
        assert _size_to_aspect("1024X768") == "4:3"


class TestDallEGenerate:
    async def test_generate_returns_image_document(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        b64 = base64.b64encode(b"fake_image_bytes").decode()
        mock_data = MagicMock()
        mock_data.b64_json = b64
        mock_data.revised_prompt = "revised version"
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        result = await gen.generate("test prompt", format=ImageFormat.PNG)

        assert isinstance(result, ImageDocument)
        assert result.content == b"fake_image_bytes"
        assert result.format == ImageFormat.PNG
        assert result.metadata.description == "test prompt"
        assert result.metadata.extra["provider"] == "openai"
        assert result.metadata.extra["model"] == "dall-e-3"
        assert result.metadata.extra["revised_prompt"] == "revised version"

    async def test_generate_with_custom_size(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        b64 = base64.b64encode(b"img_data").decode()
        mock_data = MagicMock()
        mock_data.b64_json = b64
        mock_data.revised_prompt = None
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        result = await gen.generate(
            "test", size="512x512", config=DalleConfig(model="dall-e-2")
        )

        assert result.metadata.extra["size"] == "512x512"

    async def test_generate_many_returns_multiple_documents(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        data1 = MagicMock()
        data1.b64_json = base64.b64encode(b"img1").decode()
        data1.revised_prompt = "rev1"
        data2 = MagicMock()
        data2.b64_json = base64.b64encode(b"img2").decode()
        data2.revised_prompt = "rev2"
        mock_response = MagicMock()
        mock_response.data = [data1, data2]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        results = await gen.generate_many(
            "test", n=2, config=DalleConfig(model="dall-e-2")
        )

        assert len(results) == 2
        assert results[0].content == b"img1"
        assert results[1].content == b"img2"
        assert results[0].metadata.extra["revised_prompt"] == "rev1"
        assert results[1].metadata.extra["revised_prompt"] == "rev2"

    async def test_generate_dalle3_forwards_style(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_data = MagicMock()
        mock_data.b64_json = base64.b64encode(b"img").decode()
        mock_data.revised_prompt = None
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        await gen.generate("test", config=DalleConfig(model="dall-e-3", style="vivid"))

        _, kwargs = mock_client.images.generate.call_args
        assert kwargs["style"] == "vivid"
        assert kwargs["response_format"] == "b64_json"
        assert "quality" not in kwargs

    async def test_generate_gpt_image_1_omits_response_format(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_data = MagicMock()
        mock_data.b64_json = base64.b64encode(b"img").decode()
        mock_data.revised_prompt = None
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        result = await gen.generate(
            "test", config=DalleConfig(model="gpt-image-1", quality="high")
        )

        _, kwargs = mock_client.images.generate.call_args
        assert "response_format" not in kwargs
        assert kwargs["quality"] == "high"
        assert kwargs["model"] == "gpt-image-1"
        assert "style" not in kwargs
        assert result.content == b"img"

    async def test_generate_gpt_image_1_default_omits_quality(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_data = MagicMock()
        mock_data.b64_json = base64.b64encode(b"img").decode()
        mock_data.revised_prompt = None
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        await gen.generate("test", config=DalleConfig(model="gpt-image-1"))

        _, kwargs = mock_client.images.generate.call_args
        assert "quality" not in kwargs

    async def test_generate_dalle2_omits_quality(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_data = MagicMock()
        mock_data.b64_json = base64.b64encode(b"img").decode()
        mock_data.revised_prompt = None
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        await gen.generate("test", config=DalleConfig(model="dall-e-2"))

        _, kwargs = mock_client.images.generate.call_args
        assert "quality" not in kwargs
        assert kwargs["response_format"] == "b64_json"


class TestDallEValidateN:
    def test_dalle3_only_allows_n_1(self):
        with pytest.raises(ValueError, match="Invalid n=2 for dall-e-3"):
            _validate_n("dall-e-3", 2)

    def test_dalle2_allows_1_to_10(self):
        for n in (1, 5, 10):
            _validate_n("dall-e-2", n)

    def test_dalle2_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="Invalid n=11 for dall-e-2"):
            _validate_n("dall-e-2", 11)

    def test_gpt_image_1_allows_1_to_10(self):
        for n in (1, 10):
            _validate_n("gpt-image-1", n)

    def test_gpt_image_1_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="Invalid n=0 for gpt-image-1"):
            _validate_n("gpt-image-1", 0)

    def test_generate_many_dalle3_n_greater_than_1_raises(self):
        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        with pytest.raises(ProviderError, match="Invalid n"):
            import asyncio

            asyncio.run(gen.generate_many("test", n=2, config=DalleConfig()))

    def test_generate_many_dalle2_n_out_of_range_raises(self):
        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        with pytest.raises(ProviderError, match="Invalid n"):
            import asyncio

            asyncio.run(
                gen.generate_many("test", n=11, config=DalleConfig(model="dall-e-2"))
            )


class TestDallEGenerateErrorHandling:
    async def test_response_data_none_raises_error(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = None
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        with pytest.raises(ProviderError, match="no image data"):
            await gen.generate("test")

    async def test_b64_json_none_raises_error_on_generate(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_data = MagicMock()
        mock_data.b64_json = None
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        with pytest.raises(ProviderError, match="no b64_json"):
            await gen.generate("test")

    async def test_response_data_none_on_generate_many(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = None
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        with pytest.raises(ProviderError, match="no image data"):
            await gen.generate_many("test", n=2, config=DalleConfig(model="dall-e-2"))

    async def test_b64_json_none_skipped_in_generate_many(self, mocker):
        mock_openai_cls = mocker.patch(
            "agent_platform.integrations.image_generation.dalle.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        data1 = MagicMock()
        data1.b64_json = base64.b64encode(b"img1").decode()
        data1.revised_prompt = "r1"
        data2 = MagicMock()
        data2.b64_json = None
        data2.revised_prompt = "r2"
        data3 = MagicMock()
        data3.b64_json = base64.b64encode(b"img3").decode()
        data3.revised_prompt = "r3"
        mock_response = MagicMock()
        mock_response.data = [data1, data2, data3]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(OpenAICredentials(api_key=SecretStr("test-key")))
        results = await gen.generate_many(
            "test", n=3, config=DalleConfig(model="dall-e-2")
        )

        assert len(results) == 2
        assert results[0].content == b"img1"
        assert results[1].content == b"img3"


def _mock_async_client(mock_httpx_cls: MagicMock) -> MagicMock:
    """Configure httpx.AsyncClient() to behave as an async context manager."""
    mock_http_client = MagicMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_httpx_cls.return_value = mock_http_client
    return mock_http_client


class TestMidjourneyGenerate:
    async def test_generate_returns_image_document(self, mocker):
        mock_httpx_cls = mocker.patch("httpx.AsyncClient")
        mock_http_client = _mock_async_client(mock_httpx_cls)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "image_url": "https://cdn.example.com/img.png",
            "job_id": "job_123",
        }
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_response.content = b"image_bytes"

        gen = MidjourneyGenerator(MidjourneyCredentials(api_key=SecretStr("test-key")))
        result = await gen.generate(
            "test prompt",
            size="1024x1024",
            config=MidjourneyConfig(),
        )

        assert isinstance(result, ImageDocument)
        assert result.content == b"image_bytes"
        assert result.metadata.description == "test prompt"
        assert result.metadata.extra["provider"] == "midjourney"
        assert result.metadata.extra["size"] == "1024x1024"
        assert result.metadata.extra["job_id"] == "job_123"

    async def test_generate_without_size(self, mocker):
        mock_httpx_cls = mocker.patch("httpx.AsyncClient")
        mock_http_client = _mock_async_client(mock_httpx_cls)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "image_url": "https://cdn.example.com/img.png"
        }
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_response.content = b"img_data"

        gen = MidjourneyGenerator(MidjourneyCredentials(api_key=SecretStr("test-key")))
        result = await gen.generate(
            "test prompt",
            config=MidjourneyConfig(),
        )

        assert result.metadata.extra["size"] is None

    async def test_generate_many(self, mocker):
        mock_httpx_cls = mocker.patch("httpx.AsyncClient")
        mock_http_client = _mock_async_client(mock_httpx_cls)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "image_urls": [
                "https://cdn.example.com/img1.png",
                "https://cdn.example.com/img2.png",
            ],
        }
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_response.content = b"img_data"

        gen = MidjourneyGenerator(MidjourneyCredentials(api_key=SecretStr("test-key")))
        results = await gen.generate_many(
            "test",
            n=2,
            config=MidjourneyConfig(),
        )

        assert len(results) == 2
        assert all(isinstance(r, ImageDocument) for r in results)

    async def test_generate_closes_client(self, mocker):
        mock_httpx_cls = mocker.patch("httpx.AsyncClient")
        mock_http_client = _mock_async_client(mock_httpx_cls)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"image_url": "https://cdn.example.com/i.png"}
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_response.content = b"bytes"

        gen = MidjourneyGenerator(MidjourneyCredentials(api_key=SecretStr("test-key")))
        await gen.generate("test", config=MidjourneyConfig())

        mock_http_client.__aenter__.assert_awaited_once()
        mock_http_client.__aexit__.assert_awaited_once()

    async def test_generate_many_closes_client_on_error(self, mocker):
        mock_httpx_cls = mocker.patch("httpx.AsyncClient")
        mock_http_client = _mock_async_client(mock_httpx_cls)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}
        mock_http_client.post = AsyncMock(return_value=mock_response)

        gen = MidjourneyGenerator(MidjourneyCredentials(api_key=SecretStr("test-key")))
        with pytest.raises(ProviderError):
            await gen.generate_many("test", n=2, config=MidjourneyConfig())

        # with_retry retries on failure; every attempt must still close its client.
        assert (
            mock_http_client.__aenter__.await_count
            == mock_http_client.__aexit__.await_count
        )
        assert mock_http_client.__aexit__.await_count >= 1

    async def test_generate_uses_configured_process_mode(self, mocker):
        mock_httpx_cls = mocker.patch("httpx.AsyncClient")
        mock_http_client = _mock_async_client(mock_httpx_cls)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"image_url": "https://cdn.example.com/i.png"}
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_response.content = b"bytes"

        gen = MidjourneyGenerator(MidjourneyCredentials(api_key=SecretStr("test-key")))
        await gen.generate(
            "test",
            config=MidjourneyConfig(process_mode="turbo"),
        )

        _, kwargs = mock_http_client.post.call_args
        assert kwargs["json"]["process_mode"] == "turbo"

    async def test_generate_missing_image_url_raises_provider_error(self, mocker):
        mock_httpx_cls = mocker.patch("httpx.AsyncClient")
        mock_http_client = _mock_async_client(mock_httpx_cls)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"job_id": "job_123"}
        mock_http_client.post = AsyncMock(return_value=mock_response)

        gen = MidjourneyGenerator(MidjourneyCredentials(api_key=SecretStr("test-key")))
        with pytest.raises(ProviderError, match="missing image_url"):
            await gen.generate("test", config=MidjourneyConfig())

    async def test_generate_many_missing_image_urls_raises_provider_error(self, mocker):
        mock_httpx_cls = mocker.patch("httpx.AsyncClient")
        mock_http_client = _mock_async_client(mock_httpx_cls)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}
        mock_http_client.post = AsyncMock(return_value=mock_response)

        gen = MidjourneyGenerator(MidjourneyCredentials(api_key=SecretStr("test-key")))
        with pytest.raises(ProviderError, match="missing image_url"):
            await gen.generate_many("test", n=2, config=MidjourneyConfig())


@pytest.mark.skipif(not _HAS_DIFFUSERS, reason="requires diffusers")
class TestStableDiffusionGenerate:
    async def test_generate_returns_image_document(self, mocker):
        mock_from_pretrained = mocker.patch(
            "agent_platform.integrations.image_generation.stable_diffusion.provider.StableDiffusionPipeline.from_pretrained"
        )
        from PIL import Image

        img = Image.new("RGB", (64, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img]
        mock_pipeline.return_value = mock_output

        gen = StableDiffusionGenerator(StableDiffusionConfig())
        result = await gen.generate("test prompt")

        assert isinstance(result, ImageDocument)
        assert isinstance(result.content, bytes)
        assert result.metadata.description == "test prompt"
        assert result.metadata.extra["provider"] == "stable_diffusion"

    async def test_generate_with_custom_size(self, mocker):
        mock_from_pretrained = mocker.patch(
            "agent_platform.integrations.image_generation.stable_diffusion.provider.StableDiffusionPipeline.from_pretrained"
        )
        from PIL import Image

        img = Image.new("RGB", (128, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img]
        mock_pipeline.return_value = mock_output

        gen = StableDiffusionGenerator(StableDiffusionConfig())
        result = await gen.generate("test", size="128x64")

        assert result.dimensions.width == 128
        assert result.dimensions.height == 64

    async def test_generate_model_load_failure(self, mocker):
        mock_from_pretrained = mocker.patch(
            "agent_platform.integrations.image_generation.stable_diffusion.provider.StableDiffusionPipeline.from_pretrained"
        )
        mock_pipeline = MagicMock()
        mock_pipeline.to.side_effect = RuntimeError(
            "Failed to load Stable Diffusion pipeline"
        )
        mock_from_pretrained.return_value = mock_pipeline

        gen = StableDiffusionGenerator(StableDiffusionConfig())
        with pytest.raises(RuntimeError, match="Failed to load Stable Diffusion"):
            await gen.generate("test")

    async def test_generate_many(self, mocker):
        mock_from_pretrained = mocker.patch(
            "agent_platform.integrations.image_generation.stable_diffusion.provider.StableDiffusionPipeline.from_pretrained"
        )
        from PIL import Image

        img1 = Image.new("RGB", (64, 64))
        img2 = Image.new("RGB", (64, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img1, img2]
        mock_pipeline.return_value = mock_output

        gen = StableDiffusionGenerator(StableDiffusionConfig())
        results = await gen.generate_many("test prompt", n=2)

        assert len(results) == 2
        assert all(isinstance(r, ImageDocument) for r in results)

    async def test_generate_wires_guidance_and_steps(self, mocker):
        mock_from_pretrained = mocker.patch(
            "agent_platform.integrations.image_generation.stable_diffusion.provider.StableDiffusionPipeline.from_pretrained"
        )
        from PIL import Image

        img = Image.new("RGB", (64, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img]
        mock_pipeline.return_value = mock_output

        gen = StableDiffusionGenerator(
            StableDiffusionConfig(guidance_scale=9.5, num_inference_steps=25)
        )
        await gen.generate("test prompt")

        _, call_kwargs = mock_pipeline.call_args
        assert call_kwargs["guidance_scale"] == 9.5
        assert call_kwargs["num_inference_steps"] == 25
        assert "negative_prompt" not in call_kwargs
        assert "generator" not in call_kwargs

    async def test_generate_wires_negative_prompt_from_config(self, mocker):
        mock_from_pretrained = mocker.patch(
            "agent_platform.integrations.image_generation.stable_diffusion.provider.StableDiffusionPipeline.from_pretrained"
        )
        from PIL import Image

        img = Image.new("RGB", (64, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img]
        mock_pipeline.return_value = mock_output

        gen = StableDiffusionGenerator(
            StableDiffusionConfig(negative_prompt="blurry, low quality")
        )
        await gen.generate("test prompt")

        _, call_kwargs = mock_pipeline.call_args
        assert call_kwargs["negative_prompt"] == "blurry, low quality"

    async def test_generate_wires_negative_prompt_argument_override(self, mocker):
        mock_from_pretrained = mocker.patch(
            "agent_platform.integrations.image_generation.stable_diffusion.provider.StableDiffusionPipeline.from_pretrained"
        )
        from PIL import Image

        img = Image.new("RGB", (64, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img]
        mock_pipeline.return_value = mock_output

        gen = StableDiffusionGenerator(
            StableDiffusionConfig(negative_prompt="config value")
        )
        await gen.generate("test prompt", negative_prompt="argument value")

        _, call_kwargs = mock_pipeline.call_args
        assert call_kwargs["negative_prompt"] == "argument value"

    async def test_generate_wires_seed_to_generator(self, mocker):
        mock_from_pretrained = mocker.patch(
            "agent_platform.integrations.image_generation.stable_diffusion.provider.StableDiffusionPipeline.from_pretrained"
        )
        mock_generator_cls = mocker.patch(
            "agent_platform.integrations.image_generation.stable_diffusion.provider.torch.Generator"
        )
        from PIL import Image

        img = Image.new("RGB", (64, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img]
        mock_pipeline.return_value = mock_output

        mock_generator = MagicMock()
        mock_generator_cls.return_value = mock_generator
        mock_generator.manual_seed.return_value = mock_generator

        gen = StableDiffusionGenerator(StableDiffusionConfig(seed=42, device="cpu"))
        await gen.generate("test prompt")

        mock_generator_cls.assert_called_once_with(device="cpu")
        mock_generator.manual_seed.assert_called_once_with(42)
        _, call_kwargs = mock_pipeline.call_args
        assert call_kwargs["generator"] is mock_generator

    async def test_generate_many_wires_guidance_and_steps(self, mocker):
        mock_from_pretrained = mocker.patch(
            "agent_platform.integrations.image_generation.stable_diffusion.provider.StableDiffusionPipeline.from_pretrained"
        )
        from PIL import Image

        img1 = Image.new("RGB", (64, 64))
        img2 = Image.new("RGB", (64, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img1, img2]
        mock_pipeline.return_value = mock_output

        gen = StableDiffusionGenerator(
            StableDiffusionConfig(guidance_scale=3.0, num_inference_steps=10)
        )
        await gen.generate_many("test prompt", n=2)

        _, call_kwargs = mock_pipeline.call_args
        assert call_kwargs["guidance_scale"] == 3.0
        assert call_kwargs["num_inference_steps"] == 10
        assert call_kwargs["num_images_per_prompt"] == 2
