import base64
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agent_platform.integrations.image_generation.providers.dalle import (
    DallEImageGenerator,
    _validate_size,
)
from agent_platform.integrations.image_generation.providers.midjourney import (
    MidjourneyGenerator,
    _size_to_aspect,
)
from agent_platform.models.document import ImageDocument
from agent_platform.models.enums import ImageFormat

try:
    from agent_platform.integrations.image_generation.providers.stable_diffusion import (
        StableDiffusionGenerator,
        _parse_size,
    )

    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False


class TestDallEImageGenerator:
    def test_constructor_valid_model(self):
        gen = DallEImageGenerator(api_key="test-key", model="dall-e-3")
        assert gen._model == "dall-e-3"
        assert gen._quality == "standard"

    def test_constructor_valid_model_dalle2(self):
        gen = DallEImageGenerator(api_key="test-key", model="dall-e-2", quality="hd")
        assert gen._model == "dall-e-2"
        assert gen._quality == "hd"

    def test_constructor_invalid_model(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            DallEImageGenerator(api_key="test-key", model="dall-e-4")

    def test_constructor_invalid_model_empty(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            DallEImageGenerator(api_key="test-key", model="")

    def test_constructor_default_model(self):
        gen = DallEImageGenerator(api_key="sk-test")
        assert gen._model == "dall-e-3"


class TestValidateSize:
    def test_valid_dalle2_sizes(self):
        for size in ("256x256", "512x512", "1024x1024"):
            _validate_size("dall-e-2", size)

    def test_valid_dalle3_sizes(self):
        for size in ("1024x1024", "1792x1024", "1024x1792"):
            _validate_size("dall-e-3", size)

    def test_invalid_dalle3_size(self):
        with pytest.raises(ValueError, match="Invalid size.*256x256.*dall-e-3"):
            _validate_size("dall-e-3", "256x256")

    def test_invalid_dalle2_size(self):
        with pytest.raises(ValueError, match="Invalid size.*1792x1024.*dall-e-2"):
            _validate_size("dall-e-2", "1792x1024")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            _validate_size("dall-e-2", "invalid")

    def test_empty_size_raises(self):
        with pytest.raises(ValueError):
            _validate_size("dall-e-3", "")


@pytest.mark.skipif(not _HAS_DIFFUSERS, reason="requires diffusers")
class TestStableDiffusionGenerator:
    def test_constructor_defaults(self):
        gen = StableDiffusionGenerator()
        assert gen._model_id == "runwayml/stable-diffusion-v1-5"
        assert gen._device == "cpu"
        assert gen._safety_checker is True

    def test_constructor_custom(self):
        gen = StableDiffusionGenerator(
            model_id="custom/model",
            device="cuda",
            safety_checker=False,
        )
        assert gen._model_id == "custom/model"
        assert gen._device == "cuda"
        assert gen._safety_checker is False


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

    def test_invalid_format_returns_default(self):
        assert _parse_size("invalid") == (512, 512)

    def test_empty_string_returns_default(self):
        assert _parse_size("") == (512, 512)

    def test_extra_parts_returns_default(self):
        assert _parse_size("100x200x300") == (512, 512)

    def test_non_numeric_returns_default(self):
        assert _parse_size("abcxdef") == (512, 512)

    def test_case_insensitive(self):
        assert _parse_size("1024X768") == (1024, 768)


class TestMidjourneyGenerator:
    def test_constructor_defaults(self):
        gen = MidjourneyGenerator(api_url="https://api.example.com", api_key="test-key")
        assert gen._api_url == "https://api.example.com"
        assert gen._timeout == 120.0

    def test_constructor_trailing_slash_stripped(self):
        gen = MidjourneyGenerator(api_url="https://api.example.com/", api_key="k")
        assert gen._api_url == "https://api.example.com"

    def test_constructor_custom_timeout(self):
        gen = MidjourneyGenerator(
            api_url="https://api.example.com", api_key="k", timeout=60.0
        )
        assert gen._timeout == 60.0


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
    @patch("agent_platform.integrations.image_generation.providers.dalle.AsyncOpenAI")
    async def test_generate_returns_image_document(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        b64 = base64.b64encode(b"fake_image_bytes").decode()
        mock_data = MagicMock()
        mock_data.b64_json = b64
        mock_data.revised_prompt = "revised version"
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(api_key="test-key")
        result = await gen.generate("test prompt", format=ImageFormat.PNG)

        assert isinstance(result, ImageDocument)
        assert result.content == b"fake_image_bytes"
        assert result.format == ImageFormat.PNG
        assert result.metadata.description == "test prompt"
        assert result.metadata.extra["provider"] == "openai"
        assert result.metadata.extra["model"] == "dall-e-3"
        assert result.metadata.extra["revised_prompt"] == "revised version"

    @patch("agent_platform.integrations.image_generation.providers.dalle.AsyncOpenAI")
    async def test_generate_with_custom_size(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        b64 = base64.b64encode(b"img_data").decode()
        mock_data = MagicMock()
        mock_data.b64_json = b64
        mock_data.revised_prompt = None
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(api_key="test-key", model="dall-e-2")
        result = await gen.generate("test", size="512x512")

        assert result.metadata.extra["size"] == "512x512"

    @patch("agent_platform.integrations.image_generation.providers.dalle.AsyncOpenAI")
    async def test_generate_many_returns_multiple_documents(self, mock_openai_cls):
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

        gen = DallEImageGenerator(api_key="test-key")
        results = await gen.generate_many("test", n=2)

        assert len(results) == 2
        assert results[0].content == b"img1"
        assert results[1].content == b"img2"
        assert results[0].metadata.extra["revised_prompt"] == "rev1"
        assert results[1].metadata.extra["revised_prompt"] == "rev2"


class TestDallEGenerateErrorHandling:
    @patch("agent_platform.integrations.image_generation.providers.dalle.AsyncOpenAI")
    async def test_response_data_none_raises_error(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = None
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(api_key="test-key")
        with pytest.raises(RuntimeError, match="no image data"):
            await gen.generate("test")

    @patch("agent_platform.integrations.image_generation.providers.dalle.AsyncOpenAI")
    async def test_b64_json_none_raises_error_on_generate(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_data = MagicMock()
        mock_data.b64_json = None
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(api_key="test-key")
        with pytest.raises(RuntimeError, match="no b64_json"):
            await gen.generate("test")

    @patch("agent_platform.integrations.image_generation.providers.dalle.AsyncOpenAI")
    async def test_response_data_none_on_generate_many(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = None
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        gen = DallEImageGenerator(api_key="test-key")
        with pytest.raises(RuntimeError, match="no image data"):
            await gen.generate_many("test", n=2)

    @patch("agent_platform.integrations.image_generation.providers.dalle.AsyncOpenAI")
    async def test_b64_json_none_skipped_in_generate_many(self, mock_openai_cls):
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

        gen = DallEImageGenerator(api_key="test-key")
        results = await gen.generate_many("test", n=3)

        assert len(results) == 2
        assert results[0].content == b"img1"
        assert results[1].content == b"img3"


class TestMidjourneyGenerate:
    @patch("httpx.AsyncClient")
    async def test_generate_returns_image_document(self, mock_httpx_cls):
        mock_http_client = MagicMock()
        mock_httpx_cls.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "image_url": "https://cdn.example.com/img.png",
            "job_id": "job_123",
        }
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_response.content = b"image_bytes"

        gen = MidjourneyGenerator(api_url="https://api.example.com", api_key="test-key")
        result = await gen.generate("test prompt", size="1024x1024")

        assert isinstance(result, ImageDocument)
        assert result.content == b"image_bytes"
        assert result.metadata.description == "test prompt"
        assert result.metadata.extra["provider"] == "midjourney"
        assert result.metadata.extra["size"] == "1024x1024"
        assert result.metadata.extra["job_id"] == "job_123"

    @patch("httpx.AsyncClient")
    async def test_generate_without_size(self, mock_httpx_cls):
        mock_http_client = MagicMock()
        mock_httpx_cls.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "image_url": "https://cdn.example.com/img.png"
        }
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_response.content = b"img_data"

        gen = MidjourneyGenerator(api_url="https://api.example.com", api_key="test-key")
        result = await gen.generate("test prompt")

        assert result.metadata.extra["size"] is None

    @patch("httpx.AsyncClient")
    async def test_generate_many(self, mock_httpx_cls):
        mock_http_client = MagicMock()
        mock_httpx_cls.return_value = mock_http_client

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

        gen = MidjourneyGenerator(api_url="https://api.example.com", api_key="test-key")
        results = await gen.generate_many("test", n=2)

        assert len(results) == 2
        assert all(isinstance(r, ImageDocument) for r in results)


@pytest.mark.skipif(not _HAS_DIFFUSERS, reason="requires diffusers")
class TestStableDiffusionGenerate:
    @patch(
        "agent_platform.integrations.image_generation.providers.stable_diffusion.StableDiffusionPipeline.from_pretrained"
    )
    async def test_generate_returns_image_document(self, mock_from_pretrained):
        from PIL import Image

        img = Image.new("RGB", (64, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img]
        mock_pipeline.return_value = mock_output

        gen = StableDiffusionGenerator()
        result = await gen.generate("test prompt")

        assert isinstance(result, ImageDocument)
        assert isinstance(result.content, bytes)
        assert result.metadata.description == "test prompt"
        assert result.metadata.extra["provider"] == "stable_diffusion"

    @patch(
        "agent_platform.integrations.image_generation.providers.stable_diffusion.StableDiffusionPipeline.from_pretrained"
    )
    async def test_generate_with_custom_size(self, mock_from_pretrained):
        from PIL import Image

        img = Image.new("RGB", (128, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img]
        mock_pipeline.return_value = mock_output

        gen = StableDiffusionGenerator()
        result = await gen.generate("test", size="128x64")

        assert result.width == 128
        assert result.height == 64

    @patch(
        "agent_platform.integrations.image_generation.providers.stable_diffusion.StableDiffusionPipeline.from_pretrained"
    )
    async def test_generate_model_load_failure(self, mock_from_pretrained):
        mock_pipeline = MagicMock()
        mock_pipeline.to.side_effect = RuntimeError(
            "Failed to load Stable Diffusion pipeline"
        )
        mock_from_pretrained.return_value = mock_pipeline

        gen = StableDiffusionGenerator()
        with pytest.raises(RuntimeError, match="Failed to load Stable Diffusion"):
            await gen.generate("test")

    @patch(
        "agent_platform.integrations.image_generation.providers.stable_diffusion.StableDiffusionPipeline.from_pretrained"
    )
    async def test_generate_many(self, mock_from_pretrained):
        from PIL import Image

        img1 = Image.new("RGB", (64, 64))
        img2 = Image.new("RGB", (64, 64))

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_from_pretrained.return_value = mock_pipeline

        mock_output = MagicMock()
        mock_output.images = [img1, img2]
        mock_pipeline.return_value = mock_output

        gen = StableDiffusionGenerator()
        results = await gen.generate_many("test prompt", n=2)

        assert len(results) == 2
        assert all(isinstance(r, ImageDocument) for r in results)
