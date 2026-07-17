import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("diffusers")

from agent_platform.integrations.image_generation.stable_diffusion.config import (
    StableDiffusionConfig,
)
from agent_platform.core.errors import ProviderError
from agent_platform.core.schemas.enums import ImageFormat


@patch(
    "agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion.StableDiffusionPipeline"
)
async def test_load_early_return(mock_pipe_cls):
    from agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion import (
        StableDiffusionGenerator,
    )

    gen = StableDiffusionGenerator(StableDiffusionConfig())
    gen._pipeline = MagicMock()
    await gen._load()
    mock_pipe_cls.from_pretrained.assert_not_called()


@patch(
    "agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion.StableDiffusionPipeline"
)
async def test_load_safety_checker_none(mock_pipe_cls):
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    mock_pipe_cls.from_pretrained.return_value = mock_pipe

    from agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion import (
        StableDiffusionGenerator,
    )

    gen = StableDiffusionGenerator(config=StableDiffusionConfig(safety_checker=False))
    gen._pipeline = None
    loop = asyncio.get_event_loop()

    async def _mock_run(executor, fn):
        return fn()

    with patch.object(loop, "run_in_executor", side_effect=_mock_run):
        await gen._load()

    mock_pipe_cls.from_pretrained.assert_called_once()
    call_kwargs = mock_pipe_cls.from_pretrained.call_args[1]
    assert call_kwargs["safety_checker"] is None


@patch(
    "agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion.StableDiffusionPipeline"
)
async def test_generate_pipeline_none(mock_pipe_cls):
    from agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion import (
        StableDiffusionGenerator,
    )

    gen = StableDiffusionGenerator(StableDiffusionConfig())
    gen._pipeline = None

    with patch.object(gen, "_load", AsyncMock()):
        with pytest.raises(
            ProviderError, match="Failed to load Stable Diffusion pipeline"
        ):
            await gen.generate("test prompt")


@patch(
    "agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion.StableDiffusionPipeline"
)
async def test_generate_many_pipeline_none(mock_pipe_cls):
    from agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion import (
        StableDiffusionGenerator,
    )

    gen = StableDiffusionGenerator(StableDiffusionConfig())
    gen._pipeline = None

    with patch.object(gen, "_load", AsyncMock()):
        with pytest.raises(
            ProviderError, match="Failed to load Stable Diffusion pipeline"
        ):
            await gen.generate_many("test prompt", n=2)


@patch(
    "agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion.StableDiffusionPipeline"
)
async def test_generate_success(mock_pipe_cls):
    mock_pipe = MagicMock()
    mock_pipe.to.return_value = mock_pipe
    mock_pipe_cls.from_pretrained.return_value = mock_pipe

    mock_img = MagicMock()
    mock_img.width = 512
    mock_img.height = 512

    mock_output = MagicMock()
    mock_output.images = [mock_img]
    mock_pipe.return_value = mock_output

    from agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion import (
        StableDiffusionGenerator,
    )

    gen = StableDiffusionGenerator(StableDiffusionConfig())
    gen._pipeline = mock_pipe

    with patch("torch.no_grad"):
        result = await gen.generate("test", size="256x256")

    assert result.dimensions.width == 512
    assert result.dimensions.height == 512
    assert result.format == ImageFormat.PNG


def test_pluck_images_tuple():
    from agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion import (
        _pluck_images,
    )

    mock_out = MagicMock()
    mock_img = MagicMock()
    result = _pluck_images(([mock_img],))
    assert list(result) == [mock_img]
