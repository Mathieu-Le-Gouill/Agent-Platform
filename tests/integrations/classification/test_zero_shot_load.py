import asyncio
from unittest.mock import MagicMock, patch

from agent_platform.integrations.classification.providers.zero_shot import (
    ZeroShotClassifier,
)
from agent_platform.models.chunk import TextChunk


async def test_load_runs_pipeline():
    classifier = ZeroShotClassifier()

    async def _mock_run(executor, fn):
        return fn()

    mock_pipe = MagicMock()
    loop = asyncio.get_event_loop()
    with (
        patch.object(loop, "run_in_executor", side_effect=_mock_run),
        patch(
            "agent_platform.integrations.classification.providers.zero_shot.pipeline",
            return_value=mock_pipe,
        ),
    ):
        await classifier._load()

    assert classifier._pipeline is mock_pipe
