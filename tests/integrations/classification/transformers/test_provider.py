import pytest

pytest.importorskip("transformers")

from agent_platform.integrations.classification.transformers.config import (
    TransformersClassificationConfig,
)
from agent_platform.integrations.classification.transformers.provider import (
    TransformersClassifier,
)

FAKE_RESULT = {
    "labels": ["sports", "politics"],
    "scores": [0.8, 0.2],
}


async def test_classify_returns_ranked_predictions(mocker):
    mock_pipeline_factory = mocker.patch(
        "agent_platform.integrations.classification.transformers.provider.pipeline"
    )
    mock_classifier = mocker.MagicMock(return_value=FAKE_RESULT)
    mock_pipeline_factory.return_value = mock_classifier

    provider = TransformersClassifier()
    result = await provider.classify("Real Madrid won", ["sports", "politics"])

    assert [p.label for p in result.predictions] == ["sports", "politics"]
    assert result.predictions[0].score.value == 0.8
    mock_pipeline_factory.assert_called_once_with(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device="cpu",
    )
    mock_classifier.assert_called_once_with(
        "Real Madrid won",
        ["sports", "politics"],
        multi_label=False,
        hypothesis_template="This example is {}.",
    )


async def test_classify_caches_pipeline_per_model(mocker):
    mock_pipeline_factory = mocker.patch(
        "agent_platform.integrations.classification.transformers.provider.pipeline"
    )
    mock_pipeline_factory.return_value = mocker.MagicMock(return_value=FAKE_RESULT)

    provider = TransformersClassifier()
    await provider.classify("a", ["sports", "politics"])
    await provider.classify("b", ["sports", "politics"])

    mock_pipeline_factory.assert_called_once()


async def test_classify_uses_provided_config(mocker):
    mock_pipeline_factory = mocker.patch(
        "agent_platform.integrations.classification.transformers.provider.pipeline"
    )
    mock_classifier = mocker.MagicMock(return_value=FAKE_RESULT)
    mock_pipeline_factory.return_value = mock_classifier

    provider = TransformersClassifier()
    config = TransformersClassificationConfig(
        model="valhalla/distilbart-mnli-12-3", device="cuda", multi_label=True
    )
    await provider.classify("a", ["sports", "politics"], config=config)

    mock_pipeline_factory.assert_called_once_with(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-3",
        device="cuda",
    )
    mock_classifier.assert_called_once_with(
        "a",
        ["sports", "politics"],
        multi_label=True,
        hypothesis_template="This example is {}.",
    )
