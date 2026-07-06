import pytest

from agent_platform.integrations.classification.base import ClassificationModel


def test_classification_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ClassificationModel()  # type: ignore[abstract]


def test_classify_is_abstract():
    assert "classify" in ClassificationModel.__abstractmethods__
