import pytest

from agent_platform.core.interfaces.classification.base import (
    BaseClassificationProvider,
)
from agent_platform.core.interfaces.classification.config import ClassificationConfig
from agent_platform.core.interfaces.classification.response import (
    ClassificationResult,
)


class _ConcreteClassificationProvider(BaseClassificationProvider[ClassificationConfig]):
    async def classify(self, text, candidate_labels, config=None):
        return ClassificationResult(predictions=[])


class TestBaseClassificationProvider:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseClassificationProvider()

    def test_concrete_subclass_can_be_instantiated(self):
        provider = _ConcreteClassificationProvider()
        assert isinstance(provider, BaseClassificationProvider)

    async def test_classify_abstract_method_can_be_overridden(self):
        provider = _ConcreteClassificationProvider()
        result = await provider.classify("hello", ["a", "b"])
        assert isinstance(result, ClassificationResult)
