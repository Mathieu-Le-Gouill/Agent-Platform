import pytest

pytest.importorskip("transformers")

import agent_platform.integrations.classification as classification_module


class TestClassificationInitExports:
    def test_transformers_classifier_is_lazily_importable(self):
        cls = classification_module.TransformersClassifier
        assert cls.__name__ == "TransformersClassifier"

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            classification_module.NotAProvider

    def test_dir_includes_provider(self):
        assert "TransformersClassifier" in dir(classification_module)
