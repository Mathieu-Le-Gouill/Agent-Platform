from unittest.mock import patch

import pytest

from agent_platform.integrations.image_generation.base import BaseImageGenerator


class TestBaseImageGeneratorABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseImageGenerator()

    def test_abstract_methods(self):
        expected = {"generate", "generate_many"}
        abstract = set(BaseImageGenerator.__abstractmethods__)
        assert abstract == expected

    @patch.object(BaseImageGenerator, "__abstractmethods__", set())
    def test_generate_signature(self):
        instance = BaseImageGenerator()
        import inspect

        sig = inspect.signature(instance.generate)
        params = list(sig.parameters.keys())
        assert params == ["prompt", "size", "format"]

    @patch.object(BaseImageGenerator, "__abstractmethods__", set())
    def test_generate_many_signature(self):
        instance = BaseImageGenerator()
        import inspect

        sig = inspect.signature(instance.generate_many)
        params = list(sig.parameters.keys())
        assert params == ["prompt", "n", "size", "format"]
