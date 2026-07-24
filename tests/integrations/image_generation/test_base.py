import pytest

from agent_platform.core.interfaces.image_generation.base import BaseImageGenerator


class TestBaseImageGeneratorABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseImageGenerator()

    def test_abstract_methods(self):
        expected = {"generate", "generate_many"}
        abstract = set(BaseImageGenerator.__abstractmethods__)
        assert abstract == expected

    def test_generate_signature(self, mocker):
        mocker.patch.object(BaseImageGenerator, "__abstractmethods__", set())
        instance = BaseImageGenerator()
        import inspect

        sig = inspect.signature(instance.generate)
        params = list(sig.parameters.keys())
        assert params == ["prompt", "config", "size", "format"]

    def test_generate_many_signature(self, mocker):
        mocker.patch.object(BaseImageGenerator, "__abstractmethods__", set())
        instance = BaseImageGenerator()
        import inspect

        sig = inspect.signature(instance.generate_many)
        params = list(sig.parameters.keys())
        assert params == ["prompt", "n", "config", "size", "format"]
