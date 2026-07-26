import pytest

from agent_platform.core.interfaces.translation.base import BaseTranslator


def test_base_translator_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseTranslator()  # type: ignore[abstract]


def test_subclass_must_implement_both_methods():
    class MissingBoth(BaseTranslator):
        pass

    with pytest.raises(TypeError):
        MissingBoth()  # type: ignore[abstract]


def test_subclass_missing_atranslate_cannot_instantiate():
    class MissingATranslate(BaseTranslator):
        def translate(self, content, target, source=None, config=None):
            return content

    with pytest.raises(TypeError):
        MissingATranslate()  # type: ignore[abstract]


def test_concrete_subclass_instantiates():
    class Concrete(BaseTranslator):
        def translate(self, content, target, source=None, config=None):
            return content

        async def atranslate(self, content, target, source=None, config=None):
            return content

    instance = Concrete()
    assert isinstance(instance, BaseTranslator)
