from uuid import uuid4

import pytest

from agent_platform.core.interfaces.translation.base import BaseTranslator


def test_base_translator_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseTranslator()  # type: ignore[abstract]


def test_subclass_must_implement_translate():
    class MissingTranslate(BaseTranslator):
        pass

    with pytest.raises(TypeError):
        MissingTranslate()  # type: ignore[abstract]


def test_concrete_subclass_instantiates():
    class Concrete(BaseTranslator):
        async def translate(self, content, target, source=None):
            return content

    instance = Concrete()
    assert isinstance(instance, BaseTranslator)
