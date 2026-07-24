import pytest

from agent_platform.core.interfaces.speech.base import BaseSpeechToText


def test_base_speech_to_text_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseSpeechToText()  # type: ignore[abstract]


def test_subclass_must_implement_transcribe():
    class MissingTranscribe(BaseSpeechToText):
        async def stream(self, frames):
            yield None
            return

    with pytest.raises(TypeError):
        MissingTranscribe()  # type: ignore[abstract]


def test_subclass_must_implement_stream():
    class MissingStream(BaseSpeechToText):
        async def transcribe(self, audio):
            return None

    with pytest.raises(TypeError):
        MissingStream()  # type: ignore[abstract]


def test_concrete_subclass_instantiates():
    class Concrete(BaseSpeechToText):
        async def transcribe(self, audio):
            return None

        async def stream(self, frames):
            yield None
            return

    instance = Concrete()
    assert isinstance(instance, BaseSpeechToText)
