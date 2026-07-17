import pytest

pytest.importorskip("whisperx")

from agent_platform.core.schemas.enums import Language


def test_parse_language_valid():
    from agent_platform.integrations.speech_to_text.utils import parse_language

    assert parse_language("en") == Language.EN
    assert parse_language("fr") == Language.FR
    assert parse_language("de") == Language.GE


def test_parse_language_invalid_returns_none():
    from agent_platform.integrations.speech_to_text.utils import parse_language

    assert parse_language("zz") is None
    assert parse_language("") is None
    assert parse_language("invalid") is None
