import pytest

from agent_platform.models.enums import Language


def test_parse_language_valid():
    from agent_platform.integrations.speech.providers.whisperx import _parse_language

    assert _parse_language("en") == Language.EN
    assert _parse_language("fr") == Language.FR
    assert _parse_language("de") == Language.GE


def test_parse_language_invalid_returns_none():
    from agent_platform.integrations.speech.providers.whisperx import _parse_language

    assert _parse_language("zz") is None
    assert _parse_language("") is None
    assert _parse_language("invalid") is None
