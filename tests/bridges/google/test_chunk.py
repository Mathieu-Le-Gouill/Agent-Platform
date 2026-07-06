from types import SimpleNamespace
from uuid import uuid4

from agent_platform.integrations.ocr.providers.google_vision import _from_google_vision


def test_from_google_vision_joins_words_into_paragraphs():
    document_id = uuid4()

    paragraph = SimpleNamespace(words=[_lc_word("Hello", 0.9), _lc_word("world", 0.8)])
    block = SimpleNamespace(paragraphs=[paragraph])
    page = SimpleNamespace(blocks=[block], page_number=1)
    response = SimpleNamespace(full_text_annotation=SimpleNamespace(pages=[page]))

    chunks = _from_google_vision(response, document_id=document_id, min_confidence=0.0)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"
    assert chunks[0].metadata["confidence"] == pytest.approx(0.85)
    assert chunks[0].metadata["page"] == 1


import pytest


def _lc_symbol(text):
    return SimpleNamespace(text=text, confidence=1.0)


def _lc_word(text, confidence):
    return SimpleNamespace(
        symbols=[SimpleNamespace(text=ch, confidence=confidence) for ch in text],
        confidence=confidence,
    )
