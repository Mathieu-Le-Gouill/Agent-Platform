from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("google.cloud")

from agent_platform.integrations.ocr.google_vision.google_vision import (
    _from_google_vision,
)
from agent_platform.integrations.ocr.google_vision.config import GoogleVisionConfig


def _symbol(text):
    return SimpleNamespace(text=text, confidence=0.95)


def _word(text, confidence):
    return SimpleNamespace(
        symbols=[SimpleNamespace(text=ch, confidence=confidence) for ch in text],
        confidence=confidence,
    )


def _fake_response(words_with_confidence):
    paragraph = SimpleNamespace(
        words=[_word(text, conf) for text, conf in words_with_confidence]
    )
    block = SimpleNamespace(paragraphs=[paragraph])
    page = SimpleNamespace(blocks=[block], page_number=1)
    return SimpleNamespace(
        error=SimpleNamespace(message=""),
        full_text_annotation=SimpleNamespace(pages=[page]),
    )


class TestFromGoogleVision:
    def test_joins_words_into_text(self):
        document_id = uuid4()
        response = _fake_response([("Hello", 0.9), ("world", 0.8)])
        chunks = _from_google_vision(
            response, document_id=document_id, min_confidence=0.0
        )
        assert [c.text for c in chunks] == ["Hello world"]
        assert all(c.document_id == document_id for c in chunks)

    def test_filters_by_min_confidence(self):
        document_id = uuid4()
        # Two separate paragraphs: first passes (confidence 0.9), second is filtered
        p1 = SimpleNamespace(words=[_word("Hello", 0.9)])
        p2 = SimpleNamespace(words=[_word("world", 0.2)])
        block = SimpleNamespace(paragraphs=[p1, p2])
        page = SimpleNamespace(blocks=[block], page_number=1)
        response = SimpleNamespace(
            error=SimpleNamespace(message=""),
            full_text_annotation=SimpleNamespace(pages=[page]),
        )
        chunks = _from_google_vision(
            response, document_id=document_id, min_confidence=0.5
        )
        assert [c.text for c in chunks] == ["Hello"]

    def test_empty_words_skips_paragraph(self):
        document_id = uuid4()
        paragraph = SimpleNamespace(words=[])
        block = SimpleNamespace(paragraphs=[paragraph])
        page = SimpleNamespace(blocks=[block], page_number=1)
        response = SimpleNamespace(
            error=SimpleNamespace(message=""),
            full_text_annotation=SimpleNamespace(pages=[page]),
        )
        chunks = _from_google_vision(
            response, document_id=document_id, min_confidence=0.0
        )
        assert len(chunks) == 0

    def test_sets_page_number(self):
        document_id = uuid4()
        response = _fake_response([("Hi", 0.99)])
        chunks = _from_google_vision(
            response, document_id=document_id, min_confidence=0.0
        )
        assert chunks[0].metadata["page"] == 0
