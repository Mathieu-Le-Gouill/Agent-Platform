from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("mistralai")

from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.ocr.mistral.config import MistralOCRConfig
from agent_platform.integrations.ocr.mistral.mappers import (
    from_mistral as _from_mistral,
)
from agent_platform.integrations.ocr.mistral.provider import MistralOCR


def _page(index, markdown, confidence_scores=None):
    return SimpleNamespace(
        index=index, markdown=markdown, confidence_scores=confidence_scores
    )


def _scores(average, minimum=None, words=None):
    return SimpleNamespace(
        average_page_confidence_score=average,
        minimum_page_confidence_score=minimum if minimum is not None else average,
        word_confidence_scores=words,
    )


class TestFromMistral:
    def test_maps_pages_to_chunks(self):
        document_id = uuid4()
        response = SimpleNamespace(pages=[_page(0, "Hello world")])
        chunks = _from_mistral(response, document_id)
        assert [c.text for c in chunks] == ["Hello world"]
        assert chunks[0].document_id == document_id
        assert chunks[0].index == 0
        assert chunks[0].confidence is None

    def test_skips_blank_pages(self):
        document_id = uuid4()
        response = SimpleNamespace(pages=[_page(0, "   ")])
        chunks = _from_mistral(response, document_id)
        assert chunks == []

    def test_sets_confidence_score_when_present(self):
        document_id = uuid4()
        response = SimpleNamespace(
            pages=[_page(0, "Hi", confidence_scores=_scores(0.9))]
        )
        chunks = _from_mistral(response, document_id)
        assert chunks[0].confidence.value == 0.9

    def test_filters_pages_below_min_confidence(self):
        document_id = uuid4()
        response = SimpleNamespace(
            pages=[
                _page(0, "Good page", confidence_scores=_scores(0.95)),
                _page(1, "Bad page", confidence_scores=_scores(0.2)),
            ]
        )
        chunks = _from_mistral(response, document_id, min_confidence=0.5)
        assert [c.text for c in chunks] == ["Good page"]

    def test_no_confidence_scores_are_not_filtered(self):
        document_id = uuid4()
        response = SimpleNamespace(pages=[_page(0, "No scores here")])
        chunks = _from_mistral(response, document_id, min_confidence=0.9)
        assert [c.text for c in chunks] == ["No scores here"]


class TestMistralOCRExtract:
    @pytest.mark.asyncio
    async def test_extract_forwards_new_config_fields(self, mocker):
        mock_mistral_cls = mocker.patch(
            "agent_platform.integrations.ocr.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_client.ocr.process.return_value = SimpleNamespace(pages=[_page(0, "text")])
        mock_mistral_cls.return_value = mock_client

        creds = MistralCredentials(api_key="fake-key")
        ocr = MistralOCR(creds)
        config = MistralOCRConfig(
            confidence_scores_granularity="word",
            pages=[0, 1],
            table_format="markdown",
        )

        await ocr.extract("https://example.com/doc.pdf", config=config)

        call_kwargs = mock_client.ocr.process.call_args.kwargs
        assert call_kwargs["confidence_scores_granularity"] == "word"
        assert call_kwargs["pages"] == [0, 1]
        assert call_kwargs["table_format"] == "markdown"

    @pytest.mark.asyncio
    async def test_extract_omits_optional_fields_when_unset(self, mocker):
        mock_mistral_cls = mocker.patch(
            "agent_platform.integrations.ocr.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_client.ocr.process.return_value = SimpleNamespace(pages=[_page(0, "text")])
        mock_mistral_cls.return_value = mock_client

        creds = MistralCredentials(api_key="fake-key")
        ocr = MistralOCR(creds)

        await ocr.extract("https://example.com/doc.pdf")

        call_kwargs = mock_client.ocr.process.call_args.kwargs
        assert "confidence_scores_granularity" not in call_kwargs
        assert "pages" not in call_kwargs
        assert "table_format" not in call_kwargs

    @pytest.mark.asyncio
    async def test_client_is_cached_across_calls(self, mocker):
        mock_mistral_cls = mocker.patch(
            "agent_platform.integrations.ocr.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_client.ocr.process.return_value = SimpleNamespace(pages=[_page(0, "text")])
        mock_mistral_cls.return_value = mock_client

        creds = MistralCredentials(api_key="fake-key")
        ocr = MistralOCR(creds)

        await ocr.extract("https://example.com/doc1.pdf")
        await ocr.extract("https://example.com/doc2.pdf")

        mock_mistral_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_credentials_raises(self):
        creds = MistralCredentials(api_key=None)
        ocr = MistralOCR(creds)
        with pytest.raises(MissingCredentialError):
            await ocr.extract("https://example.com/doc.pdf")

    @pytest.mark.asyncio
    async def test_client_error_is_translated_to_provider_error(self, mocker):
        mock_mistral_cls = mocker.patch(
            "agent_platform.integrations.ocr.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_client.ocr.process.side_effect = RuntimeError("boom")
        mock_mistral_cls.return_value = mock_client

        creds = MistralCredentials(api_key="fake-key")
        ocr = MistralOCR(creds)

        with pytest.raises(ProviderError, match="boom"):
            await ocr.extract("https://example.com/doc.pdf")
