from uuid import uuid4

import pytest

from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.core.interfaces.ocr.config import OCRConfig


class _ConcreteOCR(BaseOCR[OCRConfig]):
    def __init__(self, credentials=None):
        self._credentials = credentials

    async def extract(self, source, config=None, document_id=None):
        return []


class TestBaseOCR:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseOCR()

    def test_stores_credentials(self):
        marker = object()
        ocr = _ConcreteOCR(credentials=marker)
        assert ocr._credentials is marker

    async def test_concrete_subclass_can_define_extract(self):
        ocr = _ConcreteOCR()
        result = await ocr.extract("test_source")
        assert result == []

    async def test_extract_accepts_source_optional_config_and_document_id(self):
        ocr = _ConcreteOCR()
        doc_id = uuid4()
        result = await ocr.extract(source="path/to/doc.pdf", document_id=doc_id)
        assert result == []
