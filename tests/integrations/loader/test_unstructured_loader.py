from unittest.mock import MagicMock

import pytest

pytest.importorskip("langchain_community")

from agent_platform.core.schemas.enums import FileFormat
from agent_platform.integrations.loader.strategies.unstructured.config import (
    UnstructuredLoaderConfig,
)

_MODULE = "agent_platform.integrations.loader.strategies.unstructured.provider"


class TestUnstructuredFileLoader:
    def test_loader_selects_pdf_class(self, mocker):
        mock_cls = MagicMock(return_value="pdf_loader")
        mocker.patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.PDF: mock_cls})
        mocker.patch(f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.PDF)
        from agent_platform.integrations.loader.strategies.unstructured.provider import (
            UnstructuredFileLoader,
        )

        loader = UnstructuredFileLoader()
        config = UnstructuredLoaderConfig()
        result = loader._loader("doc.pdf", config)
        assert result == "pdf_loader"

    def test_loader_selects_markdown_class(self, mocker):
        mock_cls = MagicMock(return_value="md_loader")
        mocker.patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.MARKDOWN: mock_cls})
        mocker.patch(
            f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.MARKDOWN
        )
        from agent_platform.integrations.loader.strategies.unstructured.provider import (
            UnstructuredFileLoader,
        )

        loader = UnstructuredFileLoader()
        config = UnstructuredLoaderConfig()
        result = loader._loader("doc.md", config)
        assert result == "md_loader"

    def test_loader_selects_html_class(self, mocker):
        mock_cls = MagicMock(return_value="html_loader")
        mocker.patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.HTML: mock_cls})
        mocker.patch(f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.HTML)
        from agent_platform.integrations.loader.strategies.unstructured.provider import (
            UnstructuredFileLoader,
        )

        loader = UnstructuredFileLoader()
        config = UnstructuredLoaderConfig()
        result = loader._loader("doc.html", config)
        assert result == "html_loader"

    def test_loader_selects_docx_class(self, mocker):
        mock_cls = MagicMock(return_value="docx_loader")
        mocker.patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.DOCX: mock_cls})
        mocker.patch(f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.DOCX)
        from agent_platform.integrations.loader.strategies.unstructured.provider import (
            UnstructuredFileLoader,
        )

        loader = UnstructuredFileLoader()
        config = UnstructuredLoaderConfig()
        result = loader._loader("doc.docx", config)
        assert result == "docx_loader"

    def test_loader_unknown_format_falls_back(self, mocker):
        mock_fallback = MagicMock(return_value="fallback_loader")
        mocker.patch(f"{_MODULE}._LC_LOADER_MAP", {})
        mocker.patch(f"{_MODULE}.LC_FallbackLoader", mock_fallback)
        mocker.patch(
            f"{_MODULE}.FileFormat.from_path",
            return_value=FileFormat.UNKNOWN,
        )
        from agent_platform.integrations.loader.strategies.unstructured.provider import (
            UnstructuredFileLoader,
        )

        loader = UnstructuredFileLoader()
        config = UnstructuredLoaderConfig()
        result = loader._loader("doc.xyz", config)
        assert result == "fallback_loader"

    def test_loader_sets_mode_elements_default(self, mocker):
        mock_cls = MagicMock()
        mocker.patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.PDF: mock_cls})
        mocker.patch(f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.PDF)
        from agent_platform.integrations.loader.strategies.unstructured.provider import (
            UnstructuredFileLoader,
        )

        loader = UnstructuredFileLoader()
        config = UnstructuredLoaderConfig(mode="elements")
        loader._loader("doc.pdf", config)
        mock_cls.assert_called_once_with("doc.pdf", mode="elements")

    def test_loader_respects_custom_mode_kwarg(self, mocker):
        mock_cls = MagicMock()
        mocker.patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.PDF: mock_cls})
        mocker.patch(f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.PDF)
        from agent_platform.integrations.loader.strategies.unstructured.provider import (
            UnstructuredFileLoader,
        )

        loader = UnstructuredFileLoader()
        config = UnstructuredLoaderConfig(mode="single")
        loader._loader("doc.pdf", config)
        mock_cls.assert_called_once_with("doc.pdf", mode="single")

    def test_loader_map_contains_expected_keys(self):
        from agent_platform.integrations.loader.strategies.unstructured.provider import (
            _LC_LOADER_MAP,
        )

        assert FileFormat.PDF in _LC_LOADER_MAP
        assert FileFormat.MARKDOWN in _LC_LOADER_MAP
        assert FileFormat.HTML in _LC_LOADER_MAP
        assert FileFormat.DOCX in _LC_LOADER_MAP
        assert len(_LC_LOADER_MAP) == 4
