from unittest.mock import patch, MagicMock

import pytest

pytest.importorskip("langchain_community")

from agent_platform.integrations.loader.strategies.unstructured.config import (
    UnstructuredLoaderConfig,
)
from agent_platform.core.schemas.enums import FileFormat

_MODULE = "agent_platform.integrations.loader.strategies.unstructured.unstructured"


class TestUnstructuredFileLoader:
    def test_loader_selects_pdf_class(self):
        mock_cls = MagicMock(return_value="pdf_loader")
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.PDF: mock_cls}):
            with patch(f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.PDF):
                from agent_platform.integrations.loader.strategies.unstructured.unstructured import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                config = UnstructuredLoaderConfig()
                result = loader._loader("doc.pdf", config)
                assert result == "pdf_loader"

    def test_loader_selects_markdown_class(self):
        mock_cls = MagicMock(return_value="md_loader")
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.MARKDOWN: mock_cls}):
            with patch(
                f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.MARKDOWN
            ):
                from agent_platform.integrations.loader.strategies.unstructured.unstructured import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                config = UnstructuredLoaderConfig()
                result = loader._loader("doc.md", config)
                assert result == "md_loader"

    def test_loader_selects_html_class(self):
        mock_cls = MagicMock(return_value="html_loader")
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.HTML: mock_cls}):
            with patch(f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.HTML):
                from agent_platform.integrations.loader.strategies.unstructured.unstructured import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                config = UnstructuredLoaderConfig()
                result = loader._loader("doc.html", config)
                assert result == "html_loader"

    def test_loader_selects_docx_class(self):
        mock_cls = MagicMock(return_value="docx_loader")
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.DOCX: mock_cls}):
            with patch(f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.DOCX):
                from agent_platform.integrations.loader.strategies.unstructured.unstructured import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                config = UnstructuredLoaderConfig()
                result = loader._loader("doc.docx", config)
                assert result == "docx_loader"

    def test_loader_unknown_format_falls_back(self):
        mock_fallback = MagicMock(return_value="fallback_loader")
        with patch(f"{_MODULE}._LC_LOADER_MAP", {}):
            with patch(f"{_MODULE}.LC_FallbackLoader", mock_fallback):
                with patch(
                    f"{_MODULE}.FileFormat.from_path",
                    return_value=FileFormat.UNKNOWN,
                ):
                    from agent_platform.integrations.loader.strategies.unstructured.unstructured import (
                        UnstructuredFileLoader,
                    )

                    loader = UnstructuredFileLoader()
                    config = UnstructuredLoaderConfig()
                    result = loader._loader("doc.xyz", config)
                    assert result == "fallback_loader"

    def test_loader_sets_mode_elements_default(self):
        mock_cls = MagicMock()
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.PDF: mock_cls}):
            with patch(f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.PDF):
                from agent_platform.integrations.loader.strategies.unstructured.unstructured import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                config = UnstructuredLoaderConfig(mode="elements")
                loader._loader("doc.pdf", config)
                mock_cls.assert_called_once_with("doc.pdf", mode="elements")

    def test_loader_respects_custom_mode_kwarg(self):
        mock_cls = MagicMock()
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.PDF: mock_cls}):
            with patch(f"{_MODULE}.FileFormat.from_path", return_value=FileFormat.PDF):
                from agent_platform.integrations.loader.strategies.unstructured.unstructured import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                config = UnstructuredLoaderConfig(mode="single")
                loader._loader("doc.pdf", config)
                mock_cls.assert_called_once_with("doc.pdf", mode="single")

    def test_loader_map_contains_expected_keys(self):
        from agent_platform.integrations.loader.strategies.unstructured.unstructured import (
            _LC_LOADER_MAP,
        )

        assert FileFormat.PDF in _LC_LOADER_MAP
        assert FileFormat.MARKDOWN in _LC_LOADER_MAP
        assert FileFormat.HTML in _LC_LOADER_MAP
        assert FileFormat.DOCX in _LC_LOADER_MAP
        assert len(_LC_LOADER_MAP) == 4
