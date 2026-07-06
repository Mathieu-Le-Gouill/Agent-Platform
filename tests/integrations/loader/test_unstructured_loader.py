from unittest.mock import patch, MagicMock

import pytest

from agent_platform.models.enums import FileFormat

_MODULE = "agent_platform.integrations.loader.file_loaders.unstructured_loader"


class TestUnstructuredFileLoader:
    def test_loader_selects_pdf_class(self):
        mock_cls = MagicMock(return_value="pdf_loader")
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.PDF: mock_cls}):
            with patch(
                f"{_MODULE}.FileFormat.from_extension", return_value=FileFormat.PDF
            ):
                from agent_platform.integrations.loader.file_loaders.unstructured_loader import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                result = loader._loader("doc.pdf")
                assert result == "pdf_loader"

    def test_loader_selects_markdown_class(self):
        mock_cls = MagicMock(return_value="md_loader")
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.MARKDOWN: mock_cls}):
            with patch(
                f"{_MODULE}.FileFormat.from_extension", return_value=FileFormat.MARKDOWN
            ):
                from agent_platform.integrations.loader.file_loaders.unstructured_loader import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                result = loader._loader("doc.md")
                assert result == "md_loader"

    def test_loader_selects_html_class(self):
        mock_cls = MagicMock(return_value="html_loader")
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.HTML: mock_cls}):
            with patch(
                f"{_MODULE}.FileFormat.from_extension", return_value=FileFormat.HTML
            ):
                from agent_platform.integrations.loader.file_loaders.unstructured_loader import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                result = loader._loader("doc.html")
                assert result == "html_loader"

    def test_loader_selects_docx_class(self):
        mock_cls = MagicMock(return_value="docx_loader")
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.DOCX: mock_cls}):
            with patch(
                f"{_MODULE}.FileFormat.from_extension", return_value=FileFormat.DOCX
            ):
                from agent_platform.integrations.loader.file_loaders.unstructured_loader import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                result = loader._loader("doc.docx")
                assert result == "docx_loader"

    def test_loader_unknown_format_falls_back(self):
        mock_fallback = MagicMock(return_value="fallback_loader")
        with patch(f"{_MODULE}._LC_LOADER_MAP", {}):
            with patch(f"{_MODULE}.LC_FallbackLoader", mock_fallback):
                with patch(
                    f"{_MODULE}.FileFormat.from_extension",
                    return_value=FileFormat.UNKNOWN,
                ):
                    from agent_platform.integrations.loader.file_loaders.unstructured_loader import (
                        UnstructuredFileLoader,
                    )

                    loader = UnstructuredFileLoader()
                    result = loader._loader("doc.xyz")
                    assert result == "fallback_loader"

    def test_loader_sets_mode_elements_default(self):
        mock_cls = MagicMock()
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.PDF: mock_cls}):
            with patch(
                f"{_MODULE}.FileFormat.from_extension", return_value=FileFormat.PDF
            ):
                from agent_platform.integrations.loader.file_loaders.unstructured_loader import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                loader._loader("doc.pdf")
                mock_cls.assert_called_once_with("doc.pdf", mode="elements")

    def test_loader_respects_custom_mode_kwarg(self):
        mock_cls = MagicMock()
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.PDF: mock_cls}):
            with patch(
                f"{_MODULE}.FileFormat.from_extension", return_value=FileFormat.PDF
            ):
                from agent_platform.integrations.loader.file_loaders.unstructured_loader import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                loader._loader("doc.pdf", mode="single")
                mock_cls.assert_called_once_with("doc.pdf", mode="single")

    def test_loader_passes_additional_kwargs(self):
        mock_cls = MagicMock()
        with patch(f"{_MODULE}._LC_LOADER_MAP", {FileFormat.PDF: mock_cls}):
            with patch(
                f"{_MODULE}.FileFormat.from_extension", return_value=FileFormat.PDF
            ):
                from agent_platform.integrations.loader.file_loaders.unstructured_loader import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader()
                loader._loader("doc.pdf", mode="elements", strategy="ocr")
                mock_cls.assert_called_once_with(
                    "doc.pdf", mode="elements", strategy="ocr"
                )

    def test_loader_map_contains_expected_keys(self):
        from agent_platform.integrations.loader.file_loaders.unstructured_loader import (
            _LC_LOADER_MAP,
        )

        assert FileFormat.PDF in _LC_LOADER_MAP
        assert FileFormat.MARKDOWN in _LC_LOADER_MAP
        assert FileFormat.HTML in _LC_LOADER_MAP
        assert FileFormat.DOCX in _LC_LOADER_MAP
        assert len(_LC_LOADER_MAP) == 4
