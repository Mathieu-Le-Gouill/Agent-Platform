from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent_platform.agents.tools import (
    OCRInput,
    OCRSourcePolicy,
    OCRTool,
    ToolError,
    ToolSourceNotAllowedError,
)
from agent_platform.components.ingestion.ocr.component import OCR
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import ImageDocument
from agent_platform.core.schemas.enums import ImageFormat
from agent_platform.core.schemas.message import ImageBlock, TextBlock
from agent_platform.core.schemas.score import Score


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.extract = AsyncMock(
        return_value=[
            TextChunk(
                id=uuid4(), text="Hello", index=0, confidence=Score.confidence(0.95)
            ),
            TextChunk(
                id=uuid4(), text="World", index=1, confidence=Score.confidence(0.87)
            ),
        ]
    )
    return provider


@pytest.fixture
def tool(mock_provider):
    policy = OCRSourcePolicy(
        allowed_dir=Path("/path/to"), allowed_hosts=frozenset({"example.com"})
    )
    return OCRTool(ocr=OCR(backend=mock_provider), policy=policy)


class TestOCRInput:
    def test_valid_input(self):
        inp = OCRInput(source="/path/to/image.png")
        assert inp.source == "/path/to/image.png"
        assert inp.min_confidence == 0.0

    def test_min_confidence_range(self):
        with pytest.raises(ValidationError):
            OCRInput(source="img.png", min_confidence=1.5)
        with pytest.raises(ValidationError):
            OCRInput(source="img.png", min_confidence=-0.1)

    def test_defaults(self):
        inp = OCRInput(source="img.png")
        assert inp.min_confidence == 0.0


class TestOCRTool:
    def test_name_and_description(self, tool):
        assert tool.name == "ocr"
        assert tool.description

    def test_input_schema(self, tool):
        assert tool.input_schema is OCRInput

    @pytest.mark.asyncio
    async def test_run_success(self, tool, mock_provider):
        results = await tool.run(source="/path/to/image.png")
        assert len(results) == 2
        assert results[0].text == "Hello"
        assert results[1].text == "World"
        mock_provider.extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_with_min_confidence(self, tool, mock_provider):
        mock_provider.extract = AsyncMock(
            return_value=[
                TextChunk(
                    id=uuid4(),
                    text="Low",
                    index=0,
                    confidence=Score.confidence(0.3),
                ),
                TextChunk(
                    id=uuid4(),
                    text="High",
                    index=1,
                    confidence=Score.confidence(0.9),
                ),
            ]
        )
        results = await tool.run(source="/path/to/image.png", min_confidence=0.5)
        assert len(results) == 1
        assert results[0].text == "High"

    @pytest.mark.asyncio
    async def test_run_empty_results(self, tool, mock_provider):
        mock_provider.extract = AsyncMock(return_value=[])
        results = await tool.run(source="/path/to/image.png")
        assert results == []

    @pytest.mark.asyncio
    async def test_run_all_filtered_out(self, tool, mock_provider):
        mock_provider.extract = AsyncMock(
            return_value=[
                TextChunk(
                    id=uuid4(),
                    text="Low",
                    index=0,
                    confidence=Score.confidence(0.1),
                ),
            ]
        )
        results = await tool.run(source="/path/to/image.png", min_confidence=0.5)
        assert results == []

    @pytest.mark.asyncio
    async def test_run_provider_error_wrapped(self, tool, mock_provider):
        mock_provider.extract = AsyncMock(side_effect=RuntimeError("ocr failed"))
        with pytest.raises(ToolError, match="OCR extraction failed"):
            await tool.run(source="/path/to/image.png")

    @pytest.mark.asyncio
    async def test_run_missing_source_raises(self, tool):
        with pytest.raises(ValidationError):
            await tool.run(min_confidence=0.5)


class TestOCRSourcePolicy:
    @pytest.fixture
    def folder(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        (allowed / "scan.png").write_bytes(b"png")
        return allowed

    def _tool(self, mock_provider, policy=None):
        return OCRTool(ocr=OCR(backend=mock_provider), policy=policy)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("source", ["/etc/passwd", "https://example.com/a.png"])
    async def test_default_policy_denies_everything(self, mock_provider, source):
        tool = self._tool(mock_provider)

        with pytest.raises(ToolSourceNotAllowedError):
            await tool.run(source=source)

        mock_provider.extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allowed_path_reaches_the_backend_resolved(
        self, mock_provider, folder
    ):
        tool = self._tool(mock_provider, OCRSourcePolicy(allowed_dir=folder))

        await tool.run(source=str(folder / "sub" / ".." / "scan.png"))

        sent = mock_provider.extract.await_args.kwargs["source"]
        assert sent == str((folder / "scan.png").resolve())

    @pytest.mark.asyncio
    async def test_parent_traversal_is_denied(self, mock_provider, folder):
        tool = self._tool(mock_provider, OCRSourcePolicy(allowed_dir=folder))

        with pytest.raises(ToolSourceNotAllowedError):
            await tool.run(source=str(folder / ".." / "secret.txt"))

        mock_provider.extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_symlink_out_of_the_folder_is_denied(
        self, mock_provider, folder, tmp_path
    ):
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"png")
        (folder / "link.png").symlink_to(outside)
        tool = self._tool(mock_provider, OCRSourcePolicy(allowed_dir=folder))

        with pytest.raises(ToolSourceNotAllowedError):
            await tool.run(source=str(folder / "link.png"))

        mock_provider.extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_path_is_denied_when_only_hosts_are_allowed(self, mock_provider):
        policy = OCRSourcePolicy(allowed_hosts=frozenset({"example.com"}))

        with pytest.raises(ToolSourceNotAllowedError):
            await self._tool(mock_provider, policy).run(source="/etc/passwd")

    @pytest.mark.asyncio
    async def test_allowed_host_reaches_the_backend(self, mock_provider):
        policy = OCRSourcePolicy(allowed_hosts=frozenset({"example.com"}))

        await self._tool(mock_provider, policy).run(
            source="https://EXAMPLE.com/img.png"
        )

        sent = mock_provider.extract.await_args.kwargs["source"]
        assert sent == "https://EXAMPLE.com/img.png"

    @pytest.mark.asyncio
    async def test_metadata_address_is_denied_when_not_listed(self, mock_provider):
        policy = OCRSourcePolicy(allowed_hosts=frozenset({"example.com"}))

        with pytest.raises(ToolSourceNotAllowedError):
            await self._tool(mock_provider, policy).run(
                source="http://169.254.169.254/latest/meta-data"
            )

        mock_provider.extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uppercase_scheme_is_checked_as_a_url_not_a_path(
        self, mock_provider, folder
    ):
        policy = OCRSourcePolicy(
            allowed_dir=folder, allowed_hosts=frozenset({"example.com"})
        )
        tool = self._tool(mock_provider, policy)

        await tool.run(source="HTTP://example.com/img.png")
        sent = mock_provider.extract.await_args.kwargs["source"]
        assert sent == "http://example.com/img.png"
        with pytest.raises(ToolSourceNotAllowedError):
            await tool.run(source="HTTP://evil.test/img.png")

    @pytest.mark.asyncio
    async def test_other_schemes_are_denied(self, mock_provider):
        policy = OCRSourcePolicy(allowed_hosts=frozenset({"example.com"}))

        with pytest.raises(ToolSourceNotAllowedError):
            await self._tool(mock_provider, policy).run(
                source="ftp://example.com/img.png"
            )

    @pytest.mark.asyncio
    async def test_denial_message_does_not_reveal_the_policy(
        self, mock_provider, folder
    ):
        tool = self._tool(mock_provider, OCRSourcePolicy(allowed_dir=folder))

        with pytest.raises(ToolSourceNotAllowedError) as excinfo:
            await tool.run(source="/etc/passwd")

        assert str(folder) not in str(excinfo.value)
        assert "not allowed" in str(excinfo.value)

    def test_denial_is_a_tool_error(self):
        assert issubclass(ToolSourceNotAllowedError, ToolError)


class TestOCRToolToBlocks:
    def test_url_source_builds_image_url_block(self, tool):
        chunks = [TextChunk(id=uuid4(), text="Hello", index=0)]
        blocks = tool.to_blocks("https://example.com/img.png", chunks)
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].image == "https://example.com/img.png"
        assert blocks[1] == TextBlock(text="Hello")

    def test_local_source_loads_image_document(self, tool, mocker):
        chunks = [TextChunk(id=uuid4(), text="Hello", index=0)]
        fake_doc = ImageDocument(content=b"\x89PNG", format=ImageFormat.PNG)
        mocker.patch.object(ImageDocument, "load_content", return_value=fake_doc)
        blocks = tool.to_blocks("/path/to/image.png", chunks)
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].image is fake_doc

    def test_joins_multiple_chunks_into_one_text_block(self, tool):
        chunks = [
            TextChunk(id=uuid4(), text="Hello", index=0),
            TextChunk(id=uuid4(), text="World", index=1),
        ]
        blocks = tool.to_blocks("https://example.com/img.png", chunks)
        assert len(blocks) == 2
        assert blocks[1].text == "Hello\nWorld"

    def test_no_chunks_omits_text_block(self, tool):
        blocks = tool.to_blocks("https://example.com/img.png", [])
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
