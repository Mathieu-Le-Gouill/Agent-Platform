from agent_platform.core.schemas.enums import FileFormat


class TestMediaFormatFromFile:
    def test_png_file(self):
        result = FileFormat.from_path("image.png")
        assert result == FileFormat.PNG

    def test_jpg_file(self):
        result = FileFormat.from_path("photo.jpg")
        assert result == FileFormat.JPEG

    def test_mp3_file(self):
        result = FileFormat.from_path("song.mp3")
        assert result == FileFormat.MP3

    def test_mp4_file(self):
        result = FileFormat.from_path("video.mp4")
        assert result == FileFormat.MP4

    def test_pdf_file(self):
        result = FileFormat.from_path("doc.pdf")
        assert result == FileFormat.PDF

    def test_txt_file(self):
        result = FileFormat.from_path("notes.txt")
        assert result == FileFormat.TXT

    def test_unknown_extension(self):
        result = FileFormat.from_path("file.xyz")
        assert result == FileFormat.UNKNOWN

    def test_no_extension(self):
        result = FileFormat.from_path("README")
        assert result == FileFormat.UNKNOWN
