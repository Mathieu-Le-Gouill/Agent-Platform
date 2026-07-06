from agent_platform.models.enums import (
    AudioFormat,
    DataType,
    DocumentFormat,
    FileFormat,
    ImageFormat,
    Language,
    MediaFormat,
    MediaType,
    VideoFormat,
)


class TestFileFormatFromExtension:
    def test_document_extensions(self):
        assert FileFormat.from_extension("pdf") == FileFormat.PDF
        assert FileFormat.from_extension("docx") == FileFormat.DOCX
        assert FileFormat.from_extension("xlsx") == FileFormat.XLSX
        assert FileFormat.from_extension("pptx") == FileFormat.PPTX
        assert FileFormat.from_extension("txt") == FileFormat.TXT
        assert FileFormat.from_extension("csv") == FileFormat.CSV
        assert FileFormat.from_extension("json") == FileFormat.JSON
        assert FileFormat.from_extension("html") == FileFormat.HTML
        assert FileFormat.from_extension("markdown") == FileFormat.MARKDOWN

    def test_image_extensions(self):
        assert FileFormat.from_extension("png") == FileFormat.PNG
        assert FileFormat.from_extension("jpg") == FileFormat.JPEG
        assert FileFormat.from_extension("jpeg") == FileFormat.JPEG
        assert FileFormat.from_extension("webp") == FileFormat.WEBP
        assert FileFormat.from_extension("gif") == FileFormat.GIF
        assert FileFormat.from_extension("bmp") == FileFormat.BMP
        assert FileFormat.from_extension("tiff") == FileFormat.TIFF

    def test_audio_extensions(self):
        assert FileFormat.from_extension("mp3") == FileFormat.MP3
        assert FileFormat.from_extension("wav") == FileFormat.WAV
        assert FileFormat.from_extension("flac") == FileFormat.FLAC
        assert FileFormat.from_extension("aac") == FileFormat.AAC
        assert FileFormat.from_extension("m4a") == FileFormat.M4A
        assert FileFormat.from_extension("ogg") == FileFormat.OGG

    def test_video_extensions(self):
        assert FileFormat.from_extension("mp4") == FileFormat.MP4
        assert FileFormat.from_extension("mov") == FileFormat.MOV
        assert FileFormat.from_extension("avi") == FileFormat.AVI
        assert FileFormat.from_extension("mkv") == FileFormat.MKV
        assert FileFormat.from_extension("webm") == FileFormat.WEBM

    def test_unknown_extension_returns_unknown(self):
        assert FileFormat.from_extension("xyz") == FileFormat.UNKNOWN
        assert FileFormat.from_extension("") == FileFormat.UNKNOWN

    def test_case_insensitive(self):
        assert FileFormat.from_extension("PDF") == FileFormat.PDF
        assert FileFormat.from_extension("Mp3") == FileFormat.MP3


class TestFileFormatFromMime:
    def test_known_mime_types(self):
        assert FileFormat.from_mime("application/pdf") == FileFormat.PDF
        assert FileFormat.from_mime("text/plain") == FileFormat.TXT
        assert FileFormat.from_mime("text/html") == FileFormat.HTML
        assert FileFormat.from_mime("text/markdown") == FileFormat.MARKDOWN
        assert FileFormat.from_mime("text/x-markdown") == FileFormat.MARKDOWN
        assert FileFormat.from_mime("text/csv") == FileFormat.CSV
        assert FileFormat.from_mime("application/json") == FileFormat.JSON
        assert FileFormat.from_mime("application/msword") == FileFormat.DOCX

    def test_docx_long_mime(self):
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert FileFormat.from_mime(mime) == FileFormat.DOCX

    def test_xlsx_mime(self):
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert FileFormat.from_mime(mime) == FileFormat.XLSX

    def test_pptx_mime(self):
        mime = (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
        assert FileFormat.from_mime(mime) == FileFormat.PPTX

    def test_unknown_mime_returns_unknown(self):
        assert FileFormat.from_mime("application/octet-stream") == FileFormat.UNKNOWN

    def test_audio_prefix_returns_unknown(self):
        assert FileFormat.from_mime("audio/mpeg") == FileFormat.UNKNOWN
        assert FileFormat.from_mime("audio/wav") == FileFormat.UNKNOWN

    def test_image_prefix_returns_unknown(self):
        assert FileFormat.from_mime("image/png") == FileFormat.UNKNOWN
        assert FileFormat.from_mime("image/jpeg") == FileFormat.UNKNOWN

    def test_video_prefix_returns_unknown(self):
        assert FileFormat.from_mime("video/mp4") == FileFormat.UNKNOWN
        assert FileFormat.from_mime("video/webm") == FileFormat.UNKNOWN

    def test_mime_with_parameters(self):
        assert FileFormat.from_mime("text/plain; charset=utf-8") == FileFormat.TXT

    def test_mime_with_semicolon_and_spaces(self):
        assert (
            FileFormat.from_mime("  application/json; charset=utf-8  ")
            == FileFormat.JSON
        )


class TestFileFormatFromMediaType:
    def test_text_pdf(self):
        assert FileFormat.from_media_type(MediaType.TEXT, "pdf") == FileFormat.PDF

    def test_text_txt(self):
        assert FileFormat.from_media_type(MediaType.TEXT, "txt") == FileFormat.TXT

    def test_image_png(self):
        assert FileFormat.from_media_type(MediaType.IMAGE, "png") == FileFormat.PNG

    def test_audio_wav(self):
        assert FileFormat.from_media_type(MediaType.AUDIO, "wav") == FileFormat.WAV

    def test_video_mp4(self):
        assert FileFormat.from_media_type(MediaType.VIDEO, "mp4") == FileFormat.MP4

    def test_invalid_combo_returns_unknown(self):
        assert FileFormat.from_media_type(MediaType.TEXT, "png") == FileFormat.UNKNOWN

    def test_unknown_media_type_returns_unknown(self):
        assert (
            FileFormat.from_media_type(MediaType.UNKNOWN, "xyz") == FileFormat.UNKNOWN
        )

    def test_nonexistent_format_returns_unknown(self):
        assert FileFormat.from_media_type(MediaType.AUDIO, "xyz") == FileFormat.UNKNOWN


class TestFileFormatProperties:
    def test_extension_property(self):
        assert FileFormat.PDF.extension == "pdf"
        assert FileFormat.JPEG.extension == "jpeg"
        assert FileFormat.WAV.extension == "wav"
        assert FileFormat.MP4.extension == "mp4"
        assert FileFormat.UNKNOWN.extension is None

    def test_media_property(self):
        assert FileFormat.PDF.media == "text"
        assert FileFormat.JPEG.media == "image"
        assert FileFormat.WAV.media == "audio"
        assert FileFormat.MP4.media == "video"
        assert FileFormat.UNKNOWN.media == "unknown"


class TestMediaFormat:
    def test_is_union_of_format_types(self):
        assert hasattr(MediaFormat, "__args__")
        args = MediaFormat.__args__
        assert DocumentFormat in args
        assert ImageFormat in args
        assert AudioFormat in args
        assert VideoFormat in args

    def test_not_a_single_type(self):
        assert MediaFormat is not DocumentFormat
        assert MediaFormat is not ImageFormat
        assert MediaFormat is not AudioFormat
        assert MediaFormat is not VideoFormat


class TestLanguage:
    def test_code_property(self):
        assert Language.EN.code == "en"
        assert Language.FR.code == "fr"
        assert Language.GE.code == "de"
        assert Language.SP.code == "es"
        assert Language.CH.code == "zh"
        assert Language.JA.code == "ja"

    def test_code_matches_value(self):
        for lang in Language:
            assert lang.code == lang.value


class TestDataType:
    def test_values(self):
        assert DataType.INT16.value == "int16"
        assert DataType.FLOAT32.value == "float32"
        assert DataType.INT8.value == "int8"
        assert DataType.UINT8.value == "uint8"
