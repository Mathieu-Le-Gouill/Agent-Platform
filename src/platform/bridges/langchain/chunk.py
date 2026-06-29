from langchain_core.documents import Document as LCDocument
from dataclasses import fields, asdict

from utils import parse_uuid, uuid_to_str
from models.chunk import Chunk, ChunkMetadata, CoordinatesMetadata
from models.language import Language


_CHUNK_FIELDS = {f.name for f in fields(Chunk)} - {"metadata"}
_META_FIELDS  = {f.name for f in fields(ChunkMetadata)} - {"extra"}
_RESERVED     = _CHUNK_FIELDS | {"chunk_id"}

# Keys that require special handling and are mapped explicitly below.
_SPECIAL = {"language", "coordinates", "links", "title", "section"}

# Keys consumed by the mapper that are not ChunkMetadata field names.
_ALIASES = {"languages", "filename", "parent_id"}


def from_langchain(lc_doc: LCDocument) -> Chunk:
    m = lc_doc.metadata or {}

    # Fields with non-trivial extraction logic
    language = _parse_language(m.get("language") or next(iter(m.get("languages") or []), None))
    coordinates = _parse_coordinates(m.get("coordinates"))
    links = _parse_links(m.get("links"))

    # All remaining ChunkMetadata fields map 1-to-1 from metadata keys,
    # with two alias fallbacks. Reflection keeps this in sync with the model.
    direct = {
        k: m.get(k)
        for k in _META_FIELDS - _SPECIAL
    }
    direct["title"]   = m.get("title") or m.get("filename")
    direct["section"] = m.get("section") or m.get("parent_id")

    known = _META_FIELDS | _RESERVED | _SPECIAL | _ALIASES
    extra = {k: v for k, v in m.items() if k not in known}

    return Chunk(
        id          = parse_uuid(lc_doc.id),
        document_id = parse_uuid(m.get("document_id")),
        text        = lc_doc.page_content,
        index       = m.get("chunk_index", 0),
        metadata    = ChunkMetadata(**direct, language=language, coordinates=coordinates, links=links, extra=extra),
    )


def to_langchain(chunk: Chunk) -> LCDocument:
    meta = {
        k: getattr(chunk.metadata, k)
        for k in _META_FIELDS
    }

    extra = chunk.metadata.extra or {}

    return LCDocument(
        id=uuid_to_str(chunk.id),
        page_content=chunk.text,
        metadata={
            "chunk_id":    str(chunk.id),
            "document_id": str(chunk.document_id),
            "chunk_index": chunk.index,
            "start_char":  chunk.start_char,
            "end_char":    chunk.end_char,
            **meta,
            **extra,
        },
    )


# --- field parsers ---

def _parse_language(raw: str | None) -> Language | None:
    if not raw:
        return None
    try:
        return Language(raw)
    except ValueError:
        return None


def _parse_coordinates(raw: dict | None) -> CoordinatesMetadata | None:
    if not isinstance(raw, dict):
        return None
    return CoordinatesMetadata(points=raw.get("points"), system=raw.get("system"))


def _parse_links(raw: list | None) -> list[dict] | None:
    if not raw:
        return None
    # Normalise Unstructured Link objects → plain dicts
    return [vars(lk) if hasattr(lk, "__dict__") else lk for lk in raw]


"""
LANGCHAIN DOCUMENT

    id: str | None,
    page_content: str,
    metadata: dict[Any, Any]

"""