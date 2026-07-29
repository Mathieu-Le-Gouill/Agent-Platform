from agent_platform.core.schemas.enums import Language

__all__ = ["parse_language"]


def parse_language(code: str) -> Language | None:
    try:
        return Language(code.split("-")[0].split("_")[0])
    except ValueError:
        return None
