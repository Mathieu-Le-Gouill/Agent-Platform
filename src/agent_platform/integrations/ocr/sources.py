from urllib.request import urlopen

__all__ = ["load_bytes"]

_FETCH_TIMEOUT_SECONDS = 10


def load_bytes(source: str) -> bytes:
    if source.startswith("http://") or source.startswith("https://"):
        with urlopen(source, timeout=_FETCH_TIMEOUT_SECONDS) as response:
            return response.read()
    with open(source, "rb") as f:
        return f.read()
