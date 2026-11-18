from email.message import Message
from http.client import HTTPResponse
from typing import IO, NoReturn
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

__all__ = ["load_bytes", "open_url"]

_FETCH_TIMEOUT_SECONDS = 10


class _RefuseRedirects(HTTPRedirectHandler):
    """Raises on every 3xx instead of following it.

    The OCR source policy only checks the host of the first URL, so a
    redirect could reach a host it never allowed (internal services, cloud
    metadata). The operator lists the final host instead.
    """

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> NoReturn:
        raise HTTPError(req.full_url, code, msg, headers, fp)


_OPENER = build_opener(_RefuseRedirects)


def open_url(url: str) -> HTTPResponse:
    """Opens ``url`` with a 10 s timeout, refusing redirects.

    Use it as a context manager to close the response.

    Raises:
        urllib.error.HTTPError: on a 3xx answer (with its status) or any
            other error status; the redirect target is never requested.
    """
    response: HTTPResponse = _OPENER.open(url, timeout=_FETCH_TIMEOUT_SECONDS)
    return response


def load_bytes(source: str) -> bytes:
    if source.startswith("http://") or source.startswith("https://"):
        with open_url(source) as response:
            return response.read()
    with open(source, "rb") as f:
        return f.read()
