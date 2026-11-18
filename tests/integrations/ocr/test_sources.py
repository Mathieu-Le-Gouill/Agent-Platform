from urllib.error import HTTPError
from urllib.request import OpenerDirector

import pytest

from agent_platform.integrations.ocr import sources
from agent_platform.integrations.ocr.sources import load_bytes

_REDIRECT_CODES = [301, 302, 303, 307, 308]


def test_load_bytes_returns_body_of_ok_answer(http_server):
    http_server.ok("/image.png", b"image-bytes")

    assert load_bytes(http_server.url("/image.png")) == b"image-bytes"


@pytest.mark.parametrize("code", _REDIRECT_CODES)
def test_load_bytes_refuses_redirect_without_fetching_target(http_server, code):
    http_server.redirect("/image.png", code, "/target")
    http_server.ok("/target", b"target-bytes")

    with pytest.raises(HTTPError) as excinfo:
        load_bytes(http_server.url("/image.png"))

    assert excinfo.value.code == code
    assert "/target" not in http_server.requested


def test_load_bytes_refuses_redirect_to_metadata_path(http_server):
    http_server.redirect("/image.png", 302, http_server.url("/latest/meta-data/"))
    http_server.ok("/latest/meta-data/", b"credentials")

    with pytest.raises(HTTPError):
        load_bytes(http_server.url("/image.png"))

    assert http_server.requested == ["/image.png"]


def test_open_url_fetches_with_ten_second_timeout(http_server, mocker):
    http_server.ok("/image.png", b"image-bytes")
    spy = mocker.spy(OpenerDirector, "open")

    with sources.open_url(http_server.url("/image.png")) as response:
        assert response.read() == b"image-bytes"

    assert spy.call_args.kwargs["timeout"] == 10
