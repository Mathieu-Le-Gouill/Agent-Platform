import pytest


@pytest.fixture
def capture_client_kwargs(monkeypatch):
    """Patch one or more client/store classes on a module with fakes that
    record their constructor kwargs into a single shared dict.

    Usage: capture_client_kwargs(mod, "QdrantClient", "QdrantVectorStore")
    returns the dict that will hold whichever kwargs the code under test
    actually passes to any of the patched classes.
    """
    captured: dict = {}

    def _fake_cls():
        class _Fake:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        return _Fake

    def _patch(module, *attr_names: str) -> dict:
        for name in attr_names:
            monkeypatch.setattr(module, name, _fake_cls())
        return captured

    return _patch
