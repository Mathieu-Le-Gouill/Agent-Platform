from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any


def make_lazy_provider_accessors(
    providers: dict[str, str], module_globals: dict[str, Any]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build `__getattr__`/`__dir__` pair for a package that lazily re-exports
    provider classes, so importing the package doesn't pull in every provider SDK.

    `providers` maps exported name -> dotted module path to import it from.
    Assign the results to `__getattr__`/`__dir__` in the calling `__init__.py`.
    """
    module_name = module_globals["__name__"]

    def __getattr__(name: str) -> Any:
        if name in providers:
            return getattr(import_module(providers[name]), name)
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted(list(module_globals.keys()) + list(providers.keys()))

    return __getattr__, __dir__
