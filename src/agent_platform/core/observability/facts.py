from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, get_type_hints

from agent_platform.core.interfaces.tracing.span import AttributeValue
from agent_platform.core.observability.content import payload_facts

__all__ = [
    "ERROR_TYPE_KEY",
    "OPERATION_KEY",
    "Fact",
    "Observation",
    "facts_of",
    "flatten",
]

# The two keys the runtime sets itself rather than reading off an observation:
# which operation ran, and how it failed. Platform-native like every other
# key here; a backend maps them to whatever its own convention calls them.
OPERATION_KEY = "operation.name"
ERROR_TYPE_KEY = "error.type"

# Suffixes a payload field expands into. Size is always exported, the digest
# and the content only when capture is enabled (see `content.py`).
_SIZE_SUFFIX = ".size"
_DIGEST_SUFFIX = ".digest"
_CONTENT_SUFFIX = ".content"

_facts_cache: dict[type[Observation], tuple[tuple[str, Fact], ...]] = {}


@dataclass(frozen=True, slots=True)
class Fact:
    """Marks an observation field as something telemetry should carry.

    `key` is the platform's own name for it, never a backend's: the whole
    point of the split is that `core` describes what happened and an
    integration decides how to spell it on the wire.

    `payload` marks a value that is user data rather than a label. Those are
    never exported verbatim unless content capture is on; what is always
    exported is the size the runtime derives from them.
    """

    key: str
    payload: bool = False


class Observation:
    """One unit of work the platform observed, described in its own terms.

    Subclasses declare what happened as typed fields annotated with `Fact`,
    not as an attribute bag: a call site sets `call.response_model = ...` and
    the runtime derives the span attributes and the payload-capture policy
    from the same object. That is what keeps any backend's vocabulary out of
    `core`.

    Mutability is deliberate. Most of what is worth recording is only known
    once the work is done, so `observe()` hands the observation back to the
    call site, records what is already known when the span opens, and flushes
    whatever changed when it closes.
    """

    #: Name of the operation, used as the span name. Overridden through
    #: `operation_name` where it is only known per instance.
    operation: ClassVar[str] = ""

    #: Set by `record_error`. Not a dataclass field: subclasses declare their
    #: own fields and this must not force an ordering on them.
    error: Exception | None = None

    @property
    def operation_name(self) -> str:
        return self.operation

    def record_error(self, exc: Exception) -> None:
        """Mark this observation as failed by an error the call site swallowed.

        For work that catches an error and turns it into a result instead of
        re-raising (a tool call becoming a `ToolResult`): the failure is still
        worth recording, but no stack trace is attached since the exception
        never leaves that call site. An error that does propagate out of
        `observe()` is recorded without this.
        """
        self.error = exc


def facts_of(cls: type[Observation]) -> tuple[tuple[str, Fact], ...]:
    """The `Fact`-annotated fields of `cls`, including inherited ones.

    Resolved on first use rather than at class creation so a field may be
    annotated with a type defined later in its own module, then cached: every
    observation of a given kind has the same fields.
    """
    cached = _facts_cache.get(cls)
    if cached is not None:
        return cached

    resolved: list[tuple[str, Fact]] = []
    for name, hint in get_type_hints(cls, include_extras=True).items():
        for extra in getattr(hint, "__metadata__", ()):
            if isinstance(extra, Fact):
                resolved.append((name, extra))

    facts = tuple(resolved)
    _facts_cache[cls] = facts
    return facts


def flatten(observation: Observation) -> dict[str, AttributeValue]:
    """Reduce `observation` to the span attributes describing it.

    Fields left at `None` are dropped rather than exported, so an observation
    can declare everything it might learn without a call site branching on
    what it actually did.
    """
    attributes: dict[str, AttributeValue] = {}

    operation = observation.operation_name
    if operation:
        attributes[OPERATION_KEY] = operation

    for name, fact in facts_of(type(observation)):
        value = getattr(observation, name, None)
        if value is None:
            continue
        if fact.payload:
            attributes.update(_payload_attributes(fact.key, value))
            continue
        attributes[fact.key] = value

    return attributes


def _payload_attributes(key: str, value: Any) -> Iterator[tuple[str, AttributeValue]]:
    facts = payload_facts(value)
    yield key + _SIZE_SUFFIX, facts.size
    if facts.digest is not None:
        yield key + _DIGEST_SUFFIX, facts.digest
    if facts.content is not None:
        yield key + _CONTENT_SUFFIX, facts.content
