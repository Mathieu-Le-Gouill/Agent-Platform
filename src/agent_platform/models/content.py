from __future__ import annotations
from typing import TYPE_CHECKING, Union
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from agent_platform.models.document import Document
    from agent_platform.models.transcript import Transcript
    from agent_platform.models.message import Message

Content: TypeAlias = Union["Document", "Transcript", "Message"]