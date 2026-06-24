from __future__ import annotations
from typing import TYPE_CHECKING, Union
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from core.entities.document import Document
    from core.entities.transcript import Transcript
    from core.entities.message import Message

Content: TypeAlias = Union["Document", "Transcript", "Message"]