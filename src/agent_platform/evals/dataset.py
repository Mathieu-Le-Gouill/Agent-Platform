from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from agent_platform.evals.errors import EvalDatasetError
from agent_platform.evals.schemas import EvalCase


class EvalDataset:
    def __init__(self, cases: list[EvalCase]) -> None:
        self._cases = cases

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalDataset:
        cases: list[EvalCase] = []
        for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(EvalCase(**json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise EvalDatasetError(
                    f"{path}:{line_no}: invalid eval case: {exc}"
                ) from exc
        return cls(cases)

    def filter_by_tag(self, tag: str) -> EvalDataset:
        return EvalDataset([c for c in self._cases if tag in c.tags])

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self._cases)

    def __len__(self) -> int:
        return len(self._cases)


__all__ = ["EvalDataset"]
