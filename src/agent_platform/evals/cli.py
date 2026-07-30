from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from pydantic import BaseModel, ValidationError

from agent_platform.agents.agent import Agent
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.config.container import build_provider
from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.schemas.enums import DatasetSplit
from agent_platform.evals.errors import EvalDatasetError
from agent_platform.evals.runner import EvalRunner
from agent_platform.evals.schemas import EvalCase
from agent_platform.evals.scorer import (
    ExactMatchScorer,
    LLMJudgeScorer,
    Scorer,
    ToolSelectionScorer,
)
from agent_platform.evals.targets import agent_executor_target
from agent_platform.integrations.dataset.huggingface.config import (
    HuggingFaceDatasetConfig,
)
from agent_platform.integrations.dataset.huggingface.provider import (
    HuggingFaceDatasetProvider,
)


class _SearchStubInput(BaseModel):
    query: str


class _SearchStubTool(Tool):
    name = "search"
    description = "Search indexed documents by semantic similarity to a text query."
    input_schema = _SearchStubInput

    async def run(self, **kwargs: Any) -> str:
        return "stub search result"


class _OCRStubInput(BaseModel):
    image_path: str


class _OCRStubTool(Tool):
    name = "ocr"
    description = "Extract text from an image via optical character recognition."
    input_schema = _OCRStubInput

    async def run(self, **kwargs: Any) -> str:
        return "stub ocr result"


class _TranslateStubInput(BaseModel):
    text: str
    target_language: str


class _TranslateStubTool(Tool):
    name = "translate"
    description = "Translate text from one language to another."
    input_schema = _TranslateStubInput

    async def run(self, **kwargs: Any) -> str:
        return "stub translation result"


def _build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (_SearchStubTool(), _OCRStubTool(), _TranslateStubTool()):
        registry.register(tool)
    return registry


def _build_scorer(name: str, llm: BaseLLMProvider, model: str) -> Scorer:
    if name == "tool-selection":
        return ToolSelectionScorer()
    if name == "exact":
        return ExactMatchScorer()
    if name == "llm-judge":
        return LLMJudgeScorer(llm, model)
    raise SystemExit(
        f"--scorer {name!r} requires an embeddings provider, not supported by this "
        "CLI yet; use tool-selection, exact, or llm-judge"
    )


def _load_dataset(path: str) -> list[EvalCase]:
    config = HuggingFaceDatasetConfig(data_files=path)
    try:
        splits = HuggingFaceDatasetProvider[EvalCase]().load(EvalCase, "json", config)
        return list(splits[DatasetSplit.TRAIN])
    except (ProviderError, ValidationError) as exc:
        raise EvalDatasetError(f"{path}: invalid eval dataset: {exc}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-platform-eval")
    parser.add_argument("dataset", help="Path to a JSONL eval dataset")
    parser.add_argument(
        "--provider",
        default="OpenAILLM",
        help="LLM provider class name, as exported by agent_platform.integrations.llm",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model identifier")
    parser.add_argument(
        "--scorer",
        choices=["tool-selection", "exact", "llm-judge"],
        default="tool-selection",
    )
    parser.add_argument("--threshold", type=float, default=0.7)
    return parser


async def _run(args: argparse.Namespace) -> int:
    import agent_platform.integrations.llm as llm_module

    llm = build_provider(llm_module, args.provider)
    scorer = _build_scorer(args.scorer, llm, args.model)

    agent = Agent(
        name="eval-agent",
        llm=llm,
        tool_registry=_build_tool_registry(),
        model=args.model,
    )
    executor = AgentExecutor(agent)
    dataset = _load_dataset(args.dataset)
    runner = EvalRunner(agent_executor_target(executor), [scorer])
    report = await runner.arun(dataset)

    print(
        f"Eval run {report.run_id}: {len(report.results)} cases, "
        f"pass rate {report.pass_rate:.1%}"
    )
    for tag, rate in report.by_tag().items():
        print(f"  tag {tag}: {rate:.1%}")
    for result in report.results:
        if not result.passed:
            print(f"  FAIL {result.case_id}: {result.error or result.scores}")

    return 0 if report.pass_rate >= args.threshold else 1


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        exit_code = asyncio.run(_run(args))
    except EvalDatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(exit_code)


__all__ = ["main", "build_arg_parser"]
