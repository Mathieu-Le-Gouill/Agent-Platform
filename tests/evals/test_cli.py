from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_platform.core.schemas.score import Score
from agent_platform.evals import cli
from agent_platform.evals.schemas import EvalReport, EvalResult


class TestBuildArgParser:
    def test_defaults(self):
        args = cli.build_arg_parser().parse_args(["data.jsonl"])
        assert args.dataset == "data.jsonl"
        assert args.provider == "OpenAILLM"
        assert args.scorer == "tool-selection"
        assert args.threshold == 0.7

    def test_overrides(self):
        args = cli.build_arg_parser().parse_args(
            [
                "data.jsonl",
                "--provider",
                "AnthropicLLM",
                "--scorer",
                "exact",
                "--threshold",
                "0.9",
            ]
        )
        assert args.provider == "AnthropicLLM"
        assert args.scorer == "exact"
        assert args.threshold == 0.9


class TestBuildScorer:
    def test_tool_selection(self):
        assert cli._build_scorer("tool-selection", MagicMock(), "m").name == (
            "tool_selection"
        )

    def test_exact(self):
        assert cli._build_scorer("exact", MagicMock(), "m").name == "exact_match"

    def test_llm_judge(self):
        assert cli._build_scorer("llm-judge", MagicMock(), "m").name == "llm_judge"

    def test_unsupported_scorer_raises(self):
        with pytest.raises(SystemExit):
            cli._build_scorer("embedding", MagicMock(), "m")


class TestStubTools:
    @pytest.mark.asyncio
    async def test_stub_tools_run(self):
        registry = cli._build_tool_registry()
        assert await registry.get("search").run(query="x") == "stub search result"
        assert await registry.get("ocr").run(image_path="x") == "stub ocr result"
        assert (
            await registry.get("translate").run(text="x", target_language="fr")
            == "stub translation result"
        )


class TestMain:
    def test_main_exits_with_run_result(self, mocker):
        mocker.patch("agent_platform.evals.cli._run", AsyncMock(return_value=0))
        mocker.patch("sys.argv", ["agent-platform-eval", "data.jsonl"])

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0


class TestRun:
    @pytest.mark.asyncio
    async def test_exit_zero_when_pass_rate_meets_threshold(self, mocker, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"id": "a", "input": "hi"}\n')

        mocker.patch(
            "agent_platform.evals.cli.build_provider", return_value=MagicMock()
        )
        report = EvalReport(
            results=[
                EvalResult(
                    case_id="a",
                    output=None,
                    scores={"tool_selection": Score.confidence(1.0)},
                    passed=True,
                    tags=["tool_selection"],
                )
            ]
        )
        mocker.patch.object(cli.EvalRunner, "arun", AsyncMock(return_value=report))

        args = cli.build_arg_parser().parse_args([str(dataset_path)])
        exit_code = await cli._run(args)

        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_exit_one_when_pass_rate_below_threshold(self, mocker, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"id": "a", "input": "hi"}\n')

        mocker.patch(
            "agent_platform.evals.cli.build_provider", return_value=MagicMock()
        )
        report = EvalReport(
            results=[
                EvalResult(
                    case_id="a",
                    output=None,
                    scores={},
                    passed=False,
                    error="boom",
                )
            ]
        )
        mocker.patch.object(cli.EvalRunner, "arun", AsyncMock(return_value=report))

        args = cli.build_arg_parser().parse_args([str(dataset_path)])
        exit_code = await cli._run(args)

        assert exit_code == 1
