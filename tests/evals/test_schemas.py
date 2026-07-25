from agent_platform.core.schemas.score import Score
from agent_platform.evals.schemas import EvalOutput, EvalReport, EvalResult


def _result(
    case_id: str, passed: bool, tags: list[str], value: float = 1.0
) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        output=EvalOutput(text="hi"),
        scores={"exact_match": Score.confidence(value)},
        passed=passed,
        tags=tags,
    )


class TestEvalReport:
    def test_pass_rate_empty(self):
        assert EvalReport(results=[]).pass_rate == 0.0

    def test_pass_rate_mixed(self):
        report = EvalReport(
            results=[
                _result("a", True, []),
                _result("b", False, []),
                _result("c", True, []),
            ]
        )
        assert report.pass_rate == 2 / 3

    def test_mean_score(self):
        report = EvalReport(
            results=[
                _result("a", True, [], value=1.0),
                _result("b", True, [], value=0.0),
            ]
        )
        assert report.mean_score("exact_match") == 0.5

    def test_mean_score_missing_scorer_is_zero(self):
        report = EvalReport(results=[_result("a", True, [])])
        assert report.mean_score("nonexistent") == 0.0

    def test_by_tag(self):
        report = EvalReport(
            results=[
                _result("a", True, ["tool_selection"]),
                _result("b", False, ["tool_selection"]),
                _result("c", True, ["other"]),
            ]
        )
        by_tag = report.by_tag()
        assert by_tag["tool_selection"] == 0.5
        assert by_tag["other"] == 1.0
