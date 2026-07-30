# Evals: Regression Testing for Agents and Pipelines

## Design

`evals/` sits above `agents/` in the layer stack: `core` → `integrations` →
`components` → `pipelines` → `agents` → `evals`. It answers a question none
of the layers below it can: did a prompt, model, or agent-loop change
regress behavior? Nothing here is a new provider surface, scorers inject
existing `core/interfaces` ABCs (`BaseLLMProvider`, `BaseEmbeddingProvider`)
directly, the same way `agents/agent.py` injects `BaseLLMProvider` rather
than going through a component wrapper.

## Directory Layout

```
evals/
├── errors.py     # EvalError, EvalDatasetError, EvalRunError
├── schemas.py    # EvalCase, EvalOutput, EvalResult, EvalReport
├── scorer.py     # Scorer protocol + ExactMatchScorer, EmbeddingSimilarityScorer,
│                 # LLMJudgeScorer, ToolSelectionScorer
├── targets.py    # EvalTarget adapters over AgentExecutor / ConversationAgent
├── runner.py     # EvalRunner: concurrent execution, per-case isolation, tracing
├── cli.py        # `agent-platform-eval` entrypoint (registered in pyproject.toml);
│                 # `_load_dataset()` loads JSONL into `list[EvalCase]` via
│                 # `integrations/dataset/huggingface`'s `HuggingFaceDatasetProvider`
└── datasets/
    └── conversation_tool_selection.jsonl   # worked example, see below
```

There is no `evals/dataset.py`: `EvalRunner.arun()` takes any `Iterable[EvalCase]`,
so loading a dataset is just whatever gets you there, `cli.py::_load_dataset()`
for the CLI, a plain list built by hand in tests, or `integrations/dataset/huggingface`
directly for anything richer (streaming, native/ratio-based train/test/eval splits,
non-JSONL sources), see `integrations/README.md`'s `dataset/` row. `_load_dataset()`
translates the provider's `ProviderError`/pydantic `ValidationError` into
`EvalDatasetError`, which `cli.py::main()` catches to print a clean CLI error
instead of a raw traceback.

## Core Flow

1. Load a dataset into `list[EvalCase]` (`input`, optional `expected`/`reference`,
   `tags`, `threshold`), however the caller prefers, see above. The `evals` extra
   (`agent_platform[evals]`, pulling in `datasets`) is needed for `cli.py`'s
   `HuggingFaceDatasetProvider`-backed JSONL loading.
2. Wrap the system under test as an `EvalTarget`
   (`Callable[[EvalCase], Awaitable[EvalOutput]]`) via `targets.py`.
3. Pick one or more `Scorer`s.
4. `EvalRunner(target, scorers).arun(dataset) -> EvalReport`.

Each case runs in isolation: a scorer/target exception is caught, logged,
and recorded as a failed `EvalResult` (`error` set, `passed=False`) rather
than aborting the run, the same pattern `ToolRegistry.call_and_wrap`
(`agents/tools/registry.py`) already uses for tool calls. Cases run
concurrently, bounded by `EvalRunner(concurrency=...)`, and each is wrapped
in a `traced_operation_span("eval_case", ...)` (`core/genai_tracing.py`) tagged
with `agent_platform.eval.run_id`/`agent_platform.eval.case_id`, so a
failing case is debuggable through the existing OTel pipeline.

## Adding a Scorer

Implement the `Scorer` protocol: a `name: str` class attribute plus
`async def score(self, case: EvalCase, output: EvalOutput) -> Score`.
Reuse `core/schemas/score.py::Score` for the result, and if the scorer's
value range isn't `[0, 1]` (e.g. cosine similarity is `[-1, 1]`), pass
explicit `low`/`high` bounds, see `EmbeddingSimilarityScorer` for the
pattern (`core/similarity.py::similarity_bounds`).

## Adding an Eval Target

An `EvalTarget` adapter turns whatever's under test into
`Callable[[EvalCase], Awaitable[EvalOutput]]`. `agent_executor_target`
wraps `AgentExecutor.run_with_messages` and captures the tool calls the
agent actually made; `conversation_agent_target` wraps
`ConversationAgent.chat` and diffs `.history` for the same. A pipeline
target follows the same shape, no new abstraction needed.

## Worked Example: `conversation_tool_selection.jsonl`

Tests whether a `ConversationAgent`/`AgentExecutor` picks the right tool
for a given input, scored by `ToolSelectionScorer` (exact-set match against
`case.reference["expected_tools"]`). Run it via the CLI:

```bash
uv run --extra llm-openai --extra evals --extra dev agent-platform-eval \
    src/agent_platform/evals/datasets/conversation_tool_selection.jsonl \
    --provider OpenAILLM --scorer tool-selection --threshold 0.7
```

This needs a real `OPENAI_API_KEY` (or the relevant provider's credential)
since it exercises an actual LLM's tool-selection behavior, it is not part
of the default `pytest` run. The CLI builds an `Agent` with lightweight
stub tools (`search`/`ocr`/`translate`-shaped: real `name`/`description`/
`input_schema`, canned `run()`), since only tool *selection* is graded
here, not tool execution. It exits non-zero when `EvalReport.pass_rate`
falls below `--threshold`. `.github/workflows/ci.yml`'s `evals` job runs
this, but only on a manual `workflow_dispatch` trigger, never on regular
push/PR CI, since it costs real provider API usage.

## Testing

`tests/evals/` mirrors this layout 1:1, fully mocked (`mocker` fixture, no
live network), per `AGENTS.md §3`. The golden dataset itself is only
exercised for real via the CLI/CI job above, not asserted against with
mocked unit tests, that would make the suite flaky against a live LLM.
