# Agents — Status & Roadmap

The agents layer is the top-level composition: LLM reasoning, tool calls, and conversation state.

```
Agent = LLM (reasoning) + Tools (capabilities) + Executor (loop control)
```

Each agent skill should be:
- **Self-contained** — clear input/output contract
- **Reusable** — composable into larger workflows
- **Language-driven** — LLM for planning and reasoning
- **Observable** — errors surfaced via `AgentError` hierarchy

## Implemented

### Tool Abstraction (`agents/tools/`)

| Component | File | What It Does |
|---|---|---|
| `Tool` Protocol | `tools/base.py` | `name`, `description`, `input_schema: type[BaseModel]`, `output_schema`, `async run(**kwargs) -> Any`; validates at subclass definition time |
| `ToolRegistry` | `tools/registry.py` | Register/get/remove/iterate; `resolve_call()` dispatches a `ToolCall`; `call_and_wrap()` returns a `ToolMessage` (captures errors as `is_error=True`) |
| `TranscribeTool` | `tools/transcribe.py` | Wraps `BaseSpeechToText` — audio → transcript |
| `SearchTool` | `tools/search.py` | Embeds query → vector store search → ranked chunks |
| `OCRTool` | `tools/ocr.py` | Wraps `BaseOCRProvider` — image → extracted text |
| `safe_call()` | `tools/_utils.py` | Error-wrapping helper for provider calls inside tools |

### Agent Runtime (`agents/`)

| Component | File | What It Does |
|---|---|---|
| `Agent` | `agent.py` | Holds `system_prompt`, `tool_registry`, `llm`; `think()` → `AssistantMessage`; `act()` → `list[ToolMessage]`; `step()` = think + act |
| `AgentExecutor` | `executor.py` | think-act loop with `max_iterations` guard; `run(user_input)` and `run_with_messages(messages)` |
| `ConversationAgent` | `conversation.py` | Extends `Agent` with persistent `_history`, `chat()`, optional history truncation by turn count |

### Error Types (`agents/errors.py`)

```
AgentError
├── AgentThinkError    (LLM generation failed)
├── AgentActError      (tool execution failed at agent level)
└── AgentMaxIterations (loop exceeded max_iterations)
```

Tool-level errors live in `tools/errors.py`:
```
ToolError
├── ToolNotFoundError
└── ToolRegistrationError
```

## What's Left to Build

### Remaining Tools

| Tool | Backing Integration | Status |
|---|---|---|
| `TranslateTool` | `BaseTranslator` — text → target language | Not started |
| `SummarizeTool` | LLM — text → summary | Not started |
| `ClassifyTool` | `BaseClassificationProvider` — text → label | Not started |
| `GenerateImageTool` | `BaseImageGenerator` — prompt → image | Not started |

### Infrastructure

- **Workflows** (`workflows/`) — LangGraph state machine for multi-turn agents (stub)
- **API Layer** (`api/`) — FastAPI REST + WebSocket exposing agents (stub)
- **Factories** (`factories/`) — Provider resolution from config/env (stub)
- **Streaming** — `Agent.think()` does not yet support streaming responses

### Specialized Agents

`MeetingNotesAgent`, `DocumentQAAgent`, `TranslationAgent` — each is a `ConversationAgent` subclass with a fixed system prompt, tool set, and domain-specific logic.

## Adding a New Tool

1. Create `agents/tools/<name>.py`
2. Define an `input_schema` class (Pydantic `BaseModel`)
3. Implement the `Tool` protocol: set `name`, `description`, `input_schema`; implement `async run(**kwargs) -> Any`
4. Wrap provider calls with `safe_call()` from `_utils.py` so errors surface as `ToolError`
5. Register with `ToolRegistry` at construction time

```python
class MyTool:
    name = "my_tool"
    description = "Does something useful."
    input_schema = MyInput

    def __init__(self, backend: BaseMyProvider) -> None:
        self._backend = backend

    async def run(self, **kwargs: Any) -> Any:
        input = MyInput(**kwargs)
        return await safe_call(self._backend.do_thing, input.value)
```
