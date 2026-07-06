# Agents — Roadmap

This directory is currently a placeholder. Agents will be the top-level composition layer — combining pipelines, LLM reasoning, and tool calls into reusable skills.

## Vision

```
Agent = LLM (reasoning) + Tools (capabilities) + Workflow (state machine)
```

Each agent skill should be:
- **Self-contained** — has a clear input/output contract
- **Reusable** — composable into larger workflows
- **Language-driven** — uses LLM calls for planning and reasoning
- **Observable** — emits events for logging/streaming

## What Needs to Be Built

### 1. Tool Abstraction (`agents/tools/`)

Each tool wraps a pipeline or integration call with a schema:

```python
class Tool(ABC):
    name: str
    description: str
    input_schema: type  # pydantic BaseModel
    output_schema: type

    async def run(self, **kwargs) -> Any: ...
```

**Candidate tools:**
| Tool | Backing Pipeline |
|---|---|
| `Transcribe` | Speech → text |
| `Translate` | Text → target language |
| `Search` | Query → RAG results |
| `OCR` | Image → extracted text |
| `Summarize` | Text → summary |
| `Classify` | Text → label |
| `GenerateImage` | Prompt → image |

### 2. Agent Runtime (`agents/`)

- **Agent** class: LLM + system prompt + list of tools
- **AgentExecutor**: loop (think → act → observe) with LangGraph or custom state machine
- **ConversationAgent**: maintains message history, manages context window
- **Specialized agents**: `MeetingNotesAgent`, `DocumentQAAgent`, `TranslationAgent`

### 3. Workflow Framework (`workflows/`)

Graph-based orchestration using LangGraph:

```
nodes/  ← individual processing steps
edges/  ← transitions (conditional, parallel, loops)
```

### 4. API Layer (`api/`)

Transport for agents:
- REST endpoints (FastAPI) for synchronous requests
- WebSocket for streaming audio/text
- Background task queue for long-running pipelines

## Suggested Build Order

1. **`agents/tools/`** — Create the `Tool` ABC, implement 2-3 concrete tools (Search, Transcribe, OCR) using existing integrations
2. **`agents/agent.py`** — Basic `Agent` class with tool-calling LLM loop
3. **`workflows/`** — Define LangGraph state machine for multi-turn agents
4. **`api/`** — FastAPI app exposing agents via REST + WebSocket
5. **Specialized agents** — Build on top once the framework is solid
