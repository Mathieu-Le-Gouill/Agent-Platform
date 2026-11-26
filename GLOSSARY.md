# Agent Platform

A library for building LLM agents, multi-agent workflows and the pipelines that feed them (retrieval, speech, image generation), on top of interchangeable vendor providers.

## Layers

**Domain**:
One capability the platform can delegate to a vendor, such as LLM, embeddings, vector store, OCR or speech-to-text.
_Avoid_: Service, category

**Provider**:
One vendor's implementation of a Domain, such as the OpenAI LLM or the Chroma vector store.
_Avoid_: Backend, adapter, driver, integration (for a single vendor)

**Component**:
A reusable processing unit with a typed input and output that wraps one or more Providers and may add logic of its own.
_Avoid_: Service, module, block

**Pipeline**:
A fixed sequence of Components that turns one kind of input into another, such as ingesting Documents into a vector store.
_Avoid_: Flow, chain, workflow

**Model String**:
The `"<provider>:<model>"` text that names both which Provider to use and which of its models to bind.
_Avoid_: Model id, model name (for the combined form)

**LLM Task**:
One prompt paired with the type of answer it must return, runnable on any LLM Provider.
_Avoid_: Classifier, chain, prompt template

**Preset**:
A ready-made LLM Task for a common job, such as classifying or summarizing.

## Content

**Document**:
A piece of content held in memory and ready to use, together with its metadata. It may come from a file, a generator or a live stream, so it is not a file. Its source can be empty.
_Avoid_: File, asset, blob

**Text Document**:
A Document whose content is text, for example text extracted from a PDF or a web page.

**Image Document**:
A Document whose content is an image, loaded from disk or produced by an image generator.

**Audio Document**:
A Document whose content is audio already decoded into samples, not the bytes of an audio file.

**Video Document**:
A Document whose content is a video, kept as it was encoded, together with the facts of its streams.

**Chunk**:
A contiguous slice of content, usually cut from a Document, small enough to embed, search or process on its own.
_Avoid_: Segment, piece, passage

**Media Type**:
The kind of content a Document or Chunk carries: text, image, audio or video.
_Avoid_: Modality, content type

**Transcript**:
The text recognized in a stretch of speech, as an ordered list of Utterances.

**Utterance**:
One continuous stretch of speech by one speaker, with its text and timing.
_Avoid_: Segment, line

## Agents

**Agent**:
An LLM given a system prompt, Tools and optional Guardrails, which decides what to say and which Tools to call.
_Avoid_: Bot, assistant, skill

**Think-Act Loop**:
The cycle in which an Agent asks the LLM for an answer, runs the Tool Calls it requested, and repeats until the LLM answers without calling a Tool or the iteration limit is hit.
_Avoid_: ReAct loop, agent loop, reasoning loop

**Conversation**:
A sequence of turns with one Agent whose history persists between turns.
_Avoid_: Session, thread, chat

**Tool**:
A named capability an Agent can call, with a declared input shape.
_Avoid_: Function, action, plugin, skill

**Tool Call**:
One request by the model to run a Tool with given arguments, identified by a call id.
_Avoid_: Function call, invocation

**Remote Tool**:
A Tool served by an external MCP server rather than implemented in-process.
_Avoid_: MCP function, external tool

**Guardrail**:
A check that runs before or after the model answers and can replace the answer or reject it.
_Avoid_: Filter, validator, moderation

**Context Strategy**:
The rule that decides which part of a Conversation's history the model still sees.
_Avoid_: Memory, window policy

**Corrective Retry**:
The single extra attempt the Agent gets after it produced a malformed Tool Call or an answer that does not match the required shape, with the error fed back to the model.
_Avoid_: Retry (unqualified), repair, self-heal

**Idempotency Key**:
The identifier a Tool with a side effect uses to recognize that it was already asked to perform this exact action.
_Avoid_: Request id, dedup key

**Partial History**:
What a Conversation keeps after a failed turn: the completed Tool Calls and their results, without the unanswered request.

## Workflows

**Workflow**:
A graph of Nodes that runs one or more Agents and Tools over a shared, typed State, with conditional branches and the ability to resume.
_Avoid_: Pipeline, DAG, orchestration, graph (for the whole run)

**Node**:
One step of a Workflow that takes the State and returns an updated State.
_Avoid_: Task, stage, vertex

**State**:
The typed data a Workflow carries from Node to Node.
_Avoid_: Context, blackboard

**Router**:
The decision on a conditional edge that picks which Node runs next from the current State.
_Avoid_: Condition, switch

**Handoff**:
Passing control of a Workflow's shared messages from one Agent to another.
_Avoid_: Delegation, transfer

**Run**:
One execution of a Workflow, identified by a run id that must be unique.
_Avoid_: Execution, job, instance

**Checkpoint**:
The saved record of a Run after a Node, naming the next Node, the State and the steps taken so far, from which the Run can resume.
_Avoid_: Snapshot, save point

## Evals

**Eval Case**:
One input to test, with an optional expected result, tags and a pass threshold.
_Avoid_: Test case, sample, example

**Eval Target**:
The system under evaluation, seen as something that turns an Eval Case into an output.
_Avoid_: Subject, SUT, model under test

**Scorer**:
A rule that grades an Eval Target's output on an Eval Case.
_Avoid_: Metric, judge, grader

**Eval Report**:
The outcome of running Eval Cases against an Eval Target, case by case and in aggregate.
_Avoid_: Results, scorecard

## Observability

**Observation**:
One unit of work the platform recorded, described in the platform's own terms rather than a tracing backend's.
_Avoid_: Event, log entry, trace (for a single unit)

**Fact**:
One named piece of an Observation that telemetry carries.
_Avoid_: Attribute, tag, label

**Content Capture**:
The opt-in recording of prompts, answers and Tool payloads in telemetry, on top of their size and digest, which are always recorded.
_Avoid_: Payload logging, prompt logging
