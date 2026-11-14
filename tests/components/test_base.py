import asyncio

import pytest

from agent_platform.components.base import Component


class _AddOne(Component[int, int]):
    async def arun(self, input: int) -> int:
        return input + 1


class _ToStr(Component[int, str]):
    async def arun(self, input: int) -> str:
        return f"value={input}"


class _Fails(Component[int, int]):
    async def arun(self, input: int) -> int:
        raise ValueError("boom")


def test_run_works_outside_running_loop():
    component = _AddOne()
    assert component.run(1) == 2


def test_run_raises_inside_running_loop():
    component = _AddOne()

    async def _inner():
        component.run(1)

    with pytest.raises(RuntimeError, match="running event loop"):
        asyncio.run(_inner())


@pytest.mark.asyncio
async def test_rshift_chains_two_components():
    chain = _AddOne() >> _ToStr()
    assert await chain.arun(1) == "value=2"


@pytest.mark.asyncio
async def test_rshift_chains_three_components():
    chain = _AddOne() >> _AddOne() >> _ToStr()
    assert await chain.arun(1) == "value=3"


@pytest.mark.asyncio
async def test_rshift_propagates_error_from_first_stage():
    chain = _Fails() >> _ToStr()
    with pytest.raises(ValueError, match="boom"):
        await chain.arun(1)


@pytest.mark.asyncio
async def test_rshift_propagates_error_from_second_stage():
    chain = _AddOne() >> _Fails()
    with pytest.raises(ValueError, match="boom"):
        await chain.arun(1)


class _NamedOperation(Component[int, int]):
    operation_name = "embeddings"

    async def arun(self, input: int) -> int:
        return input


class _InheritsArun(_AddOne):
    pass


class TestComponentTracing:
    @pytest.mark.asyncio
    async def test_a_component_run_is_a_span_named_after_its_class(
        self, recorded_spans
    ):
        await _AddOne().arun(1)

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "add_one"
        assert span.attributes["operation.name"] == "add_one"

    @pytest.mark.asyncio
    async def test_operation_name_overrides_the_class_derived_default(
        self, recorded_spans
    ):
        await _NamedOperation().arun(1)

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "embeddings"

    @pytest.mark.asyncio
    async def test_an_inherited_arun_is_traced_once(self, recorded_spans):
        await _InheritsArun().arun(1)

        assert [s.name for s in recorded_spans.get_finished_spans()] == ["add_one"]

    @pytest.mark.asyncio
    async def test_a_chain_traces_every_step(self, recorded_spans):
        await (_AddOne() >> _ToStr()).arun(1)

        assert [s.name for s in recorded_spans.get_finished_spans()] == [
            "add_one",
            "to_str",
            "chain",
        ]

    @pytest.mark.asyncio
    async def test_a_failing_component_still_closes_its_span(self, recorded_spans):
        with pytest.raises(ValueError):
            await _Fails().arun(1)

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "fails"

    @pytest.mark.asyncio
    async def test_the_embedder_reports_the_embeddings_operation(self):
        from agent_platform.components.retrieval.embedder.component import Embedder

        assert Embedder.operation_name == "embeddings"
