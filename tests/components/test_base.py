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
