"""The retrieval recorder is what makes eval metrics trustworthy: it must capture
every tool call made during a question, across asyncio task boundaries, and never
leak between concurrent questions."""

import asyncio

from customer_agent.agent.tools import _recorder, record_retrievals
from customer_agent.retrieval.pipeline import RetrievalResult
from tests.conftest import make_chunk


def result_for(article_id: str) -> RetrievalResult:
    return RetrievalResult(query=f"q-{article_id}", ranked_chunks=[make_chunk(article_id)])


def simulate_tool_call(article_id: str) -> None:
    """What search_knowledge_base does with its RetrievalResult."""
    recorded = _recorder.get()
    if recorded is not None:
        recorded.append(result_for(article_id))


def test_records_within_context():
    with record_retrievals() as retrievals:
        simulate_tool_call("A")
        simulate_tool_call("B")
    assert [r.query for r in retrievals] == ["q-A", "q-B"]


def test_no_context_is_noop():
    simulate_tool_call("A")  # must not raise
    assert _recorder.get() is None


def test_context_resets_after_exit():
    with record_retrievals():
        simulate_tool_call("A")
    assert _recorder.get() is None


def test_records_across_asyncio_task_boundary():
    """The Agents SDK may run tools in child tasks; those inherit the context and
    mutate the shared list."""

    async def main():
        with record_retrievals() as retrievals:
            await asyncio.create_task(asyncio.to_thread(simulate_tool_call, "A"))
            await asyncio.create_task(_async_tool("B"))
        return retrievals

    async def _async_tool(article_id):
        simulate_tool_call(article_id)

    retrievals = asyncio.run(main())
    assert [r.query for r in retrievals] == ["q-A", "q-B"]


def test_concurrent_contexts_do_not_leak():
    """Two questions evaluated concurrently must record independently."""

    async def one_question(article_id: str, started: asyncio.Event, proceed: asyncio.Event):
        with record_retrievals() as retrievals:
            started.set()
            await proceed.wait()  # force interleaving with the other question
            simulate_tool_call(article_id)
        return retrievals

    async def main():
        a_started, b_started = asyncio.Event(), asyncio.Event()
        proceed = asyncio.Event()
        task_a = asyncio.create_task(one_question("A", a_started, proceed))
        task_b = asyncio.create_task(one_question("B", b_started, proceed))
        await a_started.wait()
        await b_started.wait()
        proceed.set()
        return await task_a, await task_b

    retrievals_a, retrievals_b = asyncio.run(main())
    assert [r.query for r in retrievals_a] == ["q-A"]
    assert [r.query for r in retrievals_b] == ["q-B"]
