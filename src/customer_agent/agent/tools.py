"""Agent tools. The retrieval recorder lets the eval runner capture everything the
agent retrieved (across multiple tool calls) without touching the agent loop."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from agents import function_tool

from customer_agent.retrieval.pipeline import RetrievalResult, get_pipeline

_recorder: ContextVar[list[RetrievalResult] | None] = ContextVar("retrieval_recorder", default=None)


@contextmanager
def record_retrievals() -> Iterator[list[RetrievalResult]]:
    """Within this context, every search_knowledge_base call appends its RetrievalResult.

    Works across asyncio task boundaries: child tasks inherit the context, and the
    recorded list object is shared by reference.
    """
    results: list[RetrievalResult] = []
    token = _recorder.set(results)
    try:
        yield results
    finally:
        _recorder.reset(token)


@function_tool
def search_knowledge_base(query: str) -> str:
    """Search the Wix Help Center knowledge base for articles relevant to the query.

    Use a focused query describing what the user needs (feature names, error
    messages, task descriptions). Can be called multiple times with refined
    queries if the first results don't answer the question.

    Args:
        query: A search query describing the information needed.
    """
    pipeline = get_pipeline()
    result = pipeline.search(query)
    recorded = _recorder.get()
    if recorded is not None:
        recorded.append(result)
    return pipeline.format_for_agent(result)
