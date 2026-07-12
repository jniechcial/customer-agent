"""Agent tools. The retrieval recorder lets the eval runner capture everything the
agent retrieved (across multiple tool calls) without touching the agent loop; the
search budget enforces the max_searches cap the same way."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from agents import function_tool

from customer_agent.config import get_settings
from customer_agent.retrieval.pipeline import RetrievalResult, get_pipeline

_recorder: ContextVar[list[RetrievalResult] | None] = ContextVar("retrieval_recorder", default=None)
_budget: ContextVar["SearchBudget | None"] = ContextVar("search_budget", default=None)


@dataclass
class SearchBudget:
    """Mutable so child asyncio tasks (which inherit the context by reference)
    increment the same counter — same sharing mechanism as the recorder list."""

    limit: int
    used: int = 0


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


@contextmanager
def search_budget(limit: int | None = None) -> Iterator[SearchBudget]:
    """Within this context, search_knowledge_base calls beyond `limit` (default:
    settings.max_searches) skip retrieval and return an error to the model.

    Outside any budget context the tool is uncapped; both entry points (eval runner
    per question, chat per user turn) are expected to set one.
    """
    budget = SearchBudget(limit=limit if limit is not None else get_settings().max_searches)
    token = _budget.set(budget)
    try:
        yield budget
    finally:
        _budget.reset(token)


@function_tool
def search_knowledge_base(query: str) -> str:
    """Search the Wix Help Center knowledge base for articles relevant to the query.

    Use a focused query describing what the user needs (feature names, error
    messages, task descriptions). At most 2 calls are allowed per question: one
    initial search plus one refined query if the first results don't cover the
    question. Further calls fail without returning results.

    Args:
        query: A search query describing the information needed.
    """
    budget = _budget.get()
    if budget is not None:
        if budget.used >= budget.limit:
            return (
                f"Error: search budget exhausted ({budget.limit} searches maximum). "
                "No further knowledge base retrieval is available — answer the user "
                "now using only the results already retrieved."
            )
        budget.used += 1
    pipeline = get_pipeline()
    result = pipeline.search(query)
    recorded = _recorder.get()
    if recorded is not None:
        recorded.append(result)
    return pipeline.format_for_agent(result)
