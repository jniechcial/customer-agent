"""Guards the agent-facing tool contract: name, schema, and invocation behavior.
The pipeline is faked — no OpenAI or Weaviate."""

import asyncio
import json
from types import SimpleNamespace

from agents import FunctionTool
from agents.tool_context import ToolContext
from agents.usage import Usage

import customer_agent.agent.tools as tools_module
from customer_agent.agent.tools import record_retrievals, search_budget, search_knowledge_base
from customer_agent.retrieval.pipeline import RetrievalResult
from tests.conftest import make_chunk


def test_tool_is_a_function_tool_named_correctly():
    assert isinstance(search_knowledge_base, FunctionTool)
    assert search_knowledge_base.name == "search_knowledge_base"
    schema = search_knowledge_base.params_json_schema
    assert list(schema["properties"].keys()) == ["query"]
    assert "search" in (search_knowledge_base.description or "").lower()


class FakePipeline:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return RetrievalResult(query=query, ranked_chunks=[make_chunk("A")])

    def format_for_agent(self, result):
        return f"formatted:{result.query}"


def invoke_tool(query: str) -> str:
    ctx = ToolContext(
        context=None,
        usage=Usage(),
        tool_name=search_knowledge_base.name,
        tool_call_id="call_1",
        tool_arguments=json.dumps({"query": query}),
    )
    return asyncio.run(
        search_knowledge_base.on_invoke_tool(ctx, json.dumps({"query": query}))
    )


def test_tool_invocation_returns_formatted_output_and_records(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(tools_module, "get_pipeline", lambda: fake)

    with record_retrievals() as retrievals:
        output = invoke_tool("how to connect domain")

    assert output == "formatted:how to connect domain"
    assert fake.queries == ["how to connect domain"]
    assert len(retrievals) == 1
    assert retrievals[0].ranked_article_ids == ["A"]


def test_tool_invocation_without_recorder_still_works(monkeypatch):
    monkeypatch.setattr(tools_module, "get_pipeline", lambda: FakePipeline())
    assert invoke_tool("q") == "formatted:q"


def test_call_beyond_budget_errors_without_searching_or_recording(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(tools_module, "get_pipeline", lambda: fake)

    with record_retrievals() as retrievals, search_budget(limit=2):
        assert invoke_tool("q1") == "formatted:q1"
        assert invoke_tool("q2") == "formatted:q2"
        blocked = invoke_tool("q3")

    assert blocked.startswith("Error: search budget exhausted (2 searches maximum)")
    assert fake.queries == ["q1", "q2"]  # third call never reached the pipeline
    assert len(retrievals) == 2  # nor the recorder / eval metrics


def test_budget_defaults_to_settings_max_searches(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(tools_module, "get_pipeline", lambda: fake)
    monkeypatch.setattr(
        tools_module, "get_settings", lambda: SimpleNamespace(max_searches=1)
    )

    with search_budget():
        assert invoke_tool("q1") == "formatted:q1"
        assert invoke_tool("q2").startswith("Error: search budget exhausted")
    assert fake.queries == ["q1"]


def test_budget_resets_between_contexts(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(tools_module, "get_pipeline", lambda: fake)

    with search_budget(limit=1):
        assert invoke_tool("q1") == "formatted:q1"
        assert invoke_tool("q2").startswith("Error:")
    with search_budget(limit=1):
        assert invoke_tool("q3") == "formatted:q3"
    assert fake.queries == ["q1", "q3"]


def test_budget_counts_across_asyncio_task_boundary(monkeypatch):
    """The Agents SDK may run tools in child tasks; they must share one counter
    (the SearchBudget object is inherited by reference, like the recorder list)."""
    fake = FakePipeline()
    monkeypatch.setattr(tools_module, "get_pipeline", lambda: fake)

    async def call_in_task(query):
        return await asyncio.create_task(
            search_knowledge_base.on_invoke_tool(
                ToolContext(
                    context=None,
                    usage=Usage(),
                    tool_name=search_knowledge_base.name,
                    tool_call_id="call_1",
                    tool_arguments=json.dumps({"query": query}),
                ),
                json.dumps({"query": query}),
            )
        )

    async def main():
        with search_budget(limit=2):
            return [await call_in_task(q) for q in ("q1", "q2", "q3")]

    outputs = asyncio.run(main())
    assert outputs[:2] == ["formatted:q1", "formatted:q2"]
    assert outputs[2].startswith("Error: search budget exhausted")
    assert fake.queries == ["q1", "q2"]


def test_no_budget_context_means_uncapped(monkeypatch):
    """Entry points set the budget; bare tool use (e.g. these tests above,
    ad-hoc scripts) stays uncapped rather than sharing a global counter."""
    fake = FakePipeline()
    monkeypatch.setattr(tools_module, "get_pipeline", lambda: fake)
    for i in range(5):
        assert invoke_tool(f"q{i}") == f"formatted:q{i}"
    assert len(fake.queries) == 5
