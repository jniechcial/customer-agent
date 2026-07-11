"""Guards the agent-facing tool contract: name, schema, and invocation behavior.
The pipeline is faked — no OpenAI or Weaviate."""

import asyncio
import json

from agents import FunctionTool
from agents.tool_context import ToolContext
from agents.usage import Usage

import customer_agent.agent.tools as tools_module
from customer_agent.agent.tools import record_retrievals, search_knowledge_base
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
