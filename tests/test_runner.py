"""Generation-phase tests with the agent fully faked — no LLM, no Weaviate."""

import asyncio
import json

from types import SimpleNamespace

import pytest
from datasets import Dataset

import customer_agent.evaluation.runner as runner_module
from customer_agent.agent.tools import _recorder
from customer_agent.evaluation.runner import generate_run, load_run, merge_ranked_article_ids
from customer_agent.pricing import answer_cost_usd
from customer_agent.retrieval.pipeline import RetrievalResult
from tests.conftest import make_chunk


def test_merge_preserves_call_order_and_dedupes():
    assert merge_ranked_article_ids([["A", "B"], ["B", "C"], ["A", "D"]]) == ["A", "B", "C", "D"]


def test_merge_empty():
    assert merge_ranked_article_ids([]) == []
    assert merge_ranked_article_ids([[], []]) == []


class FakeRunResult:
    def __init__(self, final_output: str):
        self.final_output = final_output
        # Mirrors the SDK shape: result.context_wrapper.usage, cached tokens nested
        # under input_tokens_details.
        self.context_wrapper = SimpleNamespace(
            usage=SimpleNamespace(
                requests=3,
                input_tokens=1000,
                input_tokens_details=SimpleNamespace(cached_tokens=200),
                output_tokens=100,
                total_tokens=1100,
            )
        )


@pytest.fixture
def fake_agent_stack(monkeypatch, tmp_path):
    """Fake split data + fake Runner.run that simulates two tool calls per question."""
    rows = [
        {"question": f"q{i}", "answer": f"expected{i}", "article_ids": [f"gold{i}", "shared"]}
        for i in range(4)
    ]
    monkeypatch.setattr(
        runner_module, "get_split", lambda name, subset=None: Dataset.from_list(rows)
    )
    monkeypatch.setattr(runner_module, "build_agent", lambda: object())
    monkeypatch.setattr(runner_module, "RUNS_DIR", tmp_path)

    async def fake_run(agent, message):
        # Simulate the agent calling search_knowledge_base twice, second call
        # partially overlapping the first — from a child task like the SDK does.
        async def tool_call(article_ids):
            recorded = _recorder.get()
            recorded.append(
                RetrievalResult(
                    query=f"search for {message}",
                    ranked_chunks=[make_chunk(a) for a in article_ids],
                )
            )

        qid = message[1:]  # "q3" -> "3"
        await asyncio.create_task(tool_call([f"gold{qid}", "noise1"]))
        await asyncio.create_task(tool_call(["noise1", "shared"]))
        return FakeRunResult(f"answer to {message}")

    monkeypatch.setattr(runner_module.Runner, "run", staticmethod(fake_run))
    return rows


def test_generate_run_writes_complete_artifact(fake_agent_stack, tmp_path):
    artifact = asyncio.run(generate_run("validation", run_id="test-run"))
    assert artifact == tmp_path / "test-run.jsonl"

    records = load_run(artifact)
    assert len(records) == 4
    assert [r["index"] for r in records] == [0, 1, 2, 3]  # sorted despite concurrency

    r0 = records[0]
    assert r0["question"] == "q0"
    assert r0["expected_answer"] == "expected0"
    assert r0["gold_article_ids"] == ["gold0", "shared"]
    assert r0["actual_answer"] == "answer to q0"
    # Two tool calls, merged in call order, deduped:
    assert r0["retrieved_article_ids"] == ["gold0", "noise1", "shared"]
    assert len(r0["tool_calls"]) == 2
    assert r0["tool_calls"][0]["article_ids"] == ["gold0", "noise1"]
    assert r0["usage"] == {
        "requests": 3,
        "input_tokens": 1000,
        "cached_input_tokens": 200,
        "output_tokens": 100,
        "total_tokens": 1100,
    }
    # Same computation as the runner's, so exact equality (None if model unpriced).
    assert r0["cost_usd"] == answer_cost_usd(r0["agent_model"], 1000, 200, 100)
    assert r0["latency_seconds"] >= 0
    # No tracer provider in tests -> ids recorded as null, keys still present.
    assert r0["otel_trace_id"] is None
    assert r0["otel_span_id"] is None


def test_generate_run_respects_limit(fake_agent_stack):
    artifact = asyncio.run(generate_run("validation", limit=2, run_id="limited"))
    assert len(load_run(artifact)) == 2


def test_generate_run_isolates_concurrent_recordings(fake_agent_stack):
    """Each question's record must only contain its own retrievals, even though
    questions run concurrently (eval_concurrency=4 covers all fake rows)."""
    artifact = asyncio.run(generate_run("validation", run_id="isolation"))
    for r in load_run(artifact):
        qid = r["question"][1:]
        assert r["retrieved_article_ids"] == [f"gold{qid}", "noise1", "shared"]


def test_artifact_is_valid_jsonl(fake_agent_stack):
    artifact = asyncio.run(generate_run("validation", run_id="jsonl-check"))
    with artifact.open() as f:
        for line in f:
            json.loads(line)  # every line parses independently
