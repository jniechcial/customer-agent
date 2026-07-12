"""Generation phase: run the agent over a split, persist per-question JSONL artifacts.

Generation is the expensive part; artifacts in runs/ let us re-score (new judge
prompt, new K) without re-running the agent. One JSON object per line:
question, gold answer/articles, agent answer, ranked retrieved article ids,
per-tool-call transcript, and the question's OTel trace/span ids (for attaching
score annotations to the Phoenix trace at scoring time).
"""

import asyncio
import json
import time
from pathlib import Path

from agents import Runner
from opentelemetry import trace as otel_trace

from customer_agent.agent.agent import build_agent
from customer_agent.agent.tools import record_retrievals
from customer_agent.config import get_settings
from customer_agent.data.splits import get_split
from customer_agent.evaluation.user_simulator import get_default_simulator
from customer_agent.pricing import answer_cost_usd
from customer_agent.retrieval.pipeline import close_pipeline

RUNS_DIR = Path("runs")


def merge_ranked_article_ids(per_call_rankings: list[list[str]]) -> list[str]:
    """Concatenate per-call article rankings in call order, dedupe to first occurrence."""
    seen: set[str] = set()
    merged: list[str] = []
    for ranking in per_call_rankings:
        for article_id in ranking:
            if article_id not in seen:
                seen.add(article_id)
                merged.append(article_id)
    return merged


async def _run_one(
    agent, simulator, row: dict, index: int, semaphore: asyncio.Semaphore, progress: dict
) -> dict:
    settings = get_settings()
    tracer = otel_trace.get_tracer("customer_agent.evaluation")
    async with semaphore:
        message = simulator.first_message(row)
        preview = " ".join(row["question"].split())[:70]
        print(f"[q{index}] start: {preview}", flush=True)
        try:
            record = await _run_agent(agent, tracer, settings, message, row, index)
        except Exception as exc:
            # One flaky question must not kill the whole run; the error is persisted
            # in the artifact and the question is skipped at scoring time.
            progress["done"] += 1
            print(
                f"[{progress['done']}/{progress['total']} done] q{index} FAILED: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return {
                "index": index,
                "question": row["question"],
                "expected_answer": row["answer"],
                "gold_article_ids": list(row["article_ids"]),
                "error": f"{type(exc).__name__}: {exc}",
            }
        progress["done"] += 1
        print(
            f"[{progress['done']}/{progress['total']} done] q{index} ok "
            f"({record['latency_seconds']:.1f}s, {len(record['tool_calls'])} tool calls)",
            flush=True,
        )
        return record


async def _run_agent(agent, tracer, settings, message: str, row: dict, index: int) -> dict:
    # Explicit per-question root span: the agent's whole trace nests under it, and
    # its ids are persisted in the artifact so scoring can attach judge annotations
    # to the right Phoenix trace (also on later --rescore).
    with record_retrievals() as retrievals, tracer.start_as_current_span(
        f"question-{index}",
        attributes={"openinference.span.kind": "CHAIN", "input.value": message},
    ) as question_span:
        start = time.perf_counter()
        result = await Runner.run(agent, message)
        latency_seconds = time.perf_counter() - start
        question_span.set_attribute("output.value", str(result.final_output))
        # max_turns > 1 + an LLM simulator plugs in here later:
        # loop turns via simulator.next_message(...) feeding result.to_input_list().
    span_context = question_span.get_span_context()
    per_call = [
        {"query": r.query, "article_ids": r.ranked_article_ids} for r in retrievals
    ]
    usage = result.context_wrapper.usage  # aggregated across all LLM requests in the run
    cached_input_tokens = usage.input_tokens_details.cached_tokens
    return {
        "index": index,
        "question": row["question"],
        "expected_answer": row["answer"],
        "gold_article_ids": list(row["article_ids"]),
        "actual_answer": str(result.final_output),
        "retrieved_article_ids": merge_ranked_article_ids(
            [c["article_ids"] for c in per_call]
        ),
        "tool_calls": per_call,
        "agent_model": settings.agent_model,
        "reranker": settings.reranker_id,
        "usage": {
            "requests": usage.requests,
            "input_tokens": usage.input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        },
        "cost_usd": answer_cost_usd(
            settings.agent_model, usage.input_tokens, cached_input_tokens, usage.output_tokens
        ),
        "latency_seconds": round(latency_seconds, 3),
        # Invalid context means tracing wasn't set up (e.g. tests) — record nulls.
        "otel_trace_id": (
            format(span_context.trace_id, "032x") if span_context.is_valid else None
        ),
        "otel_span_id": (
            format(span_context.span_id, "016x") if span_context.is_valid else None
        ),
    }


async def generate_run(
    split_name: str,
    subset: str | None = None,
    limit: int | None = None,
    run_id: str | None = None,
) -> Path:
    settings = get_settings()
    run_id = run_id or f"{split_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    rows = get_split(split_name, subset)
    if limit:
        rows = rows.select(range(min(limit, len(rows))))

    agent = build_agent()
    simulator = get_default_simulator()
    semaphore = asyncio.Semaphore(settings.eval_concurrency)
    progress = {"done": 0, "total": len(rows)}
    try:
        records = await asyncio.gather(
            *[_run_one(agent, simulator, row, i, semaphore, progress) for i, row in enumerate(rows)]
        )
    finally:
        close_pipeline()

    failed = [r for r in records if r.get("error")]
    if failed:
        print(
            f"{len(failed)}/{len(records)} questions failed "
            f"(indices {[r['index'] for r in failed]}); errors recorded in the artifact",
            flush=True,
        )

    RUNS_DIR.mkdir(exist_ok=True)
    artifact = RUNS_DIR / f"{run_id}.jsonl"
    with artifact.open("w") as f:
        for record in sorted(records, key=lambda r: r["index"]):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return artifact


def load_run(artifact: Path) -> list[dict]:
    with artifact.open() as f:
        return [json.loads(line) for line in f if line.strip()]
