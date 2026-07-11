"""Generation phase: run the agent over a split, persist per-question JSONL artifacts.

Generation is the expensive part; artifacts in runs/ let us re-score (new judge
prompt, new K) without re-running the agent. One JSON object per line:
question, gold answer/articles, agent answer, ranked retrieved article ids,
per-tool-call transcript.
"""

import asyncio
import json
import time
from pathlib import Path

from agents import Runner

from customer_agent.agent.agent import build_agent
from customer_agent.agent.tools import record_retrievals
from customer_agent.config import get_settings
from customer_agent.data.splits import get_split
from customer_agent.evaluation.user_simulator import get_default_simulator

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


async def _run_one(agent, simulator, row: dict, index: int, semaphore: asyncio.Semaphore) -> dict:
    settings = get_settings()
    async with semaphore:
        with record_retrievals() as retrievals:
            message = simulator.first_message(row)
            result = await Runner.run(agent, message)
            # max_turns > 1 + an LLM simulator plugs in here later:
            # loop turns via simulator.next_message(...) feeding result.to_input_list().
        per_call = [
            {"query": r.query, "article_ids": r.ranked_article_ids} for r in retrievals
        ]
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
    records = await asyncio.gather(
        *[_run_one(agent, simulator, row, i, semaphore) for i, row in enumerate(rows)]
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
