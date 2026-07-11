"""Scoring phase: turn run artifacts into deepeval test cases, evaluate, aggregate.

Kept separate from the CLI script so the pieces are importable and testable.
Only score() talks to an LLM (the GEval judge); everything else is pure.
"""

import statistics
from pathlib import Path

from deepeval.test_case import LLMTestCase

from customer_agent.config import get_settings


def build_test_cases(records: list[dict]) -> list[LLMTestCase]:
    """Convention: context = GOLD article ids; retrieval_context = ranked retrieved ids."""
    return [
        LLMTestCase(
            input=r["question"],
            actual_output=r["actual_answer"],
            expected_output=r["expected_answer"],
            context=r["gold_article_ids"],
            retrieval_context=r["retrieved_article_ids"],
        )
        for r in records
    ]


def extract_metric_scores(evaluation_result) -> dict[str, list[float]]:
    """Per-metric score lists out of a deepeval EvaluationResult(-like) object."""
    per_metric: dict[str, list[float]] = {}
    for test_result in evaluation_result.test_results:
        for metric_data in test_result.metrics_data or []:
            if metric_data.score is not None:
                per_metric.setdefault(metric_data.name, []).append(metric_data.score)
    return per_metric


def summarize(artifact: Path, records: list[dict], per_metric: dict[str, list[float]]) -> dict:
    # .get()-based: artifacts generated before usage/cost/latency were recorded
    # still summarize (those fields come out None/absent).
    usages = [r["usage"] for r in records if r.get("usage")]
    costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    latencies = [r["latency_seconds"] for r in records if r.get("latency_seconds") is not None]
    return {
        "artifact": str(artifact),
        "n": len(records),
        "agent_model": records[0].get("agent_model") if records else None,
        "avg_tool_calls": (
            statistics.mean(len(r["tool_calls"]) for r in records) if records else 0
        ),
        "usage": {
            "total_input_tokens": sum(u["input_tokens"] for u in usages),
            "total_cached_input_tokens": sum(u["cached_input_tokens"] for u in usages),
            "total_output_tokens": sum(u["output_tokens"] for u in usages),
            "avg_total_tokens": statistics.mean(u["total_tokens"] for u in usages),
        } if usages else None,
        "cost_usd": {
            "total": sum(costs),
            "avg_per_answer": statistics.mean(costs),
        } if costs else None,
        "latency_seconds": {
            "avg": statistics.mean(latencies),
            "max": max(latencies),
        } if latencies else None,
        "metrics": {
            name: {"mean": statistics.mean(scores), "n": len(scores)}
            for name, scores in sorted(per_metric.items())
        },
    }


def score(artifact: Path) -> dict:
    """Full scoring pass: deepeval evaluate (judge LLM + deterministic metrics) -> summary."""
    from deepeval import evaluate

    from customer_agent.evaluation.answer_metrics import make_correctness_metric
    from customer_agent.evaluation.retrieval_metrics import retrieval_metrics
    from customer_agent.evaluation.runner import load_run

    settings = get_settings()
    records = load_run(artifact)
    metrics = [make_correctness_metric(), *retrieval_metrics(settings.metric_ks)]
    result = evaluate(test_cases=build_test_cases(records), metrics=metrics)
    return summarize(artifact, records, extract_metric_scores(result))
