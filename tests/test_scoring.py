from pathlib import Path
from types import SimpleNamespace

import pytest

from customer_agent.evaluation.scoring import (
    build_test_cases,
    extract_metric_scores,
    summarize,
)

RECORDS = [
    {
        "question": "q1",
        "actual_answer": "a1",
        "expected_answer": "e1",
        "gold_article_ids": ["A"],
        "retrieved_article_ids": ["A", "B"],
        "tool_calls": [{"query": "x", "article_ids": ["A", "B"]}],
        "agent_model": "gpt-test",
        "usage": {
            "requests": 2,
            "input_tokens": 1000,
            "cached_input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 1200,
        },
        "cost_usd": 0.01,
        "latency_seconds": 4.0,
    },
    {
        "question": "q2",
        "actual_answer": "a2",
        "expected_answer": "e2",
        "gold_article_ids": ["C", "D"],
        "retrieved_article_ids": [],
        "tool_calls": [
            {"query": "y", "article_ids": []},
            {"query": "z", "article_ids": []},
            {"query": "w", "article_ids": []},
        ],
        "agent_model": "gpt-test",
        "usage": {
            "requests": 4,
            "input_tokens": 3000,
            "cached_input_tokens": 500,
            "output_tokens": 400,
            "total_tokens": 3400,
        },
        "cost_usd": 0.03,
        "latency_seconds": 10.0,
    },
]


def test_build_test_cases_maps_fields():
    cases = build_test_cases(RECORDS)
    assert len(cases) == 2
    tc = cases[0]
    assert tc.input == "q1"
    assert tc.actual_output == "a1"
    assert tc.expected_output == "e1"
    assert tc.context == ["A"]                 # gold ids
    assert tc.retrieval_context == ["A", "B"]  # ranked retrieved ids


def fake_result(scores_per_test: list[dict[str, float | None]]):
    return SimpleNamespace(
        test_results=[
            SimpleNamespace(
                metrics_data=[
                    SimpleNamespace(name=name, score=score) for name, score in scores.items()
                ]
            )
            for scores in scores_per_test
        ]
    )


def test_extract_metric_scores_groups_by_metric():
    result = fake_result([
        {"recall@5": 1.0, "map@5": 0.5},
        {"recall@5": 0.0, "map@5": 0.25},
    ])
    assert extract_metric_scores(result) == {"recall@5": [1.0, 0.0], "map@5": [0.5, 0.25]}


def test_extract_metric_scores_skips_none_and_missing():
    result = fake_result([{"recall@5": 1.0, "judge": None}])
    result.test_results.append(SimpleNamespace(metrics_data=None))  # errored test case
    assert extract_metric_scores(result) == {"recall@5": [1.0]}


def test_summarize_aggregates():
    per_metric = {"recall@5": [1.0, 0.0], "map@5": [0.5, 0.25]}
    summary = summarize(Path("runs/x.jsonl"), RECORDS, per_metric)
    assert summary["n"] == 2
    assert summary["agent_model"] == "gpt-test"
    assert summary["avg_tool_calls"] == pytest.approx(2.0)  # (1 + 3) / 2
    assert summary["metrics"]["recall@5"] == {"mean": 0.5, "n": 2}
    assert summary["metrics"]["map@5"] == {"mean": 0.375, "n": 2}
    assert summary["usage"] == {
        "total_input_tokens": 4000,
        "total_cached_input_tokens": 600,
        "total_output_tokens": 600,
        "avg_total_tokens": 2300,
    }
    assert summary["cost_usd"] == {"total": pytest.approx(0.04), "avg_per_answer": pytest.approx(0.02)}
    assert summary["latency_seconds"] == {"avg": 7.0, "max": 10.0}


def test_summarize_tolerates_records_without_usage_fields():
    """Artifacts generated before usage/cost/latency were recorded still summarize."""
    old_records = [{k: v for k, v in r.items() if k not in {"usage", "cost_usd", "latency_seconds"}}
                   for r in RECORDS]
    summary = summarize(Path("runs/x.jsonl"), old_records, {})
    assert summary["n"] == 2
    assert summary["usage"] is None
    assert summary["cost_usd"] is None
    assert summary["latency_seconds"] is None


def test_summarize_empty_records():
    summary = summarize(Path("runs/x.jsonl"), [], {})
    assert summary["n"] == 0
    assert summary["avg_tool_calls"] == 0
    assert summary["metrics"] == {}
    assert summary["usage"] is None
    assert summary["cost_usd"] is None
    assert summary["latency_seconds"] is None
