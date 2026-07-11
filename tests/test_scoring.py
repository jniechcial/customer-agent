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


def test_summarize_empty_records():
    summary = summarize(Path("runs/x.jsonl"), [], {})
    assert summary["n"] == 0
    assert summary["avg_tool_calls"] == 0
    assert summary["metrics"] == {}
