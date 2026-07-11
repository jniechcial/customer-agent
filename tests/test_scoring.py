from pathlib import Path
from types import SimpleNamespace

import pytest

from customer_agent.evaluation.scoring import (
    build_annotations,
    build_test_cases,
    extract_case_results,
    per_metric_scores,
    push_annotations,
    summarize,
    write_scores,
)

RECORDS = [
    {
        "index": 0,
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
        "otel_trace_id": "a" * 32,
        "otel_span_id": "b" * 16,
    },
    {
        "index": 1,
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
        # no otel ids: artifact predates trace-id recording
    },
]


def test_build_test_cases_maps_fields():
    cases = build_test_cases(RECORDS)
    assert len(cases) == 2
    tc = cases[0]
    assert tc.name == "q0"  # join key back to the record
    assert tc.input == "q1"
    assert tc.actual_output == "a1"
    assert tc.expected_output == "e1"
    assert tc.context == ["A"]                 # gold ids
    assert tc.retrieval_context == ["A", "B"]  # ranked retrieved ids


def fake_result(metrics_per_test: dict[str, list[dict]]):
    """metrics_per_test: case name -> [{name, score, reason?, success?, evaluation_model?}]"""
    return SimpleNamespace(
        test_results=[
            SimpleNamespace(
                name=case_name,
                metrics_data=[
                    SimpleNamespace(
                        name=m["name"],
                        score=m["score"],
                        reason=m.get("reason"),
                        success=m.get("success"),
                        evaluation_model=m.get("evaluation_model"),
                    )
                    for m in metrics
                ],
            )
            for case_name, metrics in metrics_per_test.items()
        ]
    )


CASE_RESULTS = {
    "q0": {
        "AnswerCorrectness": {
            "score": 0.9,
            "success": True,
            "reason": "matches the expected steps",
            "evaluation_model": "claude-test",
        },
        "recall@5": {"score": 1.0, "success": True, "reason": "recall@5=1.0",
                     "evaluation_model": None},
        "mrr@5": {"score": 1.0, "success": True, "reason": "mrr@5=1.0",
                  "evaluation_model": None},
    },
    "q1": {
        "AnswerCorrectness": {
            "score": 0.2,
            "success": False,
            "reason": "contradicts the expected answer",
            "evaluation_model": "claude-test",
        },
        "recall@5": {"score": 0.0, "success": True, "reason": "recall@5=0.0",
                     "evaluation_model": None},
    },
}


def test_extract_case_results_keeps_reasons():
    result = fake_result({
        name: [{"name": mname, **mdata} for mname, mdata in metrics.items()]
        for name, metrics in CASE_RESULTS.items()
    })
    assert extract_case_results(result) == CASE_RESULTS


def test_extract_case_results_skips_none_scores():
    result = fake_result({"q0": [
        {"name": "recall@5", "score": 1.0},
        {"name": "judge", "score": None},  # errored metric
    ]})
    result.test_results.append(SimpleNamespace(name="q1", metrics_data=None))
    extracted = extract_case_results(result)
    assert list(extracted["q0"]) == ["recall@5"]
    assert extracted["q1"] == {}


def test_per_metric_scores_pivots():
    assert per_metric_scores(CASE_RESULTS) == {
        "AnswerCorrectness": [0.9, 0.2],
        "recall@5": [1.0, 0.0],
        "mrr@5": [1.0],
    }


def test_write_scores_one_row_per_question(tmp_path):
    import json

    artifact = tmp_path / "run.jsonl"
    path = write_scores(artifact, RECORDS, CASE_RESULTS)
    assert path == tmp_path / "run.scores.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["index"] for r in rows] == [0, 1]
    assert rows[0]["question"] == "q1"
    assert rows[0]["otel_span_id"] == "b" * 16
    assert rows[0]["metrics"]["AnswerCorrectness"]["reason"] == "matches the expected steps"
    assert rows[1]["otel_span_id"] is None
    assert rows[1]["metrics"]["AnswerCorrectness"]["score"] == 0.2


def test_build_annotations_llm_vs_code_and_skips_missing_span():
    annotations = build_annotations(RECORDS, CASE_RESULTS)
    # Record 1 has no span id; record 0's mrr@5 is run-level-only -> 2 annotations.
    assert len(annotations) == 2
    assert not any(a["name"] == "mrr@5" for a in annotations)
    correctness = next(a for a in annotations if a["name"] == "AnswerCorrectness")
    assert correctness["span_id"] == "b" * 16
    assert correctness["annotator_kind"] == "LLM"
    assert correctness["result"] == {
        "score": 0.9,
        "label": "pass",
        "explanation": "matches the expected steps",
    }
    recall = next(a for a in annotations if a["name"] == "recall@5")
    assert recall["annotator_kind"] == "CODE"
    assert recall["result"]["label"] is None


def test_build_annotations_failed_judgement_labeled_fail():
    record = dict(RECORDS[1], otel_span_id="c" * 16)  # index=1 -> joins on "q1"
    annotations = build_annotations([record], {"q1": CASE_RESULTS["q1"]})
    correctness = next(a for a in annotations if a["name"] == "AnswerCorrectness")
    assert correctness["result"]["label"] == "fail"


def test_push_annotations_posts_batch(monkeypatch):
    calls = {}

    def fake_post(url, json=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr("customer_agent.evaluation.scoring.httpx.post", fake_post)
    annotations = build_annotations(RECORDS, CASE_RESULTS)
    pushed = push_annotations(annotations, "http://localhost:6007/v1/traces")
    assert pushed == 2
    assert calls["url"] == "http://localhost:6007/v1/span_annotations"
    assert calls["json"] == {"data": annotations}


def test_push_annotations_empty_is_noop(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("should not POST")

    monkeypatch.setattr("customer_agent.evaluation.scoring.httpx.post", boom)
    assert push_annotations([], "http://localhost:6007/v1/traces") == 0


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
