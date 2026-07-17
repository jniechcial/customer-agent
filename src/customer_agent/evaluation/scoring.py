"""Scoring phase: turn run artifacts into deepeval test cases, evaluate, aggregate.

Kept separate from the CLI script so the pieces are importable and testable.
Only score() and _score_scope() talk to an LLM (the GEval judges); everything
else is pure.

Records dispatch on expected_behavior (recorded per question by the runner;
absent in pre-extension artifacts and defaulted to answer_normally):
answer_normally — standard questions and gray traps — goes down the unchanged
correctness/partial/retrieval path; everything else (out-of-scope extension
rows) is scored by ScopeHandling ONLY and excluded from the correctness ladder
and the retrieval-metric means (their gold is empty — retrieval metrics would
score them 0.0 and poison the means).

Per-question visibility, two sinks fed from the same extracted results:
  - runs/<id>.scores.jsonl — every metric score + the judge's reasoning per question
  - Phoenix span annotations — scores/reasoning attached to each question's trace,
    browsable and sortable in the Phoenix UI
"""

import json
import statistics
import sys
from pathlib import Path

import httpx
from deepeval.test_case import LLMTestCase

from customer_agent.config import get_settings


def _case_name(record: dict, position: int) -> str:
    """Stable join key between records, deepeval test cases, and their results."""
    return f"q{record.get('index', position)}"


def split_by_behavior(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """(answer_normally records, scope records) — the scoring-path dispatch."""
    standard = [
        r for r in records if r.get("expected_behavior", "answer_normally") == "answer_normally"
    ]
    scope = [
        r for r in records if r.get("expected_behavior", "answer_normally") != "answer_normally"
    ]
    return standard, scope


_NO_GROUNDING_CONTEXT = (
    "No knowledge-base articles are available for grounding: the agent's searches "
    "returned nothing (or it never searched), so every specific claim beyond the "
    "expected answer is unsupported."
)


def _seen_article_ids(record: dict) -> list[str]:
    """Union of the articles the agent's tool output contained, across calls, in
    call/rank order. Old artifacts predate seen_article_ids; approximate with the
    top-k_final prefix of each call's full ranking (with chunk-granularity runs
    that can over-credit slightly — rescores of old runs are indicative only)."""
    k_final = get_settings().k_final
    seen: set[str] = set()
    ordered: list[str] = []
    for call in record.get("tool_calls") or []:
        for article_id in call.get("seen_article_ids", call["article_ids"][:k_final]):
            if article_id not in seen:
                seen.add(article_id)
                ordered.append(article_id)
    return ordered


def _grounding_context(record: dict) -> list[str]:
    """Full texts of every article the agent actually saw in tool output — gold or
    not. The judge accepts extras only when they are relevant to the question AND
    supported by these texts; content the agent never saw stays withheld (a fact
    matching a never-retrieved gold article is parametric memory getting lucky,
    not grounding). Nothing seen degrades to a sentinel — every claim beyond the
    expected answer is then ungrounded.
    """
    from customer_agent.data.wixqa import kb_by_article_id

    kb = kb_by_article_id()
    texts = [
        f"[{kb[article_id]['url']}]\n{kb[article_id]['contents']}"
        for article_id in _seen_article_ids(record)
        if article_id in kb
    ]
    return texts or [_NO_GROUNDING_CONTEXT]


def build_test_cases(records: list[dict]) -> list[LLMTestCase]:
    """Convention: context = full texts of the articles the agent saw in tool
    output (judge grounding); retrieval_context = ranked retrieved ids; the gold
    id list rides in metadata for the deterministic retrieval metrics."""
    return [
        LLMTestCase(
            name=_case_name(r, i),
            input=r["question"],
            actual_output=r["actual_answer"],
            expected_output=r["expected_answer"],
            context=_grounding_context(r),
            retrieval_context=r["retrieved_article_ids"],
            metadata={"gold_article_ids": r["gold_article_ids"]},
        )
        for i, r in enumerate(records)
    ]


def extract_case_results(evaluation_result) -> dict[str, dict[str, dict]]:
    """Case name -> metric name -> {score, success, reason, evaluation_model}.

    Keeps the judge's reasoning (metric reason), unlike a scores-only view.
    Metrics that errored (score None) are skipped.
    """
    case_results: dict[str, dict[str, dict]] = {}
    for test_result in evaluation_result.test_results:
        metrics: dict[str, dict] = {}
        for metric_data in test_result.metrics_data or []:
            if metric_data.score is not None:
                metrics[metric_data.name] = {
                    "score": metric_data.score,
                    "success": metric_data.success,
                    "reason": metric_data.reason,
                    "evaluation_model": metric_data.evaluation_model,
                }
        case_results[test_result.name] = metrics
    return case_results


def per_metric_scores(case_results: dict[str, dict[str, dict]]) -> dict[str, list[float]]:
    """Pivot case results into per-metric score lists for aggregation."""
    per_metric: dict[str, list[float]] = {}
    for metrics in case_results.values():
        for name, data in metrics.items():
            per_metric.setdefault(name, []).append(data["score"])
    return per_metric


def write_scores(
    artifact: Path, records: list[dict], case_results: dict[str, dict[str, dict]]
) -> Path:
    """Per-question scores JSONL next to the run artifact, judge reasoning included."""
    path = artifact.with_suffix(".scores.jsonl")
    with path.open("w") as f:
        for i, r in enumerate(records):
            row = {
                "index": r.get("index", i),
                "question": r["question"],
                "otel_trace_id": r.get("otel_trace_id"),
                "otel_span_id": r.get("otel_span_id"),
                "metrics": case_results.get(_case_name(r, i), {}),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _metric_base_name(name: str) -> str:
    """deepeval reports GEval metrics as e.g. "AnswerCorrectness [GEval]"; the
    reported name is kept everywhere it is stored (scores files, summaries,
    Phoenix annotations — renaming would break continuity with old runs), but
    metric identity checks match on the base name."""
    return name.split(" [")[0]


# Not annotated on traces: per question these are just reciprocal rank / average
# precision — they only carry meaning as means over the whole run. They stay in the
# scores file and the summary.
_RUN_LEVEL_METRIC_PREFIXES = ("mrr@", "map@")

# Binary flag metrics where the score means "has this property", not pass/fail —
# label them by the property so Phoenix filtering reads naturally.
_FLAG_METRIC_LABELS = {"AnswerPartiallyCorrect": {True: "partial", False: "not-partial"}}


def build_annotations(
    records: list[dict], case_results: dict[str, dict[str, dict]]
) -> list[dict]:
    """Phoenix span-annotation payloads for the per-question-meaningful metrics.

    LLM-judged metrics (those reporting an evaluation model) get annotator_kind LLM,
    a pass/fail label, and the judge's reasoning as the explanation; deterministic
    metrics get CODE. Records without a span id (artifacts predating trace-id
    recording, or untraced runs) are skipped.
    """
    annotations = []
    for i, r in enumerate(records):
        span_id = r.get("otel_span_id")
        if not span_id:
            continue
        for metric_name, data in case_results.get(_case_name(r, i), {}).items():
            if metric_name.startswith(_RUN_LEVEL_METRIC_PREFIXES):
                continue
            llm_judged = bool(data.get("evaluation_model"))
            if _metric_base_name(metric_name) in _FLAG_METRIC_LABELS:
                label = _FLAG_METRIC_LABELS[_metric_base_name(metric_name)][bool(data["score"])]
            elif llm_judged:
                label = "pass" if data.get("success") else "fail"
            else:
                label = None
            annotations.append(
                {
                    "span_id": span_id,
                    "name": metric_name,
                    "annotator_kind": "LLM" if llm_judged else "CODE",
                    "result": {
                        "score": data["score"],
                        "label": label,
                        "explanation": data.get("reason"),
                    },
                }
            )
    return annotations


def push_annotations(annotations: list[dict], phoenix_endpoint: str) -> int:
    """POST annotations to Phoenix; span ids are global, so no project is needed.

    Phoenix upserts by (span_id, name), so re-scoring updates annotations in place.
    """
    if not annotations:
        return 0
    base_url = phoenix_endpoint.split("/v1/")[0]
    response = httpx.post(
        f"{base_url}/v1/span_annotations",
        json={"data": annotations},
        timeout=30.0,
    )
    response.raise_for_status()
    return len(annotations)


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


def scope_summary(scope_records: list[dict], case_results: dict[str, dict[str, dict]]) -> dict:
    """The scope block: overall handled-rate + per-category counts. n≈2-5 per
    category — read as buckets, gross signal only. avg_tool_calls is informative,
    unscored (searches burned on out-of-scope questions are waste)."""
    scores: list[float] = []
    per_category: dict[str, dict[str, int]] = {}
    for i, r in enumerate(scope_records):
        metrics = case_results.get(_case_name(r, i), {})
        data = next(
            (v for k, v in metrics.items() if _metric_base_name(k) == "ScopeHandling"), None
        )
        if data is None:
            continue
        scores.append(data["score"])
        bucket = per_category.setdefault(r["category"], {"handled": 0, "n": 0})
        bucket["n"] += 1
        bucket["handled"] += int(data["score"])
    return {
        "n": len(scope_records),
        "handled_rate": statistics.mean(scores) if scores else None,
        "per_category": dict(sorted(per_category.items())),
        "avg_tool_calls": (
            statistics.mean(len(r.get("tool_calls") or []) for r in scope_records)
            if scope_records else 0
        ),
    }


def _score_scope(scope_records: list[dict]) -> dict[str, dict[str, dict]]:
    """Evaluate ScopeHandling: one deepeval pass per (category, expected_behavior)
    group, each with criteria assembled for that group. Case names join back into
    the same case_results dict as the standard metrics."""
    from deepeval import evaluate

    from customer_agent.evaluation.answer_metrics import _judge_model
    from customer_agent.evaluation.scope_metrics import make_scope_metric

    model = _judge_model()
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in scope_records:
        groups.setdefault((r["category"], r["expected_behavior"]), []).append(r)
    case_results: dict[str, dict[str, dict]] = {}
    for (category, behavior), group in groups.items():
        cases = [
            LLMTestCase(
                name=_case_name(r, i),
                input=r["question"],
                actual_output=r["actual_answer"],
                expected_output=r["expected_answer"],
            )
            for i, r in enumerate(group)
        ]
        result = evaluate(
            test_cases=cases, metrics=[make_scope_metric(category, behavior, model)]
        )
        case_results.update(extract_case_results(result))
    return case_results


def score(artifact: Path) -> dict:
    """Full scoring pass: deepeval evaluate (judge LLM + deterministic metrics) ->
    per-question scores file + Phoenix annotations + summary."""
    from deepeval import evaluate

    from customer_agent.evaluation.answer_metrics import answer_metrics
    from customer_agent.evaluation.retrieval_metrics import retrieval_metrics
    from customer_agent.evaluation.runner import load_run

    from customer_agent.evaluation.runner import merge_ranked_article_ids

    settings = get_settings()
    records = load_run(artifact)
    # The merge rule is a scoring-time decision: recompute the article ranking from
    # the per-call rankings so --rescore applies the current rule to old artifacts
    # (the persisted retrieved_article_ids reflect whichever rule generation used).
    for r in records:
        if r.get("tool_calls"):
            r["retrieved_article_ids"] = merge_ranked_article_ids(
                [c["article_ids"] for c in r["tool_calls"]]
            )
    failed = [r for r in records if r.get("error")]
    if failed:
        print(
            f"WARNING: skipping {len(failed)} failed questions "
            f"(indices {[r.get('index') for r in failed]})",
            file=sys.stderr,
        )
        records = [r for r in records if not r.get("error")]
    standard_records, scope_records = split_by_behavior(records)
    case_results: dict[str, dict[str, dict]] = {}
    if standard_records:
        metrics = [*answer_metrics(), *retrieval_metrics(settings.metric_ks)]
        result = evaluate(test_cases=build_test_cases(standard_records), metrics=metrics)
        case_results.update(extract_case_results(result))
    if scope_records:
        case_results.update(_score_scope(scope_records))
    scores_path = write_scores(artifact, records, case_results)
    try:
        pushed = push_annotations(build_annotations(records, case_results),
                                  settings.phoenix_endpoint)
    except Exception as exc:  # Phoenix being down must not lose a paid scoring run
        print(f"WARNING: could not push annotations to Phoenix: {exc}", file=sys.stderr)
        pushed = 0

    per_metric = per_metric_scores(case_results)
    # ScopeHandling means live in the scope block; the metrics dict stays the
    # standard ladder, comparable to every historical run.
    per_metric = {
        name: scores for name, scores in per_metric.items()
        if _metric_base_name(name) != "ScopeHandling"
    }
    summary = summarize(artifact, records, per_metric)
    summary["scope"] = scope_summary(scope_records, case_results) if scope_records else None
    summary["scores_file"] = str(scores_path)
    summary["phoenix_annotations"] = pushed
    return summary
