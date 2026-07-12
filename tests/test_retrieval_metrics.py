import pytest
from deepeval.test_case import LLMTestCase

from customer_agent.evaluation.retrieval_metrics import (
    MAPAtK,
    MRRAtK,
    PrecisionAtK,
    RecallAtK,
    retrieval_metrics,
)


def case(gold: list[str], retrieved: list[str]) -> LLMTestCase:
    # Gold ids ride in metadata (context belongs to the judge; see module docstring).
    return LLMTestCase(
        input="q",
        actual_output="a",
        metadata={"gold_article_ids": gold},
        retrieval_context=retrieved,
    )


# gold = {A, B}; ranking = [X, A, B, Y, Z]
STANDARD = case(["A", "B"], ["X", "A", "B", "Y", "Z"])


@pytest.mark.parametrize(
    "metric,expected",
    [
        (PrecisionAtK(5), 2 / 5),
        (PrecisionAtK(2), 1 / 2),
        (RecallAtK(5), 1.0),
        (RecallAtK(2), 1 / 2),
        (MRRAtK(5), 1 / 2),
        (MRRAtK(1), 0.0),
        (MAPAtK(5), (1 / 2 + 2 / 3) / 2),
        (MAPAtK(2), (1 / 2) / 2),
    ],
)
def test_hand_computed_values(metric, expected):
    assert metric.measure(STANDARD) == pytest.approx(expected)


def test_perfect_ranking():
    tc = case(["A", "B"], ["A", "B", "X"])
    assert PrecisionAtK(2).measure(tc) == 1.0
    assert RecallAtK(3).measure(tc) == 1.0
    assert MRRAtK(3).measure(tc) == 1.0
    assert MAPAtK(3).measure(tc) == 1.0


def test_empty_retrieval_scores_zero():
    tc = case(["A"], [])
    for metric in [PrecisionAtK(5), RecallAtK(5), MRRAtK(5), MAPAtK(5)]:
        assert metric.measure(tc) == 0.0


def test_no_gold_scores_zero_not_crash():
    tc = case([], ["X", "Y"])
    for metric in [PrecisionAtK(5), RecallAtK(5), MRRAtK(5), MAPAtK(5)]:
        assert metric.measure(tc) == 0.0


def test_k_truncates_ranking():
    tc = case(["A"], ["X", "Y", "Z", "A"])  # gold at rank 4
    assert RecallAtK(3).measure(tc) == 0.0
    assert RecallAtK(4).measure(tc) == 1.0
    assert MRRAtK(4).measure(tc) == pytest.approx(1 / 4)


def test_map_denominator_capped_by_k():
    # 3 gold articles but K=2: perfect top-2 should score 1.0, not 2/3.
    tc = case(["A", "B", "C"], ["A", "B"])
    assert MAPAtK(2).measure(tc) == 1.0


def test_metric_names_and_success():
    metric = PrecisionAtK(5)
    assert metric.__name__ == "precision@5"
    metric.measure(STANDARD)
    assert metric.is_successful()  # threshold defaults to 0
    assert MAPAtK(10).__name__ == "map@10"


def test_factory_builds_all_metrics_for_each_k():
    metrics = retrieval_metrics((5, 10))
    names = {m.__name__ for m in metrics}
    assert names == {
        "precision@5", "recall@5", "mrr@5", "map@5",
        "precision@10", "recall@10", "mrr@10", "map@10",
    }


def test_a_measure_matches_measure():
    import asyncio

    sync_metric, async_metric = MAPAtK(5), MAPAtK(5)
    assert asyncio.run(async_metric.a_measure(STANDARD)) == sync_metric.measure(STANDARD)
