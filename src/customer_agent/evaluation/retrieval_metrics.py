"""Deterministic retrieval metrics as deepeval custom metrics (no LLM involved).

Convention: LLMTestCase.additional_metadata["gold_article_ids"] holds the GOLD
article ids, and LLMTestCase.retrieval_context holds the RETRIEVED article ids
in rank order (article-level, deduped to first occurrence, merged across tool
calls). LLMTestCase.context belongs to the LLM judge (gold article texts), not
to these metrics.
"""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class _RankedRetrievalMetric(BaseMetric):
    metric_label = "base"

    # threshold=0 on purpose: per-case pass/fail is meaningless for ranked-retrieval
    # metrics (e.g. precision@10 is capped at 0.2 with 2 gold articles); we compare means.
    def __init__(self, k: int, threshold: float = 0.0):
        self.k = k
        self.threshold = threshold
        self.score: float | None = None
        self.success: bool | None = None
        self.reason: str | None = None
        self.evaluation_cost = 0.0

    def _compute(self, gold: set[str], retrieved: list[str]) -> float:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        gold = set((test_case.additional_metadata or {}).get("gold_article_ids", []))
        retrieved = list(test_case.retrieval_context or [])
        self.score = self._compute(gold, retrieved) if gold else 0.0
        self.success = self.score >= self.threshold
        self.reason = (
            f"{self.metric_label}@{self.k}={self.score:.3f} "
            f"(gold={sorted(gold)}, retrieved@{self.k}={retrieved[: self.k]})"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return f"{self.metric_label}@{self.k}"


class PrecisionAtK(_RankedRetrievalMetric):
    metric_label = "precision"

    def _compute(self, gold: set[str], retrieved: list[str]) -> float:
        top = retrieved[: self.k]
        if not top:
            return 0.0
        return sum(1 for a in top if a in gold) / len(top)


class RecallAtK(_RankedRetrievalMetric):
    metric_label = "recall"

    def _compute(self, gold: set[str], retrieved: list[str]) -> float:
        top = retrieved[: self.k]
        return sum(1 for a in gold if a in top) / len(gold)


class MRRAtK(_RankedRetrievalMetric):
    metric_label = "mrr"

    def _compute(self, gold: set[str], retrieved: list[str]) -> float:
        for rank, article_id in enumerate(retrieved[: self.k], start=1):
            if article_id in gold:
                return 1.0 / rank
        return 0.0


class MAPAtK(_RankedRetrievalMetric):
    metric_label = "map"

    def _compute(self, gold: set[str], retrieved: list[str]) -> float:
        top = retrieved[: self.k]
        hits = 0
        precision_sum = 0.0
        for rank, article_id in enumerate(top, start=1):
            if article_id in gold:
                hits += 1
                precision_sum += hits / rank
        denom = min(len(gold), self.k)
        return precision_sum / denom if denom else 0.0


def retrieval_metrics(ks: tuple[int, ...]) -> list[BaseMetric]:
    metrics: list[BaseMetric] = []
    for k in ks:
        metrics.extend([PrecisionAtK(k), RecallAtK(k), MRRAtK(k), MAPAtK(k)])
    return metrics
