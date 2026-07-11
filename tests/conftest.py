import os

import pytest

# Never let tests touch external telemetry or real keys.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from customer_agent.retrieval.retriever import RetrievedChunk


def make_chunk(article_id: str, chunk_index: int = 0, score: float = 0.9, **overrides) -> RetrievedChunk:
    defaults = dict(
        article_id=article_id,
        url=f"https://support.wix.com/{article_id}",
        title=f"Article {article_id}",
        article_type="article",
        chunk_index=chunk_index,
        text=f"content of {article_id} part {chunk_index}",
        score=score,
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


@pytest.fixture
def chunk_factory():
    return make_chunk
