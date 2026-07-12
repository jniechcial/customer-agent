"""First-stage search against Weaviate: pure vector or hybrid (BM25 + vector),
selected by the `search_mode` config knob.

Future variants to experiment with (same interface):
  - query expansion / multi-query fusion
  - filtering by article_type
"""

from dataclasses import dataclass

from weaviate.classes.query import MetadataQuery

from customer_agent.config import get_settings
from customer_agent.indexing.indexer import connect


@dataclass
class RetrievedChunk:
    article_id: str
    url: str
    title: str
    article_type: str
    chunk_index: int
    text: str
    score: float  # higher is better; similarity (1 - cosine distance) or hybrid fusion score, until a reranker replaces it with its own relevance score


class WeaviateRetriever:
    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or get_settings().collection_name
        self._client = None

    def _collection(self):
        if self._client is None:
            self._client = connect()
        return self._client.collections.get(self.collection_name)

    def search(self, query: str, vector: list[float], k: int) -> list[RetrievedChunk]:
        settings = get_settings()
        if settings.search_mode == "hybrid":
            response = self._collection().query.hybrid(
                query=query,
                vector=vector,
                alpha=settings.hybrid_alpha,
                limit=k,
                return_metadata=MetadataQuery(score=True),
            )
            scores = [obj.metadata.score or 0.0 for obj in response.objects]
        else:
            response = self._collection().query.near_vector(
                near_vector=vector,
                limit=k,
                return_metadata=MetadataQuery(distance=True),
            )
            scores = [1.0 - (obj.metadata.distance or 0.0) for obj in response.objects]
        return [
            RetrievedChunk(
                article_id=obj.properties["article_id"],
                url=obj.properties["url"],
                title=obj.properties["title"],
                article_type=obj.properties["article_type"],
                chunk_index=obj.properties["chunk_index"],
                text=obj.properties["text"],
                score=score,
            )
            for obj, score in zip(response.objects, scores)
        ]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
