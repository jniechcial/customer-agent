"""Reranking stage. This is the main experimentation surface.

The contract: take the query and the full candidate list from the retriever,
return the SAME candidates reordered (do not truncate — the pipeline slices
top-k_final for the agent and keeps the full ranking for eval metrics).

Implemented:
  - VoyageReranker: Voyage AI rerank API (rerank-2.5 by default).

Planned variants to implement here:
  - CrossEncoderReranker: local sentence-transformers cross-encoder
    (e.g. BAAI/bge-reranker-v2-m3); no API key, needs torch.
  - LLMReranker: listwise or pairwise prompting of a cheap LLM
    (e.g. "rank these passages by relevance to the query").
  - FusionReranker: reciprocal rank fusion over multiple retrievers
    (vector + BM25/hybrid) — pairs with retriever-side experiments.
  - MMR: maximal marginal relevance for diversity across articles.
"""

from dataclasses import replace
from typing import Protocol

import voyageai
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    RerankerAttributes,
    SpanAttributes,
)
from opentelemetry import trace

from customer_agent.config import get_settings
from customer_agent.retrieval.retriever import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class IdentityReranker:
    """Passthrough: keeps the retriever's similarity ordering."""

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


class VoyageReranker:
    """Voyage AI rerank API. Reorders the full candidate list; each returned
    chunk's `score` is the Voyage relevance score (replaces vector similarity —
    the two are not comparable)."""

    def __init__(self, model: str | None = None):
        settings = get_settings()
        self.model = model or settings.rerank_model
        self._client = voyageai.Client(api_key=settings.voyage_api_key, max_retries=3)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        tracer = trace.get_tracer("customer_agent.retrieval.reranker")
        with tracer.start_as_current_span("voyage.rerank") as span:
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RERANKER.value
            )
            span.set_attribute(RerankerAttributes.RERANKER_QUERY, query)
            span.set_attribute(RerankerAttributes.RERANKER_MODEL_NAME, self.model)
            _set_document_attributes(span, RerankerAttributes.RERANKER_INPUT_DOCUMENTS, chunks)

            response = self._client.rerank(
                query=query, documents=[chunk.text for chunk in chunks], model=self.model
            )
            reranked = [
                replace(chunks[result.index], score=result.relevance_score)
                for result in response.results
            ]

            _set_document_attributes(span, RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS, reranked)
        return reranked


def _set_document_attributes(span: trace.Span, prefix: str, chunks: list[RetrievedChunk]) -> None:
    for i, chunk in enumerate(chunks):
        span.set_attribute(
            f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_ID}",
            f"{chunk.article_id}#{chunk.chunk_index}",
        )
        span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_SCORE}", chunk.score)
        span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}", chunk.text)


def get_default_reranker() -> Reranker:
    if get_settings().reranker == "voyage":
        return VoyageReranker()
    return IdentityReranker()
