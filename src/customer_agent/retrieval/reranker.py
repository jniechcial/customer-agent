"""Reranking stage — STUB. This is the main experimentation surface.

The contract: take the query and the full candidate list from the retriever,
return the SAME candidates reordered (do not truncate — the pipeline slices
top-k_final for the agent and keeps the full ranking for eval metrics).

Planned variants to implement here:
  - CrossEncoderReranker: local sentence-transformers cross-encoder
    (e.g. BAAI/bge-reranker-v2-m3); no API key, needs torch.
  - APIReranker: Cohere Rerank / Jina Reranker endpoints; needs keys.
  - LLMReranker: listwise or pairwise prompting of a cheap LLM
    (e.g. "rank these passages by relevance to the query").
  - FusionReranker: reciprocal rank fusion over multiple retrievers
    (vector + BM25/hybrid) — pairs with retriever-side experiments.
  - MMR: maximal marginal relevance for diversity across articles.
"""

from typing import Protocol

from customer_agent.retrieval.retriever import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class IdentityReranker:
    """Passthrough: keeps the retriever's similarity ordering."""

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


def get_default_reranker() -> Reranker:
    return IdentityReranker()
