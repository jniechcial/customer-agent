"""RetrievalPipeline: embed query -> vector search -> rerank -> format for the agent.

The pipeline keeps the FULL post-rerank ranking (k_retrieve candidates) for eval
metrics, while only the top k_final results are handed to the agent. That way
recall@10 etc. measure the pipeline, not just the tool-output slice.
"""

from dataclasses import dataclass, field

from customer_agent.config import get_settings
from customer_agent.data.wixqa import kb_by_article_id
from customer_agent.retrieval.embeddings import get_default_embedder
from customer_agent.retrieval.reranker import get_default_reranker
from customer_agent.retrieval.retriever import RetrievedChunk, WeaviateRetriever


@dataclass
class RetrievalResult:
    query: str
    ranked_chunks: list[RetrievedChunk]  # full post-rerank ranking (len ~ k_retrieve)
    tool_chunks: list[RetrievedChunk] = field(default_factory=list)  # top k_final, shown to agent

    @property
    def ranked_article_ids(self) -> list[str]:
        """Article-level ranking: chunks deduped to first occurrence. Eval consumes this."""
        seen: set[str] = set()
        ordered: list[str] = []
        for chunk in self.ranked_chunks:
            if chunk.article_id not in seen:
                seen.add(chunk.article_id)
                ordered.append(chunk.article_id)
        return ordered


class RetrievalPipeline:
    def __init__(self):
        self.embedder = get_default_embedder()
        self.retriever = WeaviateRetriever()
        self.reranker = get_default_reranker()

    def search(self, query: str) -> RetrievalResult:
        settings = get_settings()
        vector = self.embedder.embed_query(query)
        candidates = self.retriever.search(vector, k=settings.k_retrieve)
        ranked = self.reranker.rerank(query, candidates)
        return RetrievalResult(
            query=query,
            ranked_chunks=ranked,
            tool_chunks=ranked[: settings.k_final],
        )

    def format_for_agent(self, result: RetrievalResult) -> str:
        """Render tool output. Granularity knob: chunks (default) or full articles."""
        settings = get_settings()
        if not result.tool_chunks:
            return "No results found. Try a different query."

        blocks: list[str] = []
        if settings.tool_output_granularity == "articles":
            kb = kb_by_article_id()
            for article_id in RetrievalResult(
                query=result.query, ranked_chunks=result.tool_chunks
            ).ranked_article_ids:
                row = kb[article_id]
                blocks.append(
                    f"[{row['title']}] ({row['article_type']})\n"
                    f"URL: {row['url']}\n{row['contents']}"
                )
        else:
            for chunk in result.tool_chunks:
                blocks.append(
                    f"[{chunk.title} — part {chunk.chunk_index}] "
                    f"({chunk.article_type}, score {chunk.score:.3f})\n"
                    f"URL: {chunk.url}\n{chunk.text}"
                )
        return "\n\n---\n\n".join(blocks)


_pipeline: RetrievalPipeline | None = None


def get_pipeline() -> RetrievalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrievalPipeline()
    return _pipeline


def close_pipeline() -> None:
    """Close the singleton's Weaviate connection. Entry points call this on exit;
    the next get_pipeline() reconnects lazily."""
    global _pipeline
    if _pipeline is not None:
        _pipeline.retriever.close()
        _pipeline = None
