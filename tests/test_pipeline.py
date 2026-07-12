import customer_agent.retrieval.pipeline as pipeline_module
from customer_agent.config import Settings
from customer_agent.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from tests.conftest import make_chunk


def build_pipeline(candidates, k_retrieve=20, k_final=5, monkeypatch=None, settings=None):
    """RetrievalPipeline with fakes injected; never touches OpenAI/Weaviate."""

    class FakeEmbedder:
        def embed_query(self, text):
            return [0.1, 0.2]

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def search(self, query, vector, k):
            self.calls.append((query, vector, k))
            return candidates[:k]

    class ReverseReranker:
        def rerank(self, query, chunks):
            return list(reversed(chunks))

    pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
    pipeline.embedder = FakeEmbedder()
    pipeline.retriever = FakeRetriever()
    pipeline.reranker = ReverseReranker()
    if monkeypatch is not None:
        s = settings or Settings(k_retrieve=k_retrieve, k_final=k_final)
        monkeypatch.setattr(pipeline_module, "get_settings", lambda: s)
    return pipeline


def test_ranked_article_ids_dedupe_to_first_occurrence():
    result = RetrievalResult(
        query="q",
        ranked_chunks=[make_chunk("A", 0), make_chunk("B", 0), make_chunk("A", 1), make_chunk("C", 0)],
    )
    assert result.ranked_article_ids == ["A", "B", "C"]


def test_search_keeps_full_ranking_and_slices_tool_chunks(monkeypatch):
    candidates = [make_chunk(f"art{i}", score=1 - i / 10) for i in range(8)]
    pipeline = build_pipeline(candidates, k_retrieve=8, k_final=3, monkeypatch=monkeypatch)
    result = pipeline.search("some question")

    assert len(result.ranked_chunks) == 8  # full post-rerank ranking kept for eval
    assert len(result.tool_chunks) == 3    # agent only sees k_final
    # ReverseReranker reordered: tool chunks are the reranker's top, not the retriever's.
    assert [c.article_id for c in result.tool_chunks] == ["art7", "art6", "art5"]
    assert result.tool_chunks == result.ranked_chunks[:3]
    assert pipeline.retriever.calls[0][2] == 8  # retriever got k_retrieve
    assert pipeline.retriever.calls[0][0] == "some question"  # query text passed through (hybrid needs it)


def test_format_chunks_includes_title_url_and_text(monkeypatch):
    pipeline = build_pipeline([], monkeypatch=monkeypatch)
    result = RetrievalResult(query="q", ranked_chunks=[], tool_chunks=[make_chunk("A", 2)])
    output = pipeline.format_for_agent(result)
    assert "Article A" in output       # title
    assert "https://support.wix.com/A" in output
    assert "content of A part 2" in output


def test_format_empty_results(monkeypatch):
    pipeline = build_pipeline([], monkeypatch=monkeypatch)
    result = RetrievalResult(query="q", ranked_chunks=[])
    assert "No results" in pipeline.format_for_agent(result)


def test_format_articles_granularity_returns_full_articles(monkeypatch):
    settings = Settings(tool_output_granularity="articles")
    pipeline = build_pipeline([], monkeypatch=monkeypatch, settings=settings)
    kb = {
        "A": {"id": "A", "title": "Full Article A", "url": "u/A",
              "contents": "FULL BODY A", "article_type": "article"},
    }
    monkeypatch.setattr(pipeline_module, "kb_by_article_id", lambda: kb)

    # Two chunks of the same article must collapse into one full-article block.
    result = RetrievalResult(
        query="q", ranked_chunks=[], tool_chunks=[make_chunk("A", 0), make_chunk("A", 1)]
    )
    output = pipeline.format_for_agent(result)
    assert output.count("FULL BODY A") == 1
    assert "Full Article A" in output
