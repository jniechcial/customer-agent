"""Print the top-k retrieved articles for a query — inspect the retrieval pipeline without starting a chat.

Usage:
    uv run python scripts/search.py "your query here" [--k K]

--k overrides k_final from config; the pipeline still fetches k_retrieve candidates
and reranks all of them, so any k up to k_retrieve is valid.
"""

import argparse

from customer_agent.tracing import setup_tracing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="search query")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        metavar="K",
        help="number of articles to display (default: k_final from config)",
    )
    args = parser.parse_args()

    setup_tracing("search")
    from customer_agent.config import get_settings
    from customer_agent.retrieval.pipeline import close_pipeline, get_pipeline

    settings = get_settings()
    k = args.k if args.k is not None else settings.k_final

    pipeline = get_pipeline()
    try:
        result = pipeline.search(args.query)

        # Dedupe ranked_chunks to article level, take top-k.
        seen: set[str] = set()
        hits = []
        for chunk in result.ranked_chunks:
            if chunk.article_id not in seen:
                seen.add(chunk.article_id)
                hits.append(chunk)
            if len(hits) >= k:
                break

        print(f"query      : {args.query!r}")
        print(
            f"config     : collection={settings.collection_name}"
            f", search_mode={settings.search_mode}"
            f", reranker={settings.reranker}"
            f", k={k}"
        )
        print(f"candidates : {len(result.ranked_chunks)} chunks → {len(seen)} unique articles\n")

        for i, chunk in enumerate(hits, 1):
            print(f"[{i}] {chunk.title}")
            print(f"    article_id : {chunk.article_id}")
            print(f"    type       : {chunk.article_type}")
            print(f"    score      : {chunk.score:.4f}")
            print(f"    url        : {chunk.url}")
            print()
    finally:
        close_pipeline()


if __name__ == "__main__":
    main()
