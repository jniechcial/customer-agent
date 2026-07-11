"""Index the WixQA knowledge base into Weaviate.

Usage:
    uv run python scripts/index_kb.py [--limit N]

The target collection is derived from chunking/embedding config (see
Settings.collection_name), so different index configs coexist in Weaviate.
Re-running recreates the collection for the CURRENT config only.
"""

import argparse

from customer_agent.config import get_settings
from customer_agent.tracing import setup_tracing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="index only the first N articles")
    args = parser.parse_args()

    setup_tracing("indexing")
    from customer_agent.indexing.indexer import index_kb

    settings = get_settings()
    print(f"indexing into collection {settings.collection_name!r} "
          f"(chunk={settings.chunk_size_tokens}/{settings.chunk_overlap_tokens}, "
          f"embed={settings.embedding_model})")
    name, count = index_kb(limit=args.limit)
    print(f"done: {count} chunks in {name!r}")


if __name__ == "__main__":
    main()
