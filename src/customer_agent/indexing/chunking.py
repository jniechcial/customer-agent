"""Chunking strategies. This is a primary experimentation surface.

Implement the Chunker protocol to try alternatives:
  - heading/structure-aware splitting (KB articles have step-by-step sections)
  - semantic chunking (split on embedding-similarity valleys)
  - whole-article chunks (no splitting; pairs well with long-context models)
  - parent-document style (index small chunks, return bigger windows)
Changing chunker params changes Settings.collection_name, so each variant
gets its own Weaviate collection and they can be A/B'd without re-indexing.
"""

from dataclasses import dataclass
from typing import Protocol

import tiktoken

from customer_agent.config import get_settings


@dataclass(frozen=True)
class Chunk:
    article_id: str
    url: str
    title: str
    article_type: str
    chunk_index: int
    text: str


class Chunker(Protocol):
    def chunk_article(self, article: dict) -> list[Chunk]:
        """article is a KB corpus row: {id, url, contents, article_type}."""
        ...


class TokenChunker:
    """Default: fixed-size token windows with overlap.

    cl100k_base matches the text-embedding-3-* tokenizer closely enough for sizing.
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size_tokens
        self.overlap = overlap if overlap is not None else settings.chunk_overlap_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

    def chunk_article(self, article: dict) -> list[Chunk]:
        # Prefix the title so every chunk stays identifiable after splitting.
        body = f"{article['title']}\n\n{article['contents']}"
        tokens = self._enc.encode(body, disallowed_special=())
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        for i, start in enumerate(range(0, max(len(tokens), 1), step)):
            window = tokens[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(
                Chunk(
                    article_id=article["id"],
                    url=article["url"],
                    title=article["title"],
                    article_type=article["article_type"],
                    chunk_index=i,
                    text=self._enc.decode(window),
                )
            )
            if start + self.chunk_size >= len(tokens):
                break
        return chunks


def get_default_chunker() -> Chunker:
    return TokenChunker()
