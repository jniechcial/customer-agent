"""Embedding client. Client-side by design: the embedding model is a config knob,
not baked into the Weaviate collection. Swap models by changing Settings.embedding_model
(which changes the collection name, so old indexes stay intact)."""

import time

from openai import OpenAI, RateLimitError

from customer_agent.config import get_settings


class OpenAIEmbedder:
    def __init__(self, model: str | None = None):
        settings = get_settings()
        self.model = model or settings.embedding_model
        self._client = OpenAI(api_key=settings.openai_api_key)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        settings = get_settings()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), settings.embed_batch_size):
            batch = texts[start : start + settings.embed_batch_size]
            for attempt in range(6):
                try:
                    response = self._client.embeddings.create(model=self.model, input=batch)
                    break
                except RateLimitError:
                    if attempt == 5:
                        raise
                    time.sleep(2**attempt)  # TPM limits reset within a minute
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


def get_default_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()
