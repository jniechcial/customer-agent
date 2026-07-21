"""KB corpus -> chunks -> vectors -> Weaviate collection."""

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.config import AdditionalConfig

from customer_agent.config import get_settings
from customer_agent.data.wixqa import load_kb
from customer_agent.indexing.chunking import Chunk, get_default_chunker
from customer_agent.retrieval.embeddings import get_default_embedder


def connect() -> weaviate.WeaviateClient:
    settings = get_settings()
    return weaviate.connect_to_local(
        host=settings.weaviate_http_host,
        port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
        additional_config=AdditionalConfig(trust_env=True),
        skip_init_checks=True,
    )


def recreate_collection(client: weaviate.WeaviateClient, name: str) -> None:
    if client.collections.exists(name):
        client.collections.delete(name)
    client.collections.create(
        name=name,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="article_id", data_type=DataType.TEXT),
            Property(name="url", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="article_type", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="text", data_type=DataType.TEXT),
        ],
    )


def index_kb(limit: int | None = None, batch_size: int = 256) -> tuple[str, int]:
    """Chunk, embed, and index the KB corpus. Returns (collection_name, chunk_count)."""
    settings = get_settings()
    chunker = get_default_chunker()
    embedder = get_default_embedder()

    kb = load_kb()
    if limit:
        kb = kb.select(range(limit))

    chunks: list[Chunk] = []
    for article in kb:
        chunks.extend(chunker.chunk_article(article))

    client = connect()
    try:
        recreate_collection(client, settings.collection_name)
        collection = client.collections.get(settings.collection_name)
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            vectors = embedder.embed_batch([c.text for c in batch_chunks])
            with collection.batch.fixed_size(batch_size=len(batch_chunks)) as batch:
                for chunk, vector in zip(batch_chunks, vectors):
                    batch.add_object(
                        properties={
                            "article_id": chunk.article_id,
                            "url": chunk.url,
                            "title": chunk.title,
                            "article_type": chunk.article_type,
                            "chunk_index": chunk.chunk_index,
                            "text": chunk.text,
                        },
                        vector=vector,
                    )
            failed = collection.batch.failed_objects
            if failed:
                raise RuntimeError(f"{len(failed)} objects failed to index: {failed[:3]}")
            print(f"indexed {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")
        total = len(collection)
    finally:
        client.close()
    return settings.collection_name, total
