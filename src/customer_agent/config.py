"""Central configuration. Every experiment knob lives here, overridable via .env / env vars."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API keys ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    hf_token: str = ""
    voyage_api_key: str = ""

    # --- Models ---
    agent_model: str = "gpt-5.5"
    embedding_model: str = "text-embedding-3-small"
    judge_model: str = "claude-sonnet-5"

    # --- Dataset ---
    dataset_name: str = "Wix/WixQA"
    # pinned commit sha for full reproducibility (dataset snapshot of 2026-07-11)
    dataset_revision: str | None = "d662dc42479c14e202eccd832f8c4b66a035c4cc"
    qa_subset: str = "wixqa_expertwritten"
    split_seed: int = 42
    train_fraction: float = 0.5

    # --- Chunking / indexing ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    embed_batch_size: int = 128

    # --- Retrieval ---
    k_retrieve: int = 20  # candidates fetched from Weaviate
    k_final: int = 5      # returned to the agent after reranking
    search_mode: Literal["vector", "hybrid"] = "hybrid"
    hybrid_alpha: float = 0.5  # hybrid fusion weight: 1.0 = pure vector, 0.0 = pure BM25
    reranker: Literal["identity", "voyage"] = "voyage"
    rerank_model: str = "rerank-2.5"
    # What the tool hands to the agent: raw chunks or the full articles the hits belong to.
    tool_output_granularity: Literal["chunks", "articles"] = "chunks"

    # --- Infra ---
    weaviate_http_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051
    phoenix_endpoint: str = "http://localhost:6007/v1/traces"

    # --- Eval ---
    eval_concurrency: int = 4
    metric_ks: tuple[int, ...] = (3, 5)
    max_turns: int = 1  # single-turn eval for now; >1 once the synthetic user lands

    @property
    def reranker_id(self) -> str:
        """Reranker identity recorded in run artifacts so runs stay comparable."""
        return f"voyage:{self.rerank_model}" if self.reranker == "voyage" else self.reranker

    @property
    def search_mode_id(self) -> str:
        """Search-mode identity recorded in run artifacts so runs stay comparable."""
        return f"hybrid:a{self.hybrid_alpha}" if self.search_mode == "hybrid" else self.search_mode

    @property
    def collection_name(self) -> str:
        """Weaviate collection derived from index config so experiments coexist."""
        embed_tag = self.embedding_model.replace("text-embedding-", "te").replace("-", "")
        return f"KB_chunk{self.chunk_size_tokens}o{self.chunk_overlap_tokens}_{embed_tag}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
