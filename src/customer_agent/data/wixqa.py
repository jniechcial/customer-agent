"""Loaders for the Wix/WixQA Hugging Face dataset.

QA subsets share the schema: question (str), answer (str, markdown), article_ids (list[str]).
KB corpus schema: id (str), url (str), contents (str, HTML-stripped text), article_type (str).
"""

from functools import lru_cache

from datasets import Dataset, load_dataset

from customer_agent.config import get_settings

QA_SUBSETS = ("wixqa_expertwritten", "wixqa_simulated", "wixqa_synthetic")
KB_SUBSET = "wix_kb_corpus"


def _load(config_name: str) -> Dataset:
    settings = get_settings()
    return load_dataset(
        settings.dataset_name,
        config_name,
        split="train",
        revision=settings.dataset_revision,
        token=settings.hf_token or None,
    )


@lru_cache(maxsize=4)
def load_qa(subset: str | None = None) -> Dataset:
    subset = subset or get_settings().qa_subset
    if subset not in QA_SUBSETS:
        raise ValueError(f"Unknown QA subset {subset!r}; expected one of {QA_SUBSETS}")
    return _load(subset)


@lru_cache(maxsize=1)
def load_kb() -> Dataset:
    return _load(KB_SUBSET)


@lru_cache(maxsize=1)
def kb_by_article_id() -> dict[str, dict]:
    """article_id -> full KB row; used when the tool returns whole articles."""
    return {row["id"]: row for row in load_kb()}
