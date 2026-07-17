"""Loaders for the Wix/WixQA Hugging Face dataset and the WixQA_Extended extension.

QA subsets share the schema: question (str), answer (str, markdown), article_ids (list[str]).
KB corpus schema: id (str), url (str), contents (str, HTML-stripped text), article_type (str).
Extension rows are a superset of the QA schema: + id (str), category (str),
expected_behavior (str), split (str, explicit train/validation assignment).
"""

import json
from functools import lru_cache
from pathlib import Path

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
def load_extended() -> tuple[dict, ...]:
    """WixQA_Extended rows (out-of-scope questions + gray traps), all splits.

    Loads the published HF dataset (revision-pinned); with
    extended_dataset_name="" falls back to the local JSONL built by
    scripts/build_extended.py (pre-upload iteration). Rows keep their explicit
    `split` column either way. Tuple for lru_cache safety.
    """
    settings = get_settings()
    if settings.extended_dataset_name:
        rows: list[dict] = []
        for split in ("train", "validation"):
            ds = load_dataset(
                settings.extended_dataset_name,
                split=split,
                revision=settings.extended_dataset_revision,
                token=settings.hf_token or None,
            )
            rows.extend(dict(r) for r in ds)
        return tuple(rows)
    path = Path(settings.extended_dataset_path)
    return tuple(
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    )


@lru_cache(maxsize=1)
def kb_by_article_id() -> dict[str, dict]:
    """article_id -> full KB row; used when the tool returns whole articles."""
    return {row["id"]: row for row in load_kb()}
