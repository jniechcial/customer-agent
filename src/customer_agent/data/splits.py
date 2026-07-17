"""Deterministic train/validation split, computed at load time.

Same seed + same dataset revision => identical split. Nothing is persisted;
pin dataset_revision in config if upstream reproducibility matters.
"""

import random

from datasets import Dataset

from customer_agent.config import get_settings
from customer_agent.data.wixqa import load_extended, load_qa


def train_validation_split(subset: str | None = None) -> tuple[Dataset, Dataset]:
    settings = get_settings()
    ds = load_qa(subset)
    indices = list(range(len(ds)))
    random.Random(settings.split_seed).shuffle(indices)
    cut = int(len(indices) * settings.train_fraction)
    return ds.select(indices[:cut]), ds.select(indices[cut:])


def get_split(name: str, subset: str | None = None) -> Dataset:
    train, validation = train_validation_split(subset)
    if name == "train":
        return train
    if name == "validation":
        return validation
    raise ValueError(f"Unknown split {name!r}; expected 'train' or 'validation'")


def get_extended_split(name: str) -> list[dict]:
    """Extension rows for a split. Unlike the seeded standard split, extension
    rows carry an explicit split column (stratified by category — a seeded
    random split over n=40 can strand a whole category on one side)."""
    if name not in ("train", "validation"):
        raise ValueError(f"Unknown split {name!r}; expected 'train' or 'validation'")
    return [dict(row) for row in load_extended() if row["split"] == name]
