import pytest
from datasets import Dataset

import customer_agent.data.splits as splits_module
from customer_agent.data.splits import get_split, train_validation_split


@pytest.fixture(autouse=True)
def fake_dataset(monkeypatch):
    rows = [
        {"question": f"q{i}", "answer": f"a{i}", "article_ids": [f"art{i}"]}
        for i in range(10)
    ]
    ds = Dataset.from_list(rows)
    monkeypatch.setattr(splits_module, "load_qa", lambda subset=None: ds)
    return ds


def test_split_is_50_50():
    train, val = train_validation_split()
    assert len(train) == 5
    assert len(val) == 5


def test_split_is_deterministic():
    t1, v1 = train_validation_split()
    t2, v2 = train_validation_split()
    assert [r["question"] for r in t1] == [r["question"] for r in t2]
    assert [r["question"] for r in v1] == [r["question"] for r in v2]


def test_split_halves_are_disjoint_and_complete():
    train, val = train_validation_split()
    train_qs = {r["question"] for r in train}
    val_qs = {r["question"] for r in val}
    assert train_qs.isdisjoint(val_qs)
    assert train_qs | val_qs == {f"q{i}" for i in range(10)}


def test_split_is_shuffled_not_positional():
    train, _ = train_validation_split()
    # Seeded shuffle should not just take the first half in order.
    assert [r["question"] for r in train] != [f"q{i}" for i in range(5)]


def test_get_split_names():
    train, val = train_validation_split()
    assert [r["question"] for r in get_split("train")] == [r["question"] for r in train]
    assert [r["question"] for r in get_split("validation")] == [r["question"] for r in val]


def test_get_split_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown split"):
        get_split("test")
