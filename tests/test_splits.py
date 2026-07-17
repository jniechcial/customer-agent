import pytest
from datasets import Dataset

import customer_agent.data.splits as splits_module
from customer_agent.data.splits import get_extended_split, get_split, train_validation_split


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


EXTENDED_ROWS = (
    {"id": "off_topic-01", "question": "eq0", "answer": "ea0", "article_ids": [],
     "category": "off_topic", "expected_behavior": "deflect_redirect", "split": "train"},
    {"id": "out_of_kb-01", "question": "eq1", "answer": "ea1", "article_ids": [],
     "category": "out_of_kb", "expected_behavior": "escalate_human", "split": "validation"},
    {"id": "abusive_gray-01", "question": "eq2", "answer": "ea2", "article_ids": ["g1"],
     "category": "abusive_gray", "expected_behavior": "answer_normally", "split": "train"},
)


@pytest.fixture
def fake_extended(monkeypatch):
    monkeypatch.setattr(splits_module, "load_extended", lambda: EXTENDED_ROWS)


def test_get_extended_split_filters_on_explicit_column(fake_extended):
    assert [r["id"] for r in get_extended_split("train")] == ["off_topic-01", "abusive_gray-01"]
    assert [r["id"] for r in get_extended_split("validation")] == ["out_of_kb-01"]


def test_get_extended_split_returns_copies(fake_extended):
    # load_extended is lru_cached; mutating a returned row must not poison the cache.
    get_extended_split("train")[0]["question"] = "mutated"
    assert get_extended_split("train")[0]["question"] == "eq0"


def test_get_extended_split_rejects_unknown_name(fake_extended):
    with pytest.raises(ValueError, match="Unknown split"):
        get_extended_split("test")
