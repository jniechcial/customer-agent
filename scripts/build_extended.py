"""Materialize the final extension dataset from the curated candidates file.

Usage:
    uv run python scripts/build_extended.py

Takes the `decision == "keep"` rows of extended/candidates.jsonl, drops the
curation-only fields, normalizes `article_ids` (";"-separated string -> list),
and writes extended/wixqa_extended.jsonl in the published schema. That file is
what the eval loads locally (config: extended_dataset_path) and what
upload_extended.py will push to HF once the set has proven useful.
"""

import json
import sys
from collections import Counter
from pathlib import Path

SOURCE = Path("extended/candidates.jsonl")
TARGET = Path("extended/wixqa_extended.jsonl")

EXPECTED_BEHAVIORS = {"answer_normally", "deflect_redirect", "refuse", "escalate_human"}
FIELDS = ("id", "question", "answer", "article_ids", "category", "expected_behavior", "split")


def build_rows() -> list[dict]:
    candidates = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    rows = []
    for candidate in candidates:
        if candidate["decision"] != "keep":
            continue
        row = {field: candidate[field] for field in FIELDS}
        row["article_ids"] = [a for a in candidate["article_ids"].split(";") if a]
        rows.append(row)
    return rows


def validate(rows: list[dict]) -> None:
    problems = []
    for row in rows:
        if row["expected_behavior"] not in EXPECTED_BEHAVIORS:
            problems.append(f"{row['id']}: unknown expected_behavior {row['expected_behavior']!r}")
        if row["split"] not in ("train", "validation"):
            problems.append(f"{row['id']}: unknown split {row['split']!r}")
        # Gray traps are normal QA rows and need gold articles; everything else
        # must have none (empty gold is what excludes them from retrieval means).
        if row["expected_behavior"] == "answer_normally" and not row["article_ids"]:
            problems.append(f"{row['id']}: answer_normally without gold article_ids")
        if row["expected_behavior"] != "answer_normally" and row["article_ids"]:
            problems.append(f"{row['id']}: {row['expected_behavior']} with gold article_ids")
        if not row["question"].strip() or not row["answer"].strip():
            problems.append(f"{row['id']}: empty question or answer")
    if len(set(r["id"] for r in rows)) != len(rows):
        problems.append("duplicate ids")
    if problems:
        sys.exit("validation failed:\n  " + "\n  ".join(problems))


def main() -> None:
    rows = build_rows()
    validate(rows)
    with TARGET.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_split = Counter(r["split"] for r in rows)
    by_category = Counter((r["category"], r["split"]) for r in rows)
    print(f"{len(rows)} rows -> {TARGET} ({dict(by_split)})")
    for (category, split), n in sorted(by_category.items()):
        print(f"  {category:<24} {split:<11} {n}")


if __name__ == "__main__":
    main()
