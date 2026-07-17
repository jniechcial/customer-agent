"""Attach retrieval evidence to WixQA_Extended candidates (PLAN.md Phase 1, steps 2-3).

Runs every candidate that needs KB verification through the retrieval pipeline:
- out_of_kb: evidence that the KB does NOT answer the question (hits reviewed
  manually; candidates the KB answers get dropped)
- competitor_brand_gray / abusive_gray: evidence of which gold article(s) the
  question maps to (reference answers get written from those articles)

Reads extended/candidates.raw.jsonl, writes extended/evidence.jsonl.
Requires Weaviate up and OPENAI/VOYAGE keys in .env.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from customer_agent.retrieval.pipeline import close_pipeline, get_pipeline  # noqa: E402

BASE = Path(__file__).parent.parent / "extended"
IN_PATH = BASE / "candidates.raw.jsonl"
OUT_PATH = BASE / "evidence.jsonl"
VET_CATEGORIES = {"out_of_kb", "competitor_brand_gray", "abusive_gray"}


def main() -> None:
    rows = [json.loads(line) for line in IN_PATH.open()]
    to_vet = [r for r in rows if r["category"] in VET_CATEGORIES]
    pipeline = get_pipeline()
    try:
        with OUT_PATH.open("w") as f:
            for i, row in enumerate(to_vet, start=1):
                result = pipeline.search(row["question"])
                hits = [
                    {
                        "article_id": c.article_id,
                        "title": c.title,
                        "score": round(c.score, 3),
                        "snippet": c.text[:500],
                    }
                    for c in result.tool_chunks
                ]
                f.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "category": row["category"],
                            "question": row["question"],
                            "hits": hits,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(f"[{i}/{len(to_vet)}] {row['id']}")
    finally:
        close_pipeline()
    print(f"\nWrote evidence for {len(to_vet)} candidates to {OUT_PATH}")


if __name__ == "__main__":
    main()
