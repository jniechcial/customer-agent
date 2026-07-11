"""Run the eval: generate agent answers over a split, then score with deepeval.

Usage:
    uv run python scripts/run_eval.py --split validation [--subset wixqa_expertwritten]
                                      [--limit N] [--rescore runs/<id>.jsonl]

Two phases:
  1. generate — run the agent per question, persist runs/<run_id>.jsonl (skipped with --rescore)
  2. score    — deepeval: GEval answer correctness + deterministic retrieval metrics
Summary is printed and written to runs/<run_id>.summary.json.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from customer_agent.tracing import setup_tracing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--subset", default=None, help="QA subset (default from config)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--rescore", type=Path, default=None,
                        help="skip generation; score an existing runs/<id>.jsonl")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    if args.rescore:
        artifact = args.rescore
    else:
        import asyncio

        from customer_agent.evaluation.runner import generate_run

        run_name = args.run_id or f"{args.split}-{os.getpid()}"
        setup_tracing(f"eval-{run_name}")
        artifact = asyncio.run(
            generate_run(args.split, subset=args.subset, limit=args.limit, run_id=args.run_id)
        )
        print(f"generation done -> {artifact}")

    from customer_agent.evaluation.scoring import score

    summary = score(artifact)
    summary_path = artifact.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== {artifact.name}: n={summary['n']}, "
          f"avg tool calls={summary['avg_tool_calls']:.2f} ===")
    for name, stats in summary["metrics"].items():
        print(f"  {name:<24} {stats['mean']:.3f}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
